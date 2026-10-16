//! Pressureless dust: conserved density plus conserved momentum, with exact
//! primitive recovery (no iteration — dust has no pressure).
//!
//! Evolved (conservative) variables, stored on [`DustState`]:
//!   Dtilde = sqrt(det gamma) rho W            (conserved rest-mass density)
//!   Ptilde_j = sqrt(det gamma) S_j            (densitized momentum,
//!                                              S_j = rho W^2 v_j)
//! Primitive variables, recovered exactly from (Dtilde, Ptilde_j, gamma):
//!   R = gamma^{jk} S_j S_k / D^2,  v^2 = R / (1 + R),  W = sqrt(1 + R),
//!   v_j = S_j / (D W),  v^i = gamma^{ij} v_j,  rho = D / W,
//! with D = Dtilde / sqrt(det gamma), S_j = Ptilde_j / sqrt(det gamma).
//! Since v^2 = R/(1+R) < 1 whenever D > 0, recovery can never yield a
//! superluminal velocity; points with Dtilde <= 0 are treated as vacuum
//! (rho = 0, v = 0). That vacuum floor is the only limiter in the scheme.
//!
//! Time evolution lives in [`crate::physics::dust_evolution`]; this module
//! holds the state representation, the primitive/conserved conversions,
//! and the 3+1 source construction.
//!
//! NOT implemented: pressure/EOS, magnetic fields, primitive recovery by
//! iteration (unneeded — analytic here).
//!
//! Caustics: pressureless dust forms shell-crossing singularities where
//! trajectories intersect; a single-valued velocity field cannot represent
//! them, and gradients will grow without bound. This scheme is valid only
//! for smooth flows before caustic formation.

use std::sync::Arc;

use crate::geometry::metric::{LapseField, ShiftField, SliceMetric};
use crate::geometry::tensor::{det, invert, Vector};
use crate::grid::{Field, Grid};
use crate::physics::stress_energy::dust_stress_energy_field;
use crate::relativity::matter::{
    four_velocity_covariant_field, matter_sources, FourVelocityError, MatterSources,
};

/// Rest-mass density rho: mass per proper volume in the fluid rest frame.
pub type RestMassDensity<const D: usize> = Field<f64, D>;

/// Eulerian 3-velocity v^i measured by normal observers.
pub type DustVelocity<const D: usize> = Field<Vector<D>, D>;

/// Dust state: conserved density plus conserved momentum.
///
/// `densitized_rest_mass` is Dtilde = sqrt(det gamma) rho W.
/// `densitized_momentum` is Ptilde_j = sqrt(det gamma) S_j with
/// S_j = rho W^2 v_j. Primitives (rho, v^i) are recovered exactly via
/// `rest_density` / `velocity`.
#[derive(Debug, Clone)]
pub struct DustState<const D: usize> {
    pub grid: Arc<Grid<D>>,
    pub densitized_rest_mass: Field<f64, D>,
    pub densitized_momentum: Field<Vector<D>, D>,
}

impl<const D: usize> DustState<D> {
    /// Build conserved (Dtilde, Ptilde_j) from primitive rest density and
    /// velocity.
    ///
    /// Requires a positive-definite metric and subluminal velocity.
    pub fn from_primitives(
        rho: &RestMassDensity<D>,
        velocity: &DustVelocity<D>,
        spatial: &SliceMetric<D>,
    ) -> Self {
        assert_eq!(velocity.grid.shape, rho.grid.shape);
        assert_eq!(spatial.grid.shape, rho.grid.shape);

        let mut conserved = Field::new(rho.grid.clone(), 0.0);
        let mut momentum = Field::new(rho.grid.clone(), Vector::<D>::zero());
        rho.grid.for_each_index(|idx| {
            let gamma = spatial.get(idx);
            let vel = velocity.get(idx);

            let mut gmat = [[0.0f64; D]; D];
            for i in 0..D {
                for j in 0..D {
                    gmat[i][j] = gamma[(i, j)];
                }
            }
            let gdet = det(&gmat);
            assert!(gdet > 0.0, "the spatial metric must be positive definite");

            let mut v_squared = 0.0;
            for i in 0..D {
                for j in 0..D {
                    v_squared += vel[i] * gamma[(i, j)] * vel[j];
                }
            }
            assert!(
                v_squared < 1.0,
                "dust velocity must stay subluminal"
            );

            let w = 1.0 / (1.0 - v_squared).sqrt();
            conserved.set(idx, gdet.sqrt() * rho.get(idx) * w);

            // Ptilde_j = sqrt(det gamma) rho W^2 v_j
            let mut mom = Vector::<D>::zero();
            for i in 0..D {
                let mut v_low = 0.0;
                for j in 0..D {
                    v_low += gamma[(i, j)] * vel[j];
                }
                mom[i] = gdet.sqrt() * rho.get(idx) * w * w * v_low;
            }
            momentum.set(idx, mom);
        });

        Self {
            grid: rho.grid.clone(),
            densitized_rest_mass: conserved,
            densitized_momentum: momentum,
        }
    }

    /// Recover primitives (rho, v^i) from (Dtilde, Ptilde_j, gamma).
    ///
    /// Exact algebra, no iteration. Dtilde <= 0 is treated as vacuum
    /// (rho = 0, v = 0); otherwise v^2 = R/(1+R) < 1 always.
    /// Crate-visible for the evolution module; external users go through
    /// `rest_density` / `velocity`.
    pub(crate) fn primitives(
        &self,
        spatial: &SliceMetric<D>,
    ) -> (RestMassDensity<D>, DustVelocity<D>) {
        assert_eq!(spatial.grid.shape, self.grid.shape);

        let mut rho = Field::new(self.grid.clone(), 0.0);
        let mut vel = Field::new(self.grid.clone(), Vector::<D>::zero());
        self.grid.for_each_index(|idx| {
            let gamma = spatial.get(idx);

            let mut gmat = [[0.0f64; D]; D];
            for i in 0..D {
                for j in 0..D {
                    gmat[i][j] = gamma[(i, j)];
                }
            }
            let gdet = det(&gmat);
            assert!(gdet > 0.0, "the spatial metric must be positive definite");
            let sqrtg = gdet.sqrt();
            let ginv = invert(&gmat);

            let d_tilde = *self.densitized_rest_mass.get(idx);
            let mom = *self.densitized_momentum.get(idx);

            if d_tilde <= 0.0 {
                rho.set(idx, 0.0);
                vel.set(idx, Vector::<D>::zero());
                return;
            }

            // D = Dtilde / sqrt(det gamma), S_j = Ptilde_j / sqrt(det gamma)
            let dens = d_tilde / sqrtg;
            let mut s_cov = [0.0f64; D];
            for j in 0..D {
                s_cov[j] = mom[j] / sqrtg;
            }

            // R = gamma^{jk} S_j S_k / D^2, v^2 = R / (1 + R), W = sqrt(1 + R)
            let mut r = 0.0;
            for j in 0..D {
                for k in 0..D {
                    r += ginv[j][k] * s_cov[j] * s_cov[k];
                }
            }
            r /= dens * dens;
            let w = (1.0 + r).sqrt();

            // v^i = gamma^{ij} S_j / (D W)
            let mut v_up = Vector::<D>::zero();
            for i in 0..D {
                for j in 0..D {
                    v_up[i] += ginv[i][j] * s_cov[j] / (dens * w);
                }
            }
            rho.set(idx, dens / w);
            vel.set(idx, v_up);
        });
        (rho, vel)
    }

    /// Recover the primitive rest density rho = Dtilde / (sqrt(det gamma) W).
    pub fn rest_density(&self, spatial: &SliceMetric<D>) -> RestMassDensity<D> {
        self.primitives(spatial).0
    }

    /// Recover the Eulerian velocity v^i from (Dtilde, Ptilde_j, gamma).
    pub fn velocity(&self, spatial: &SliceMetric<D>) -> DustVelocity<D> {
        self.primitives(spatial).1
    }

    /// Covariant four-velocity fields from the current velocity and geometry.
    pub fn four_velocity(
        &self,
        lapse: &LapseField<D>,
        shift: &ShiftField<D>,
        spatial: &SliceMetric<D>,
    ) -> Result<(Field<f64, D>, Field<Vector<D>, D>), FourVelocityError> {
        let vel = self.velocity(spatial);
        four_velocity_covariant_field(lapse, shift, spatial, &vel)
    }

    /// self + a * rhs
    pub fn axpy(&self, a: f64, rhs: &Self) -> Self {
        assert_eq!(self.grid.shape, rhs.grid.shape);

        let mut out = self.clone();
        out.add_scaled_in_place(a, rhs);

        out
    }

    pub(crate) fn add_scaled_in_place(&mut self, a: f64, rhs: &Self) {
        self.grid.for_each_index(|idx| {
            self.densitized_rest_mass.set(
                idx,
                *self.densitized_rest_mass.get(idx) + a * *rhs.densitized_rest_mass.get(idx),
            );

            let mut mom = *self.densitized_momentum.get(idx);
            let mom_rhs = *rhs.densitized_momentum.get(idx);
            for i in 0..D {
                mom[i] += a * mom_rhs[i];
            }
            self.densitized_momentum.set(idx, mom);
        });
    }
}

/// 3+1 matter sources built from a dust state and the current geometry.
///
/// Returns an error if any point has a non-positive lapse or a
/// superluminal velocity.
pub fn dust_sources<const N: usize, const D: usize>(
    dust: &DustState<D>,
    lapse: &LapseField<D>,
    shift: &ShiftField<D>,
    spatial: &SliceMetric<D>,
) -> Result<MatterSources<D>, FourVelocityError> {
    let rho = dust.rest_density(spatial);
    let (u_0, u_cov) = dust.four_velocity(lapse, shift, spatial)?;
    let t = dust_stress_energy_field::<N, D>(&rho, &u_0, &u_cov);
    Ok(matter_sources::<N, D>(
        &t,
        lapse,
        shift,
        spatial,
    ))
}
