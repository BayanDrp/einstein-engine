use std::sync::Arc;

use crate::geometry::metric::{LapseField, ShiftField, SliceMetric};
use crate::geometry::tensor::{invert, Tensor2, Vector};
use crate::grid::{Field, Grid};
use crate::physics::stress_energy::StressEnergyField;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FourVelocityError {
    NonFiniteInput,
    InvalidLapse,
    InvalidSpeed,
}

/// Build the covariant four-velocity (u_0, u_i) for any spatial dimension.
///
/// velocity: physical 3-velocity v^i measured by Eulerian observers.
/// alpha: lapse (must be positive).
/// beta: contravariant shift vector beta^i.
/// spatial_metric: gamma_ij.
///
/// Returns the time component u_0 alongside the spatial covariant components
/// u_i, split because a D+1-sized array is not expressible with stable
/// const-generics.
///
/// Uses c = 1 and the (-+++) metric signature, so that u^mu u_mu = -1 with
/// u^0 = W / alpha and u^i = W (v^i - beta^i / alpha).
pub fn four_velocity_covariant<const D: usize>(
    alpha: f64,
    beta: &Vector<D>,
    spatial_metric: &Tensor2<D>,
    velocity: &Vector<D>,
) -> Result<(f64, Vector<D>), FourVelocityError> {
    if !alpha.is_finite() {
        return Err(FourVelocityError::NonFiniteInput);
    }
    for i in 0..D {
        if !beta[i].is_finite() || !velocity[i].is_finite() {
            return Err(FourVelocityError::NonFiniteInput);
        }
        for j in 0..D {
            if !spatial_metric[(i, j)].is_finite() {
                return Err(FourVelocityError::NonFiniteInput);
            }
        }
    }

    if alpha <= 0.0 {
        return Err(FourVelocityError::InvalidLapse);
    }

    // Lower the spatial velocity index:
    // v_i = gamma_ij * v^j
    let mut v_cov: Vector<D> = Vector::zero();

    for i in 0..D {
        for j in 0..D {
            v_cov[i] += spatial_metric[(i, j)] * velocity[j];
        }
    }

    // v^2 = gamma_ij * v^i * v^j
    let mut v_squared = 0.0;

    for i in 0..D {
        v_squared += velocity[i] * v_cov[i];
    }

    if !v_squared.is_finite()
        || !(0.0..1.0).contains(&v_squared)
    {
        return Err(FourVelocityError::InvalidSpeed);
    }

    // Lorentz factor
    let w = 1.0 / (1.0 - v_squared).sqrt();

    // beta^i * v_i
    let mut beta_dot_v = 0.0;

    for i in 0..D {
        beta_dot_v += beta[i] * v_cov[i];
    }

    // Covariant four-velocity (u_0, u_i)
    let u_0 = w * (-alpha + beta_dot_v);
    let mut u_cov: Vector<D> = Vector::zero();

    for i in 0..D {
        u_cov[i] = w * v_cov[i];
    }

    Ok((u_0, u_cov))
}

/// Build the covariant four-velocity (u_0, u_i) on the whole grid.
///
/// Applies `four_velocity_covariant` pointwise; the first failing point
/// aborts the field build and its error is returned.
pub fn four_velocity_covariant_field<const D: usize>(
    alpha: &Field<f64, D>,
    beta: &Field<Vector<D>, D>,
    spatial_metric: &Field<Tensor2<D>, D>,
    velocity: &Field<Vector<D>, D>,
) -> Result<(Field<f64, D>, Field<Vector<D>, D>), FourVelocityError> {
    assert_eq!(beta.grid.shape, alpha.grid.shape);
    assert_eq!(spatial_metric.grid.shape, alpha.grid.shape);
    assert_eq!(velocity.grid.shape, alpha.grid.shape);

    let mut u_0 = Field::new(alpha.grid.clone(), 0.0);
    let mut u_cov = Field::new(alpha.grid.clone(), Vector::zero());
    let mut err: Option<FourVelocityError> = None;

    alpha.grid.for_each_index(|idx| {
        if err.is_some() {
            return;
        }
        match four_velocity_covariant(
            *alpha.get(idx),
            beta.get(idx),
            spatial_metric.get(idx),
            velocity.get(idx),
        ) {
            Ok((t, s)) => {
                u_0.set(idx, t);
                u_cov.set(idx, s);
            }
            Err(e) => err = Some(e),
        }
    });

    match err {
        Some(e) => Err(e),
        None => Ok((u_0, u_cov)),
    }
}

/// 3+1 matter sources: the projections of T_mu_nu that the constraints and
/// evolution equations consume.
///
/// E / rho: energy density measured by Eulerian observers; enters H.
/// S^i: contravariant momentum density; enters M^i directly.
/// S_ij: spatial stress; enters the K / A-tilde evolution.
/// S: stress trace gamma^ij S_ij; enters d_t K.
#[derive(Debug, Clone)]
pub struct MatterSources<const D: usize> {
    pub rho: Field<f64, D>,
    pub momentum: Field<Vector<D>, D>,
    pub stress: Field<Tensor2<D>, D>,
    pub stress_trace: Field<f64, D>,
}

impl<const D: usize> MatterSources<D> {
    /// Vacuum (identically zero) sources on the given grid: every evolution
    /// equation below reduces to its vacuum form exactly.
    pub fn vacuum(grid: &Arc<Grid<D>>) -> Self {
        Self {
            rho: Field::new(grid.clone(), 0.0),
            momentum: Field::new(grid.clone(), Vector::zero()),
            stress: Field::new(grid.clone(), Tensor2::zero()),
            stress_trace: Field::new(grid.clone(), 0.0),
        }
    }
}

/// Project T_mu_nu onto the slice (c = 1, -+++ signature):
///
///   rho = (T_00 - 2 beta^i T_0i + beta^i beta^j T_ij) / alpha^2
///   s_i = -(T_0i - beta^j T_ij) / alpha
///   S^i = gamma^ij s_j
///   S_ij = T_ij
///   S = gamma^ij S_ij
pub fn matter_sources<const N: usize, const D: usize>(
    stress_energy: &StressEnergyField<N, D>,
    lapse: &LapseField<D>,
    shift: &ShiftField<D>,
    spatial: &SliceMetric<D>,
) -> MatterSources<D> {
    assert!(N == D + 1, "spacetime dim N must be D + 1, got N = {N}, D = {D}");
    assert_eq!(lapse.grid.shape, spatial.grid.shape);
    assert_eq!(shift.grid.shape, spatial.grid.shape);
    assert_eq!(stress_energy.grid.shape, spatial.grid.shape);

    let mut rho = Field::new(spatial.grid.clone(), 0.0);
    let mut momentum = Field::new(spatial.grid.clone(), Vector::zero());
    let mut stress = Field::new(spatial.grid.clone(), Tensor2::zero());
    let mut stress_trace = Field::new(spatial.grid.clone(), 0.0);

    spatial.grid.for_each_index(|idx| {
        let t = stress_energy.get(idx);
        let alpha = *lapse.get(idx);
        let beta = *shift.get(idx);
        let gamma = *spatial.get(idx);

        let mut gmat = [[0.0f64; D]; D];
        for i in 0..D {
            for j in 0..D {
                gmat[i][j] = gamma[(i, j)];
            }
        }
        let ginv = invert(&gmat);

        // E: rho = (T_00 - 2 b^i T_0i + b^i b^j T_ij) / a^2
        let mut r = t.data[0][0];
        for i in 0..D {
            r -= 2.0 * beta[i] * t.data[0][i + 1];
        }
        for i in 0..D {
            for j in 0..D {
                r += beta[i] * beta[j] * t.data[i + 1][j + 1];
            }
        }
        rho.set(idx, r / (alpha * alpha));

        // s_i = -(T_0i - b^j T_ij) / a, then S^i = g^ij s_j
        let mut s_cov = [0.0f64; D];
        for i in 0..D {
            let mut s = t.data[0][i + 1];
            for j in 0..D {
                s -= beta[j] * t.data[i + 1][j + 1];
            }
            s_cov[i] = -s / alpha;
        }
        let mut mom = Vector::<D>::zero();
        for i in 0..D {
            for j in 0..D {
                mom[i] += ginv[i][j] * s_cov[j];
            }
        }
        momentum.set(idx, mom);

        // S_ij = T_ij, S = g^ij S_ij
        let mut sij = Tensor2::<D>::zero();
        let mut tr = 0.0;
        for i in 0..D {
            for j in 0..D {
                sij[(i, j)] = t.data[i + 1][j + 1];
                tr += ginv[i][j] * t.data[i + 1][j + 1];
            }
        }
        stress.set(idx, sij);
        stress_trace.set(idx, tr);
    });

    MatterSources {
        rho,
        momentum,
        stress,
        stress_trace,
    }
}