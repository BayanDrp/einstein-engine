use std::sync::Arc;

use crate::geometry::metric::{ExtrinsicCurvature, LapseField, ShiftField, SliceMetric};
use crate::geometry::tensor::{Tensor2, Vector};
use crate::grid::boundary::{apply_nonperiodic, BoundaryConfig, BoundaryValues};
use crate::grid::{Field, Grid};
use crate::physics::dust::dust_sources;
use crate::physics::dust::DustState;
use crate::physics::dust_evolution::{dust_rhs, DustRk4Stages};
use crate::relativity::evolution::{extrinsic_curvature_rhs_with_matter, spatial_metric_rhs};
use crate::relativity::gauge;
use crate::relativity::matter::{FourVelocityError, MatterSources};

#[derive(Debug, Clone)]
pub struct AdmState<const D: usize> {
    pub grid: Arc<Grid<D>>,
    pub gamma: SliceMetric<D>,
    pub k: ExtrinsicCurvature<D>,
    pub lapse: LapseField<D>,
    pub shift: ShiftField<D>,
}

/// The stage data of one classical RK4 ADM step: the four stage states
/// y0, y2 = y0 + dt/2 k1, y3 = y0 + dt/2 k2, y4 = y0 + dt k3 and the
/// rates k1..k4 evaluated on them (ki is the right-hand side of yi-1 and
/// of yi for i = 2..4). y0 carries the boundary conditions, the stage
/// stages y2..y4 are re-imposed to them.
///
/// This exists so advection-style couplings (e.g. geodesic test particles)
/// can sample the same geometry the ADM integrator saw at the same stage
/// times 0, dt/2, dt/2, dt instead of reconstructing it after the fact.
pub struct AdmRk4Stages<const D: usize> {
    pub dt: f64,
    pub y0: AdmState<D>,
    pub y2: AdmState<D>,
    pub y3: AdmState<D>,
    pub y4: AdmState<D>,
    pub k1: AdmState<D>,
    pub k2: AdmState<D>,
    pub k3: AdmState<D>,
    pub k4: AdmState<D>,
}

impl<const D: usize> AdmRk4Stages<D> {
    /// Collapse the stage data into Y_{n+1} = y0 + dt/6 k1 + dt/3 k2 +
    /// dt/3 k3 + dt/6 k4, with the boundary conditions re-imposed
    /// afterwards. Reproduces exactly the return value of
    /// `step_rk4_with_matter` on the same inputs.
    pub fn collapse(
        &self,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> AdmState<D> {
        let mut next = self.y0.clone();

        next.add_scaled_in_place(self.dt / 6.0, &self.k1);
        next.add_scaled_in_place(self.dt / 3.0, &self.k2);
        next.add_scaled_in_place(self.dt / 3.0, &self.k3);
        next.add_scaled_in_place(self.dt / 6.0, &self.k4);

        next.enforce_boundaries(boundary, boundary_values);

        next
    }
}

impl<const D: usize> AdmState<D> {
    /// Minkowski initial data:
    ///
    /// γ_ij = δ_ij
    /// K_ij = 0
    /// α = 1
    /// β^i = 0
    pub fn minkowski(grid: Arc<Grid<D>>) -> Self {
        let mut identity = Tensor2::<D>::zero();

        for i in 0..D {
            identity[(i, i)] = 1.0;
        }

        Self {
            gamma: Field::new(grid.clone(), identity),
            k: Field::new(grid.clone(), Tensor2::<D>::zero()),
            lapse: Field::new(grid.clone(), 1.0),
            shift: Field::new(grid.clone(), Vector::<D>::zero()),
            grid,
        }
    }

    /// Compute d(state)/dt.
    ///
    /// Gauge: 1+log slicing for the lapse, frozen shift.
    ///
    /// ∂t α = β^i ∂i α - 2 α K
    /// ∂t β^i = 0
    ///
    /// Vacuum specialization: no matter sources.
    pub fn rhs(&self, boundary: &BoundaryConfig<D>, boundary_values: &BoundaryValues<D>) -> Self {
        self.rhs_with_matter(
            &MatterSources::vacuum(&self.grid),
            boundary,
            boundary_values,
        )
    }

    /// Compute d(state)/dt with matter sources feeding the K equation.
    ///
    /// Gauge: 1+log slicing for the lapse, frozen shift.
    ///
    /// ∂t α = β^i ∂i α - 2 α K
    /// ∂t β^i = 0
    pub fn rhs_with_matter(
        &self,
        matter: &MatterSources<D>,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> Self {
        let gamma_rhs = spatial_metric_rhs(
            &self.lapse,
            &self.shift,
            &self.gamma,
            &self.k,
            boundary,
            boundary_values,
        );

        let k_rhs = extrinsic_curvature_rhs_with_matter(
            &self.lapse,
            &self.shift,
            &self.gamma,
            &self.k,
            matter,
            boundary,
            boundary_values,
        );

        Self {
            grid: self.grid.clone(),
            gamma: gamma_rhs,
            k: k_rhs,
            lapse: self.lapse_rhs(boundary, boundary_values),
            shift: self.shift_rhs(),
        }
    }

    /// 1+log slicing for the lapse:
    ///
    /// ∂t α = β^i ∂i α - 2 α K
    ///
    /// with K = γ^ij K_ij. Returns the time derivative of the lapse field.
    pub fn lapse_rhs(
        &self,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> LapseField<D> {
        gauge::lapse_rhs(
            &self.lapse,
            &self.shift,
            &self.gamma,
            &self.k,
            boundary,
            boundary_values,
        )
    }

    /// Time derivative of the shift.
    ///
    /// There is no shift equation yet: a Γ-driver needs ∂t Γ^i, i.e. third
    /// derivatives of the metric, so β^i stays put and this is always zero.
    pub fn shift_rhs(&self) -> ShiftField<D> {
        gauge::shift_rhs(&self.shift)
    }

    /// Re-impose the boundary conditions on the evolved state.
    ///
    /// RK4 mixes in derivatives taken across the whole grid, so after a step
    /// the face values no longer satisfy the configured conditions.
    ///
    /// Only the lapse is enforced here. `BoundaryValues` carries a single
    /// scalar per face, which describes a scalar field and nothing more:
    /// holding a vector to one value across all of its components, or a
    /// rank-2 field to one value across its components, is not a statement
    /// about the physics but a way to silently overwrite the state (a uniform
    /// Dirichlet value on the metric sets its off-diagonals equal to its
    /// diagonal and the metric becomes singular). The shift, metric and
    /// extrinsic curvature are therefore left alone until a per-component
    /// boundary type exists.
    ///
    /// Periodic faces are left to the wrapping stencils in every case, since
    /// `apply`'s endpoint averaging would identify the two ends and clip the
    /// field. Corners, where several faces meet, are written by whichever axis
    /// is handled last; with a single value per face they all agree anyway.
    fn enforce_boundaries(
        &mut self,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) {
        apply_nonperiodic(&mut self.lapse, boundary, boundary_values);
    }

    /// self + a * rhs
    pub fn axpy(&self, a: f64, rhs: &Self) -> Self {
        assert_eq!(self.grid.shape, rhs.grid.shape);

        let mut out = self.clone();
        out.add_scaled_in_place(a, rhs);

        out
    }

    /// Advance the full state by one RK4 timestep (vacuum).
    pub fn step_rk4(
        &self,
        dt: f64,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> Self {
        self.step_rk4_with_matter(
            dt,
            &MatterSources::vacuum(&self.grid),
            boundary,
            boundary_values,
        )
    }

    /// Advance the full state by one RK4 timestep (vacuum), returning
    /// the stage data instead of the collapsed next state.
    pub fn step_rk4_stages(
        &self,
        dt: f64,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> AdmRk4Stages<D> {
        self.step_rk4_with_matter_stages(
            dt,
            &MatterSources::vacuum(&self.grid),
            boundary,
            boundary_values,
        )
    }

    /// Advance the full state by one RK4 timestep with matter sources,
    /// returning the stage data instead of the collapsed next state.
    pub fn step_rk4_with_matter_stages(
        &self,
        dt: f64,
        matter: &MatterSources<D>,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> AdmRk4Stages<D> {
        assert!(dt.is_finite() && dt > 0.0);

        // Y_n, with the boundary conditions already satisfied, so every stage
        // starts from compliant data.
        let mut y0 = self.clone();
        y0.enforce_boundaries(boundary, boundary_values);

        // k1 = F(Y_n)
        let k1 = y0.rhs_with_matter(matter, boundary, boundary_values);

        // k2 = F(Y_n + dt/2 * k1)
        let mut y2 = y0.axpy(0.5 * dt, &k1);

        y2.enforce_boundaries(boundary, boundary_values);

        let k2 = y2.rhs_with_matter(matter, boundary, boundary_values);

        // k3 = F(Y_n + dt/2 * k2)
        let mut y3 = y0.axpy(0.5 * dt, &k2);

        y3.enforce_boundaries(boundary, boundary_values);

        let k3 = y3.rhs_with_matter(matter, boundary, boundary_values);

        // k4 = F(Y_n + dt * k3)
        let mut y4 = y0.axpy(dt, &k3);

        y4.enforce_boundaries(boundary, boundary_values);

        let k4 = y4.rhs_with_matter(matter, boundary, boundary_values);

        AdmRk4Stages {
            dt,
            y0,
            y2,
            y3,
            y4,
            k1,
            k2,
            k3,
            k4,
        }
    }

    /// Advance the full state by one RK4 timestep with matter sources.
    pub fn step_rk4_with_matter(
        &self,
        dt: f64,
        matter: &MatterSources<D>,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> Self {
        self.step_rk4_with_matter_stages(dt, matter, boundary, boundary_values)
            .collapse(boundary, boundary_values)
    }

    /// Advance geometry and dust together by one RK4 timestep, exposing
    /// the stage data instead of the collapsed states.
    ///
    /// This is the ADM counterpart of `BssnState::step_rk4_with_dust_stages`
    /// and closes the gap the frozen-source `step_rk4_with_matter` leaves
    /// for explicit matter: `step_rk4_with_matter` evaluates its
    /// `MatterSources` once, at Y_n, and feeds those same sources to all
    /// four stages -- fine while matter is much lighter than geometry, but
    /// not stage-consistent. Here every stage rebuilds the matter sources
    /// from the intermediate dust state (m_i) and the intermediate
    /// geometry (y_i), so neither system ever sees the other's stale
    /// values. Dust evolution itself lives in `physics::dust_evolution`.
    ///
    /// The two returned stage bundles are consistent with what RK4
    /// evaluated: geometry pairs (y_i, k_g i), dust pairs (m_i, k_m i),
    /// each dust rate carrying the geometry the dust actually saw at that
    /// stage. Geodesic couplings read the geometry side through
    /// `physics::geodesics::stage_connections` and matter consumers
    /// collapse the same stage set, so one step is one shared object.
    pub fn step_rk4_with_dust_stages(
        &self,
        dust: &DustState<D>,
        dt: f64,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> Result<(AdmRk4Stages<D>, DustRk4Stages<D>), FourVelocityError> {
        assert!(dt.is_finite() && dt > 0.0);

        // Y_n, with the boundary conditions already satisfied, so every
        // stage starts from compliant data.
        let mut y0 = self.clone();
        y0.enforce_boundaries(boundary, boundary_values);
        let m0 = dust.clone();

        // k1 = F(Y_n)
        let s1 = dust_sources::<4, D>(&m0, &y0.lapse, &y0.shift, &y0.gamma)?;
        let k1g = y0.rhs_with_matter(&s1, boundary, boundary_values);
        let k1m = dust_rhs(
            &m0,
            &y0.lapse,
            &y0.shift,
            &y0.gamma,
            boundary,
            boundary_values,
        );

        // k2 = F(Y_n + dt/2 * k1)
        let mut y2 = y0.axpy(0.5 * dt, &k1g);
        let m2 = m0.axpy(0.5 * dt, &k1m);
        y2.enforce_boundaries(boundary, boundary_values);
        let s2 = dust_sources::<4, D>(&m2, &y2.lapse, &y2.shift, &y2.gamma)?;
        let k2g = y2.rhs_with_matter(&s2, boundary, boundary_values);
        let k2m = dust_rhs(
            &m2,
            &y2.lapse,
            &y2.shift,
            &y2.gamma,
            boundary,
            boundary_values,
        );

        // k3 = F(Y_n + dt/2 * k2)
        let mut y3 = y0.axpy(0.5 * dt, &k2g);
        let m3 = m0.axpy(0.5 * dt, &k2m);
        y3.enforce_boundaries(boundary, boundary_values);
        let s3 = dust_sources::<4, D>(&m3, &y3.lapse, &y3.shift, &y3.gamma)?;
        let k3g = y3.rhs_with_matter(&s3, boundary, boundary_values);
        let k3m = dust_rhs(
            &m3,
            &y3.lapse,
            &y3.shift,
            &y3.gamma,
            boundary,
            boundary_values,
        );

        // k4 = F(Y_n + dt * k3)
        let mut y4 = y0.axpy(dt, &k3g);
        let m4 = m0.axpy(dt, &k3m);
        y4.enforce_boundaries(boundary, boundary_values);
        let s4 = dust_sources::<4, D>(&m4, &y4.lapse, &y4.shift, &y4.gamma)?;
        let k4g = y4.rhs_with_matter(&s4, boundary, boundary_values);
        let k4m = dust_rhs(
            &m4,
            &y4.lapse,
            &y4.shift,
            &y4.gamma,
            boundary,
            boundary_values,
        );

        Ok((
            AdmRk4Stages {
                dt,
                y0,
                y2,
                y3,
                y4,
                k1: k1g,
                k2: k2g,
                k3: k3g,
                k4: k4g,
            },
            DustRk4Stages {
                dt,
                m0,
                m2,
                m3,
                m4,
                k1: k1m,
                k2: k2m,
                k3: k3m,
                k4: k4m,
            },
        ))
    }

    /// Advance geometry and dust together by one RK4 timestep.
    ///
    /// The collapse of [`AdmState::step_rk4_with_dust_stages`]: geometry
    /// via [`AdmRk4Stages::collapse`] (boundary conditions re-imposed,
    /// the same sequential accumulation `step_rk4_with_matter` always
    /// performed), dust via [`DustRk4Stages::collapse`] (no enforcement
    /// step -- conserved densities carry no algebraic constraints).
    pub fn step_rk4_with_dust(
        &self,
        dust: &DustState<D>,
        dt: f64,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> Result<(Self, DustState<D>), FourVelocityError> {
        let (stages, dust_stages) =
            self.step_rk4_with_dust_stages(dust, dt, boundary, boundary_values)?;

        Ok((
            stages.collapse(boundary, boundary_values),
            dust_stages.collapse(),
        ))
    }

    fn add_scaled_in_place(&mut self, a: f64, rhs: &Self) {
        self.grid.for_each_index(|idx| {
            let mut gamma = *self.gamma.get(idx);
            let gamma_rhs = *rhs.gamma.get(idx);

            for i in 0..D {
                for j in 0..D {
                    gamma[(i, j)] += a * gamma_rhs[(i, j)];
                }
            }

            self.gamma.set(idx, gamma);

            let mut k = *self.k.get(idx);
            let k_rhs = *rhs.k.get(idx);

            for i in 0..D {
                for j in 0..D {
                    k[(i, j)] += a * k_rhs[(i, j)];
                }
            }

            self.k.set(idx, k);

            let lapse_rhs = *rhs.lapse.get(idx);
            let lapse = *self.lapse.get(idx);

            self.lapse.set(idx, lapse + a * lapse_rhs);

            let mut shift = *self.shift.get(idx);
            let shift_rhs = *rhs.shift.get(idx);

            for i in 0..D {
                shift[i] += a * shift_rhs[i];
            }

            self.shift.set(idx, shift);
        });
    }
}
