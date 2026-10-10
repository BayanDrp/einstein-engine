use std::sync::Arc;

use crate::geometry::metric::{
    ExtrinsicCurvature,
    LapseField,
    ShiftField,
    SliceMetric,
};
use crate::geometry::tensor::{Tensor2, Vector};
use crate::grid::boundary::{apply_nonperiodic, BoundaryConfig, BoundaryValues};
use crate::grid::{Field, Grid};
use crate::relativity::evolution::{
    extrinsic_curvature_rhs,
    spatial_metric_rhs,
};
use crate::relativity::gauge;

#[derive(Debug, Clone)]
pub struct EvolutionState<const D: usize> {
    pub grid: Arc<Grid<D>>,
    pub gamma: SliceMetric<D>,
    pub k: ExtrinsicCurvature<D>,
    pub lapse: LapseField<D>,
    pub shift: ShiftField<D>,
}

impl<const D: usize> EvolutionState<D> {
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
            k: Field::new(
                grid.clone(),
                Tensor2::<D>::zero(),
            ),
            lapse: Field::new(grid.clone(), 1.0),
            shift: Field::new(
                grid.clone(),
                Vector::<D>::zero(),
            ),
            grid,
        }
    }

    /// Compute d(state)/dt.
    ///
    /// Gauge: 1+log slicing for the lapse, frozen shift.
    ///
    /// ∂t α = β^i ∂i α - 2 α K
    /// ∂t β^i = 0
    pub fn rhs(
        &self,
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

        let k_rhs = extrinsic_curvature_rhs(
            &self.lapse,
            &self.shift,
            &self.gamma,
            &self.k,
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

    /// Advance the full state by one RK4 timestep.
    pub fn step_rk4(
        &self,
        dt: f64,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> Self {
        assert!(dt.is_finite() && dt > 0.0);

        // Y_n, with the boundary conditions already satisfied, so every stage
        // starts from compliant data.
        let mut y0 = self.clone();
        y0.enforce_boundaries(boundary, boundary_values);

        // k1 = F(Y_n)
        let k1 = y0.rhs(boundary, boundary_values);

        // k2 = F(Y_n + dt/2 * k1)
        let mut y2 = y0.axpy(0.5 * dt, &k1);

        y2.enforce_boundaries(boundary, boundary_values);

        let k2 = y2.rhs(boundary, boundary_values);

        // k3 = F(Y_n + dt/2 * k2)
        let mut y3 = y0.axpy(0.5 * dt, &k2);

        y3.enforce_boundaries(boundary, boundary_values);

        let k3 = y3.rhs(boundary, boundary_values);

        // k4 = F(Y_n + dt * k3)
        let mut y4 = y0.axpy(dt, &k3);

        y4.enforce_boundaries(boundary, boundary_values);

        let k4 = y4.rhs(boundary, boundary_values);

        // Y_{n+1}
        let mut next = y0;

        next.add_scaled_in_place(
            dt / 6.0,
            &k1,
        );

        next.add_scaled_in_place(
            dt / 3.0,
            &k2,
        );

        next.add_scaled_in_place(
            dt / 3.0,
            &k3,
        );

        next.add_scaled_in_place(
            dt / 6.0,
            &k4,
        );

        next.enforce_boundaries(
            boundary,
            boundary_values,
        );

        next
    }

    fn add_scaled_in_place(
        &mut self,
        a: f64,
        rhs: &Self,
    ) {
        self.grid.for_each_index(|idx| {
            let mut gamma = *self.gamma.get(idx);
            let gamma_rhs = *rhs.gamma.get(idx);

            for i in 0..D {
                for j in 0..D {
                    gamma[(i, j)] +=
                        a * gamma_rhs[(i, j)];
                }
            }

            self.gamma.set(idx, gamma);

            let mut k = *self.k.get(idx);
            let k_rhs = *rhs.k.get(idx);

            for i in 0..D {
                for j in 0..D {
                    k[(i, j)] +=
                        a * k_rhs[(i, j)];
                }
            }

            self.k.set(idx, k);

            let lapse_rhs = *rhs.lapse.get(idx);
            let lapse = *self.lapse.get(idx);

            self.lapse.set(
                idx,
                lapse + a * lapse_rhs,
            );

            let mut shift = *self.shift.get(idx);
            let shift_rhs = *rhs.shift.get(idx);

            for i in 0..D {
                shift[i] +=
                    a * shift_rhs[i];
            }

        self.shift.set(idx, shift);
        });
    }
}
