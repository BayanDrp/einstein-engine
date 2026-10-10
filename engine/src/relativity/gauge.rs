use crate::geometry::metric::{
    ExtrinsicCurvature,
    LapseField,
    ShiftField,
    SliceMetric,
};
use crate::geometry::tensor::{invert, Vector};
use crate::grid::boundary::{
    BoundaryConfig,
    BoundaryValues,
};
use crate::grid::Field;
use crate::numerics::derivatives::shift_derivative;

/// 1+log slicing:
///
/// ∂t α = β^i ∂i α - 2 α K
///
/// where
///
/// K = γ^ij K_ij
///
/// This returns the time derivative of the lapse field.
pub fn lapse_rhs<const D: usize>(
    lapse: &LapseField<D>,
    shift: &ShiftField<D>,
    spatial: &SliceMetric<D>,
    extrinsic: &ExtrinsicCurvature<D>,
    boundary: &BoundaryConfig<D>,
    boundary_values: &BoundaryValues<D>,
) -> LapseField<D> {
    let advection = shift_derivative(lapse, shift, boundary, boundary_values);
    let mut out: LapseField<D> = Field::new(lapse.grid.clone(), 0.0);

    spatial.grid.for_each_index(|idx| {
        let gamma = spatial.get(idx);
        let k = extrinsic.get(idx);

        // Build γ^{ij}
        let mut gamma_matrix = [[0.0; D]; D];

        for i in 0..D {
            for j in 0..D {
                gamma_matrix[i][j] = gamma[(i, j)];
            }
        }

        let gamma_inv = invert(&gamma_matrix);

        // K = γ^{ij} K_ij
        let mut trace_k = 0.0;

        for i in 0..D {
            for j in 0..D {
                trace_k += gamma_inv[i][j] * k[(i, j)];
            }
        }

        let alpha = *lapse.get(idx);

        out.set(idx, *advection.get(idx) - 2.0 * alpha * trace_k);
    });

    out
}

/// The shift is held fixed:
///
/// ∂t β^i = 0
///
/// A Γ-driver condition would evolve β^i from Γ^i = γ^jk Γ^i_jk, which needs
/// ∂t Γ^i, i.e. third derivatives of the metric. The derivative operators
/// here only go up to second order, so every gauge offered so far freezes the
/// shift. This returns the resulting zero derivative; only the grid of the
/// argument is used.
pub fn shift_rhs<const D: usize>(shift: &ShiftField<D>) -> ShiftField<D> {
    Field::new(shift.grid.clone(), Vector::<D>::zero())
}
