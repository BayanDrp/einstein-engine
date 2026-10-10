use crate::geometry::metric::{
    ExtrinsicCurvature,
    SliceMetric,
};
use crate::geometry::tensor::Vector;
use crate::grid::boundary::{
    BoundaryConfig,
    BoundaryValues,
};
use crate::grid::Field;
use crate::relativity::constraints::{
    hamiltonian_constraint,
    momentum_constraint,
};

/// Maximum constraint violation over the whole spatial grid.
///
/// For an exact Einstein solution:
///     hamiltonian_max = 0
///     momentum_max    = 0
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstraintResidual {
    pub hamiltonian_max: f64,
    pub momentum_max: f64,
}

/// Largest absolute value across a scalar field.
fn max_abs_scalar<const D: usize>(field: &Field<f64, D>) -> f64 {
    field
        .data
        .iter()
        .fold(0.0f64, |worst, &value| worst.max(value.abs()))
}

/// Largest absolute component across a vector field.
fn max_abs_vector<const D: usize>(field: &Field<Vector<D>, D>) -> f64 {
    field.data.iter().fold(0.0f64, |worst, value| {
        let mut worst = worst;
        for i in 0..D {
            worst = worst.max(value[i].abs());
        }
        worst
    })
}

/// Evaluate both Einstein constraints on one slice and reduce them to a
/// single number each.
///
/// `rho` and `S^i` are required arguments rather than defaulted, so vacuum
/// costs one zero field per call and matter needs no API change later.
///
/// Nothing here depends on the time integrator: the inputs are the slice
/// variables, so this can measure any slice however it was produced.
pub fn residuals<const D: usize>(
    spatial: &SliceMetric<D>,
    extrinsic: &ExtrinsicCurvature<D>,
    rho: &Field<f64, D>,
    momentum: &Field<Vector<D>, D>,
    boundary: &BoundaryConfig<D>,
    boundary_values: &BoundaryValues<D>,
) -> ConstraintResidual {
    assert_eq!(spatial.grid.shape, extrinsic.grid.shape);
    assert_eq!(spatial.grid.shape, rho.grid.shape);
    assert_eq!(spatial.grid.shape, momentum.grid.shape);

    let hamiltonian = hamiltonian_constraint(
        spatial,
        extrinsic,
        rho,
        boundary,
        boundary_values,
    );

    let momentum_field = momentum_constraint(
        spatial,
        extrinsic,
        momentum,
        boundary,
        boundary_values,
    );

    ConstraintResidual {
        hamiltonian_max: max_abs_scalar(&hamiltonian),
        momentum_max: max_abs_vector(&momentum_field),
    }
}
