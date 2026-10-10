use std::sync::Arc;

use crate::geometry::metric::{
    ExtrinsicCurvature,
    LapseField,
    ShiftField,
    SliceMetric,
};
use crate::geometry::tensor::Tensor2;
use crate::grid::{Field, Grid};
use crate::numerics::rk4::EvolutionState;

/// Construct Schwarzschild initial data in isotropic Cartesian coordinates.
///
/// Requires a 3D grid whose every point lies outside the isotropic
/// horizon: r > M/2.
///
/// gamma_ij = psi^4 delta_ij
/// K_ij     = 0
/// alpha    = (1 - M/(2r)) / (1 + M/(2r))
/// beta^i   = 0

pub fn schwarzschild<const D: usize>(
    grid: Arc<Grid<D>>,
    mass: f64,
) -> Result<EvolutionState<D>, &'static str> {
    // check that the grid is 3D, since Schwarzschild initial data is only implemented in 3D
    if D != 3 {
        return Err("Schwarzschild initial data only implemented in 3D");
    }

    // check that the mass is positive
    if mass <= 0.0 || !mass.is_finite() {
        return Err("Schwarzschild initial data requires mass > 0 and finite");
    }

    let mut invalid_radius = false;

    grid.for_each_index(|idx| {
        let coords = grid.coords(idx);

        let r_squared: f64 = coords.iter().map(|x| x * x).sum();
        let r = r_squared.sqrt();

        if !r.is_finite() || r <= mass / 2.0 {
            invalid_radius = true;
        }
    });

    if invalid_radius {
        return Err(
            "Every grid point must satisfy r > M/2",
        );
    }

    let mut gamma: SliceMetric<D> =
        Field::new(grid.clone(), Tensor2::<D>::zero());

    let k: ExtrinsicCurvature<D> =
        Field::new(grid.clone(), Tensor2::<D>::zero());

    let mut lapse: LapseField<D> =
        Field::zeros(grid.clone());

    let shift: ShiftField<D> =
        Field::new(
            grid.clone(),
            crate::geometry::tensor::Vector::<D>::zero(),
        );

    grid.for_each_index(|idx| {
        let coords = grid.coords(idx);

        let r_squared: f64 = coords.iter().map(|x| x * x).sum();
        let r = r_squared.sqrt();

        let psi = 1.0 + mass / (2.0 * r);
        let psi_fourth = psi * psi * psi * psi; // psi^4

        let alpha = (1.0 - mass / (2.0 * r)) / (1.0 + mass / (2.0 * r));

        lapse.set(idx, alpha);

        let mut metric = Tensor2::<D>::zero();
        for i in 0..D {
            metric[(i, i)] = psi_fourth;
        }
        gamma.set(idx, metric);
    });

    Ok(EvolutionState {
        grid,
        gamma,
        k,
        lapse,
        shift,
    })
  
}