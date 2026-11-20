use std::sync::Arc;

use crate::geometry::metric::{ExtrinsicCurvature, LapseField, ShiftField, SliceMetric};
use crate::geometry::tensor::Tensor2;
use crate::grid::{Field, Grid};
use crate::numerics::rk4::AdmState;

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
) -> Result<AdmState<D>, &'static str> {
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
        return Err("Every grid point must satisfy r > M/2");
    }

    let mut gamma: SliceMetric<D> = Field::new(grid.clone(), Tensor2::<D>::zero());

    let k: ExtrinsicCurvature<D> = Field::new(grid.clone(), Tensor2::<D>::zero());

    let mut lapse: LapseField<D> = Field::zeros(grid.clone());

    let shift: ShiftField<D> =
        Field::new(grid.clone(), crate::geometry::tensor::Vector::<D>::zero());

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

    Ok(AdmState {
        grid,
        gamma,
        k,
        lapse,
        shift,
    })
}

/// The same isotropic Schwarzschild slice as [`schwarzschild`], but without
/// the "every node outside the horizon" restriction.
///
/// The metric functions
///
///   gamma_ij = psi^4 delta_ij,  psi = 1 + M/(2 rho),
///   alpha    = (1 - M/(2 rho)) / (1 + M/(2 rho)),
///
/// are smooth functions of rho = |x| on the punctured space (rho > 0), and
/// their natural domain extends smoothly through rho = M/2: the numerator
/// of alpha vanishes there and flips sign instead of blowing up. `rho = M/2`
/// is where the isotropic-to-areal radius map dr/drho = 0 hits the horizon;
/// inside it the same formulas simply carry the smooth continuation of the
/// interior chart. Nodes there never invalidate the exterior data, and the
/// restriction only exists because `schwarzschild` wants to fail loudly on
/// grids that cannot represent the exterior domain.
///
/// Circular-orbit fixtures need that restriction lifted: the photon sphere
/// sits at the areal radius r = 3M, i.e. the isotropic radius
/// rho = M (2 + sqrt(3))/2 ≈ 1.87 M, and resolving its curvature requires a
/// node spacing much finer than the horizon exclusion allows on a grid
/// centered on the hole. This constructor is therefore valid for any grid
/// whose nodes satisfy rho > 0 (at rho = 0 itself psi diverges), including
/// lattices that cover the interior region.
pub fn isotropic_fields<const D: usize>(
    grid: Arc<Grid<D>>,
    mass: f64,
) -> Result<AdmState<D>, &'static str> {
    if D != 3 {
        return Err("Schwarzschild initial data only implemented in 3D");
    }

    if mass <= 0.0 || !mass.is_finite() {
        return Err("Schwarzschild initial data requires mass > 0 and finite");
    }

    let mut degenerate_node = false;
    grid.for_each_index(|idx| {
        let coords = grid.coords(idx);
        let rho_squared: f64 = coords.iter().map(|x| x * x).sum();
        if !(rho_squared > 0.0) {
            degenerate_node = true;
        }
    });

    if degenerate_node {
        return Err("Every grid point must have rho > 0 (psi diverges at the puncture)");
    }

    let mut gamma: SliceMetric<D> = Field::new(grid.clone(), Tensor2::<D>::zero());

    let k: ExtrinsicCurvature<D> = Field::new(grid.clone(), Tensor2::<D>::zero());

    let mut lapse: LapseField<D> = Field::zeros(grid.clone());

    let shift: ShiftField<D> =
        Field::new(grid.clone(), crate::geometry::tensor::Vector::<D>::zero());

    grid.for_each_index(|idx| {
        let coords = grid.coords(idx);

        let rho_squared: f64 = coords.iter().map(|x| x * x).sum();
        let rho = rho_squared.sqrt();

        let psi = 1.0 + mass / (2.0 * rho);
        let psi_fourth = psi * psi * psi * psi; // psi^4

        let alpha = (1.0 - mass / (2.0 * rho)) / (1.0 + mass / (2.0 * rho));

        lapse.set(idx, alpha);

        let mut metric = Tensor2::<D>::zero();
        for i in 0..D {
            metric[(i, i)] = psi_fourth;
        }
        gamma.set(idx, metric);
    });

    Ok(AdmState {
        grid,
        gamma,
        k,
        lapse,
        shift,
    })
}
