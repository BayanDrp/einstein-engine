use einstein_engine::geometry::christoffel::christoffel_symbols;
use einstein_engine::geometry::curvature::ricci_scalar;
use einstein_engine::geometry::ricci::ricci_from_riemann;
use einstein_engine::geometry::riemann::riemann_from_christoffel;
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use std::sync::Arc;

// flat Minkowski: everything vanishes, so the Ricci scalar is
// zero at every point.
#[test]
fn curvature_flat_minkowski_zero() {
    let grid = Arc::new(Grid::new([5, 5, 5], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
    let lapse: Field<f64, 3> = Field::new(grid.clone(), 1.0);
    let shift: Field<Vector<3>, 3> = Field::new(grid.clone(), Vector::zero());
    let spatial: Field<Tensor2<3>, 3> = Field::new(
        grid.clone(),
        Tensor2::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    let config = BoundaryConfig::<3>::periodic();
    let values = BoundaryValues::<3>::zeros();

    let gamma = christoffel_symbols::<3, 4>(&lapse, &shift, &spatial, &config, &values);
    let riemann = riemann_from_christoffel(&gamma, &config, &values);
    let ricci = ricci_from_riemann(&riemann);
    let r = ricci_scalar::<3, 4>(&lapse, &shift, &spatial, &ricci);

    let v = *r.get([2, 2, 2]);
    assert!(v.abs() < 1e-12);
}

// lapse alpha = 1 + x^2/2, flat spatial metric. both R_tt and R_xx
// are nonzero at x = 1 (see ricci.rs test), so
// R = g^ab R_ab = -R_tt / alpha^2 - R_xx
//   = -(13/8) / (9/4) - 12/17 = -437/306.
#[test]
fn curvature_curved_lapse() {
    let grid = Arc::new(Grid::new([5, 3, 3], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
    let mut lapse: Field<f64, 3> = Field::zeros(grid.clone());
    let fill = grid.clone();
    fill.for_each_index(|idx| {
        let x = grid.coords(idx)[0];
        lapse.set(idx, 1.0 + 0.5 * x * x);
    });
    let shift: Field<Vector<3>, 3> = Field::new(grid.clone(), Vector::zero());
    let spatial: Field<Tensor2<3>, 3> = Field::new(
        grid.clone(),
        Tensor2::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    let config = BoundaryConfig::<3>::periodic();
    let values = BoundaryValues::<3>::zeros();

    let gamma = christoffel_symbols::<3, 4>(&lapse, &shift, &spatial, &config, &values);
    let riemann = riemann_from_christoffel(&gamma, &config, &values);
    let ricci = ricci_from_riemann(&riemann);
    let r = ricci_scalar::<3, 4>(&lapse, &shift, &spatial, &ricci);

    let expected = -(437.0 / 306.0);
    let v = *r.get([2, 1, 1]);
    assert!((v - expected).abs() < 1e-12);
}