use einstein_engine::geometry::christoffel::christoffel_symbols;
use einstein_engine::geometry::curvature::ricci_scalar;
use einstein_engine::geometry::ricci::ricci_from_riemann;
use einstein_engine::geometry::riemann::riemann_from_christoffel;
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::relativity::einstein::einstein_tensor;
use std::sync::Arc;

// flat Minkowski vacuum: all curvature vanishes, so G_mu nu = 0.
#[test]
fn einstein_vacuum_zero() {
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
    let scalar = ricci_scalar::<3, 4>(&lapse, &shift, &spatial, &ricci);
    let g = einstein_tensor::<3, 4>(&lapse, &shift, &spatial, &ricci, &scalar);

    let c = *g.get([2, 2, 2]);
    assert_eq!(c.data, [[0.0f64; 4]; 4]);
}

// a curved static lapse has nonzero Ricci and scalar, so G_mu nu
// is not zero, but it is consistent: every component must match
// R_mu nu - 1/2 R g_mu nu built by hand at the same point.
#[test]
fn einstein_curved_lapse_consistent() {
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
    let scalar = ricci_scalar::<3, 4>(&lapse, &shift, &spatial, &ricci);
    let g = einstein_tensor::<3, 4>(&lapse, &shift, &spatial, &ricci, &scalar);

    // rebuild g_mu nu at x = 1 by hand and verify the formula
    let c = *g.get([2, 1, 1]);
    let r = *ricci.get([2, 1, 1]);
    let r_scalar = *scalar.get([2, 1, 1]);
    let alpha = 1.5;
    let mut metric = [[0.0f64; 4]; 4];
    metric[0][0] = -alpha * alpha;
    metric[1][1] = 1.0;
    metric[2][2] = 1.0;
    metric[3][3] = 1.0;

    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
    for mu in 0..4 {
        for nu in 0..4 {
            let expected = r.data[mu][nu] - 0.5 * r_scalar * metric[mu][nu];
            assert!(close(c.data[mu][nu], expected));
        }
    }
    // spot-check one component against the values from the ricci test:
    // R_tt = 1.625, R_xx = -12/17, R = -437/306.
    // G_tt = R_tt - 1/2 R g_tt
    let expected_tt = 1.625 - 0.5 * (-437.0 / 306.0) * (-2.25);
    assert!(close(c.data[0][0], expected_tt));
}