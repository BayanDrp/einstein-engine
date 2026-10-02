use einstein_engine::geometry::christoffel::christoffel_symbols;
use einstein_engine::geometry::ricci::ricci_from_riemann;
use einstein_engine::geometry::riemann::riemann_from_christoffel;
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use std::sync::Arc;

// flat Minkowski: Christoffel symbols vanish, so both Riemann
// and the contracted Ricci tensor are zero everywhere.
#[test]
fn ricci_flat_minkowski_all_zero() {
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

    let c = *ricci.get([2, 2, 2]);
    assert_eq!(c, Default::default());
}

// lapse alpha = 1 + x^2/2, flat spatial metric. Riemann has
// R^x_txt = 1.625 at the grid point x = 1 (see riemann.rs test),
// and Ricci contracts it: R_tt = R^x_txt, everything else zero.
#[test]
fn ricci_curved_lapse() {
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

    let c = *ricci.get([2, 1, 1]);
    // R_tt = R^x_txt = 1.625 (see riemann.rs test)
    let close = |a: f64, expected: f64| (a - expected).abs() < 1e-12;
    assert!(close(c.data[0][0], 1.625)); // R_tt
    assert_eq!(c.data[0][1], 0.0); // R_tx
    assert_eq!(c.data[1][0], 0.0); // R_xt
    // R_xx = R^t_xtx = -d/dx(alpha'/alpha) - (alpha'/alpha)^2,
    // which is exactly -12/17 in floating point on this grid.
    let expected_xx = 12.0 / 17.0;
    assert!(close(-c.data[1][1], expected_xx)); // R_xx
    assert_eq!(c.data[1][2], 0.0); // R_xy
    assert_eq!(c.data[2][2], 0.0); // R_yy
}