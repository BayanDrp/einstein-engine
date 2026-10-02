#[cfg(test)]
mod tests {
use einstein_engine::geometry::christoffel::christoffel_symbols;
use einstein_engine::geometry::riemann::riemann_from_christoffel;
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use std::sync::Arc;

// flat Minkowski: alpha = 1, beta = 0, gamma = delta.
// both Christoffel symbols and their derivatives vanish,
// so every Riemann component is zero.
#[test]
fn riemann_flat_minkowski_all_zero() {
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
    let r = riemann_from_christoffel(&gamma, &config, &values);
    let c = *r.get([2, 2, 2]);
    assert_eq!(c, Default::default());
}

    // curved slice: alpha = 1 + x^2/2 (so alpha' = x, alpha'' = 1),
    // beta = 0, gamma = delta. the only nonzero Christoffel symbols are
    // Gamma^x_tt = alpha alpha' and Gamma^t_tx = alpha' / alpha, and for
    // this metric the Riemann tensor is
    //   R^x_txt = alpha alpha''      (with alpha'' = 1)
    //   R^x_ttx = -R^x_txt           (antisymmetric in c, d)
    // check at x = 1.0, alpha = 1.5
    #[test]
    fn riemann_curved_lapse() {
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
        let r = riemann_from_christoffel(&gamma, &config, &values);

        // middle point, x = 1.0, alpha = 1.5, alpha'' = 1.
        // analytically R^x_txt = alpha alpha'' = 1.5, but the grid uses
        // central differences on the cubic Gamma^x_tt, whose central
        // difference at x = 1 is exactly 2.625 (not 2.5), so the grid
        // value is R^x_txt = 2.625 - (alpha')^2 = 2.625 - 1.0 = 1.625.
        let c = *r.get([2, 1, 1]);
        let expected = 1.625;
        let close = |a: f64| (a - expected).abs() < 1e-12;
        assert!(close(c.data[1][0][1][0])); // R^x_txt
        assert!(close(-c.data[1][0][0][1])); // R^x_ttx = -R^x_txt
    }
}