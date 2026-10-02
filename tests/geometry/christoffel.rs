#[cfg(test)]
mod tests {
    use einstein_engine::geometry::christoffel::christoffel_symbols;
    use einstein_engine::geometry::tensor::{Tensor2, Vector};
    use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryKind, BoundaryValues};
    use einstein_engine::grid::{Field, Grid};
    use std::sync::Arc;

    // flat Minkowski: alpha = 1, beta = 0, gamma = delta.
    // all Christoffel symbols are zero.
    #[test]
    fn christoffel_flat_minkowski_all_zero() {
        let grid = Arc::new(Grid::new([5, 5, 5], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
        let lapse: Field<f64, 3> = Field::new(grid.clone(), 1.0);
        let shift: Field<Vector<3>, 3> = Field::new(grid.clone(), Vector::zero());
        let spatial: Field<Tensor2<3>, 3> = Field::new(
            grid.clone(),
            Tensor2::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        );
        let config = BoundaryConfig::<3>::periodic();
        let values = BoundaryValues::<3>::zeros();

        let g = christoffel_symbols::<3, 4>(&lapse, &shift, &spatial, &config, &values);
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    assert_eq!(*g.get([i, j, k]), Default::default());
                }
            }
        }
    }

    // static, flat spatial slices: alpha = 1.0 + x, beta = 0, gamma = delta.
    // the nonzero symbols are Gamma^t_tx = alpha' / alpha and
    // Gamma^x_tt = alpha * alpha', where x is spatial index 1.
    #[test]
    fn christoffel_static_lapse() {
        let grid = Arc::new(Grid::new([5, 3, 3], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
        let mut lapse: Field<f64, 3> = Field::zeros(grid.clone());
        for i in 0..5 {
            let x = i as f64 * 0.5;
            lapse.set([i, 0, 0], 1.0 + x);
            lapse.set([i, 1, 0], 1.0 + x);
            lapse.set([i, 2, 0], 1.0 + x);
            lapse.set([i, 1, 1], 1.0 + x);
            lapse.set([i, 2, 1], 1.0 + x);
            lapse.set([i, 2, 2], 1.0 + x);
            lapse.set([i, 0, 1], 1.0 + x);
            lapse.set([i, 0, 2], 1.0 + x);
            lapse.set([i, 1, 2], 1.0 + x);
        }
        let shift: Field<Vector<3>, 3> = Field::new(grid.clone(), Vector::zero());
        let spatial: Field<Tensor2<3>, 3> = Field::new(
            grid.clone(),
            Tensor2::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        );
        let mut config = BoundaryConfig::<3>::periodic();
        config.lower = [BoundaryKind::Dirichlet; 3];
        config.upper = [BoundaryKind::Dirichlet; 3];
        let values = BoundaryValues::<3>::zeros();

        let g = christoffel_symbols::<3, 4>(&lapse, &shift, &spatial, &config, &values);

        // check the middle point, x = 1.0, alpha = 2.0
        let c = *g.get([2, 1, 1]);
        assert_eq!(c.data[1][0][0], 2.0); // Gamma^x_tt = alpha * alpha'
        assert_eq!(c.data[0][0][1], 0.5); // Gamma^t_tx = alpha' / alpha
        assert_eq!(c.data[0][1][0], 0.5); // symmetric in the lower indices
    }

    // boosted slice: alpha = 1 + x, beta = (0.5, 0, 0), gamma = delta.
    // the boost makes g_tt = -alpha^2 + v^2 and g_tx = v, so the symbols
    // change too. with central differences in x (alpha linear, v constant):
    //   Gamma^a_bc = 1/2 g^ad (d_b g_dc + d_c g_bd - d_d g_bc),
    // with d_x g_tt the only nonzero derivative.
    #[test]
    fn christoffel_boosted_lapse() {
        let grid = Arc::new(Grid::new([5, 3, 3], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
        let mut lapse: Field<f64, 3> = Field::zeros(grid.clone());
        for i in 0..5 {
            let x = i as f64 * 0.5;
            lapse.set([i, 0, 0], 1.0 + x);
            lapse.set([i, 1, 0], 1.0 + x);
            lapse.set([i, 2, 0], 1.0 + x);
            lapse.set([i, 1, 1], 1.0 + x);
            lapse.set([i, 2, 1], 1.0 + x);
            lapse.set([i, 2, 2], 1.0 + x);
            lapse.set([i, 0, 1], 1.0 + x);
            lapse.set([i, 0, 2], 1.0 + x);
            lapse.set([i, 1, 2], 1.0 + x);
        }
        let shift: Field<Vector<3>, 3> = Field::new(grid.clone(), Vector::new([0.5, 0.0, 0.0]));
        let spatial: Field<Tensor2<3>, 3> = Field::new(
            grid.clone(),
            Tensor2::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        );
        let config = BoundaryConfig::<3>::periodic();
        let values = BoundaryValues::<3>::zeros();

        let g = christoffel_symbols::<3, 4>(&lapse, &shift, &spatial, &config, &values);

        // check the middle point, x = 1.0, alpha = 2.0, v = 0.5
        let c = *g.get([2, 1, 1]);
        let alpha = 2.0;
        let dalpha = 1.0; // alpha = 1 + x
        let v = 0.5;
        let det = -alpha * alpha; // g: (-alpha^2+v^2, v; v, 1)
        let gtt = 1.0 / det;
        let gtx = -v / det;
        let gxx = (-alpha * alpha + v * v) / det;
        let dg_tt = -2.0 * alpha * dalpha; // d/dx g_tt
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12;

        // Gamma^a_bc = 1/2 g^ad (d_b g_dc + d_c g_bd - d_d g_bc),
        // with d_x g_tt the only nonzero derivative.
        assert!(close(c.data[0][0][1], 0.5 * gtt * dg_tt)); // Gamma^t_tx
        assert!(close(c.data[0][1][0], 0.5 * gtt * dg_tt)); // symmetric in the lower indices
        assert!(close(c.data[0][0][0], 0.5 * gtx * (-dg_tt))); // Gamma^t_tt, needs g^tx != 0
        assert!(close(c.data[1][1][0], 0.5 * gtx * dg_tt)); // Gamma^x_xt
        assert!(close(c.data[1][0][0], 0.5 * gxx * (-dg_tt))); // Gamma^x_tt
        assert!(close(c.data[1][0][1], 0.5 * gtx * dg_tt)); // Gamma^x_tx
    }

    // same boosted lapse, but in 2+1 (two spatial dimensions) to prove
    // the Christoffel symbols are sized by N = D + 1, not by a fixed 4.
    #[test]
    fn christoffel_boosted_lapse_2plus1() {
        let grid = Arc::new(Grid::new([5, 3], [0.5, 0.5], [0.0, 0.0]));
        let mut lapse: Field<f64, 2> = Field::zeros(grid.clone());
        for i in 0..5 {
            let x = i as f64 * 0.5;
            for j in 0..3 {
                lapse.set([i, j], 1.0 + x);
            }
        }
        let shift: Field<Vector<2>, 2> = Field::new(grid.clone(), Vector::new([0.5, 0.0]));
        let spatial: Field<Tensor2<2>, 2> = Field::new(
            grid.clone(),
            Tensor2::new([[1.0, 0.0], [0.0, 1.0]]),
        );
        let config = BoundaryConfig::<2>::periodic();
        let values = BoundaryValues::<2>::zeros();

        let g = christoffel_symbols::<2, 3>(&lapse, &shift, &spatial, &config, &values);
        let c = *g.get([2, 1]);
        let alpha = 2.0;
        let dalpha = 1.0;
        let v = 0.5;
        let det = -alpha * alpha;
        let gtt = 1.0 / det;
        let gtx = -v / det;
        let gxx = (-alpha * alpha + v * v) / det;
        let dg_tt = -2.0 * alpha * dalpha;
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12;

        assert!(close(c.data[0][0][1], 0.5 * gtt * dg_tt)); // Gamma^t_tx
        assert!(close(c.data[0][0][0], 0.5 * gtx * (-dg_tt))); // Gamma^t_tt
        assert!(close(c.data[1][0][0], 0.5 * gxx * (-dg_tt))); // Gamma^x_tt
        assert!(close(c.data[1][0][1], 0.5 * gtx * dg_tt)); // Gamma^x_tx
    }
}