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

        let g = christoffel_symbols(&lapse, &shift, &spatial, &config, &values);
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

        let g = christoffel_symbols(&lapse, &shift, &spatial, &config, &values);

        // check the middle point, x = 1.0, alpha = 2.0
        let c = *g.get([2, 1, 1]);
        assert_eq!(c.data[1][0][0], 2.0); // Gamma^x_tt = alpha * alpha'
        assert_eq!(c.data[0][0][1], 0.5); // Gamma^t_tx = alpha' / alpha
        assert_eq!(c.data[0][1][0], 0.5); // symmetric in the lower indices
    }
}