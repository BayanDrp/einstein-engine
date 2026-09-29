#[cfg(test)]
mod tests {
    use einstein_engine::grid::boundary::{
        BoundaryConfig, BoundaryKind, BoundaryValues,
    };
    use einstein_engine::grid::{Field, Grid};
    use einstein_engine::numerics::derivatives::central_partial;
    use std::sync::Arc;

    /// f(x) = 2x + 1 on a 1D grid: derivative is 2 everywhere.
    fn linear_field() -> Field<f64, 1> {
        let grid = Arc::new(Grid::new([5], [0.5], [0.0]));
        let mut f: Field<f64, 1> = Field::zeros(grid);
        for i in 0..5 {
            f.set([i], 2.0 * (i as f64 * 0.5) + 1.0);
        }
        f
    }
    // f(x,y) = x^2 + y^2. d/dx f = 2x, so values along x are 0.5, 1.0, 1.5.
    fn quadratic_field() -> Field<f64, 2> {
        let grid = Arc::new(Grid::new([3, 3], [0.5, 0.5], [0.0, 0.0]));
        let mut f: Field<f64, 2> = Field::zeros(grid);
        for i in 0..3 {
            for j in 0..3 {
                f.set([i, j], (i as f64 * 0.5) * (i as f64 * 0.5) + (j as f64 * 0.5) * (j as f64 * 0.5));
            }
        }
        f
    }

    #[test]
    fn central_partial_dirichlet_edges() {
        let f = linear_field();
        let mut config = BoundaryConfig::periodic();
        config.lower = [BoundaryKind::Dirichlet];
        config.upper = [BoundaryKind::Dirichlet];
        // Exact face values of f(x) = 2x + 1.
        let values = BoundaryValues {
            lower: [1.0],
            upper: [5.0],
        };
        let d = central_partial(&f, 0, &config, &values);
        for i in 0..5 {
            assert_eq!(*d.get([i]), 2.0);
        }
    }

    #[test]
    fn central_partial_neumann_edges() {
        let f = linear_field();

        let mut config = BoundaryConfig::periodic();

        config.lower = [BoundaryKind::Neumann];
        config.upper = [BoundaryKind::Neumann];

        let values = BoundaryValues {
            lower: [-2.0],
            upper: [2.0],
        };

        let d = central_partial(&f, 0, &config, &values);
        // derivative should be 2 at every point
        for i in 0..5 {
            assert_eq!(*d.get([i]), 2.0);
        }
    }

    #[test]
    fn central_partial_2d_neumann() {
        let f2 = quadratic_field();

        let config = BoundaryConfig {
            lower: [BoundaryKind::Neumann, BoundaryKind::Neumann],
            upper: [BoundaryKind::Neumann, BoundaryKind::Neumann],
        };
        let values = BoundaryValues {
            lower: [0.0, 0.0],
            upper: [0.0, 0.0],
        };

        let d2 = central_partial(&f2, 0, &config, &values);

        // d/dx f = 2x = 0, 1, 2 at i = 0, 1, 2 (x = 0, 0.5, 1.0)
        // but the stencils at the edges are one-sided, so we get
        // 0.5, 1.0, 1.5 instead of the exact 0, 1, 2.
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == 0 {
                    0.5
                } else if i == 1 {
                    1.0
                } else {
                    1.5
                };
                assert_eq!(*d2.get([i, j]), expected);
            }
        }
    }

    #[test]
    // constant field: derivative is 0 everywhere, even at the edges
    fn central_partial_periodic_constant_field() {
        let grid = Arc::new(Grid::new([4], [1.0], [0.0]));
        let f: Field<f64, 1> = Field::new(grid, 3.0);
        let config = BoundaryConfig::periodic();
        let d = central_partial(&f, 0, &config, &BoundaryValues::zeros());
        for i in 0..4 {
            assert_eq!(*d.get([i]), 0.0);
        }
    }

    #[test]
    // grid with one point: nothing to differentiate, so 0
    fn central_partial_single_point_is_zero() {
        let grid = Arc::new(Grid::new([1], [1.0], [0.0]));
        let f: Field<f64, 1> = Field::new(grid, 3.0);
        let config = BoundaryConfig::periodic();
        let d = central_partial(&f, 0, &config, &BoundaryValues::zeros());
        assert_eq!(*d.get([0]), 0.0);
    }
}
