#[cfg(test)]
mod tests {
    use einstein_engine::grid::boundary::{
        apply, BoundaryConfig, BoundaryKind, BoundaryValues,
    };
    use einstein_engine::grid::{Field, Grid};
    use std::sync::Arc;

    #[test]
    fn shared_grid() {
        let grid = Grid::new([2, 3, 4], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]);
        let boundary = BoundaryConfig::<3>::periodic();
        assert_eq!(boundary.lower.len(), grid.dim());
        assert_eq!(boundary.upper.len(), grid.dim());
    }
    #[test]
    fn periodic() {
        let boundary = BoundaryConfig::<3>::periodic();
        assert!(boundary.lower.iter().all(|&k| k == BoundaryKind::Periodic));
        assert!(boundary.upper.iter().all(|&k| k == BoundaryKind::Periodic));
    }

    fn dirichlet_1d() -> (Field<f64, 1>, BoundaryConfig<1>, BoundaryValues<1>) {
        let grid = Arc::new(Grid::new([4], [0.5], [0.0]));
        let mut f: Field<f64, 1> = Field::zeros(grid);
        f.set([1], 1.0);
        f.set([2], 2.0);
        let mut config = BoundaryConfig::periodic();
        config.lower = [BoundaryKind::Dirichlet];
        config.upper = [BoundaryKind::Dirichlet];
        let values = BoundaryValues {
            lower: [10.0],
            upper: [20.0],
        };
        (f, config, values)
    }

    #[test]
    fn dirichlet_sets_faces() {
        let (mut f, config, values) = dirichlet_1d();
        apply(&mut f, &config, &values);
        assert_eq!(*f.get([0]), 10.0);
        assert_eq!(*f.get([3]), 20.0);
        assert_eq!(*f.get([1]), 1.0);
        assert_eq!(*f.get([2]), 2.0);
    }

    #[test]
    fn neumann_extrapolates_from_interior() {
        let grid = Arc::new(Grid::new([3], [2.0], [0.0]));
        let mut f: Field<f64, 1> = Field::zeros(grid);
        f.set([1], 5.0);
        let mut config = BoundaryConfig::periodic();
        config.lower = [BoundaryKind::Neumann];
        config.upper = [BoundaryKind::Neumann];
        let values = BoundaryValues {
            lower: [1.0],
            upper: [0.0],
        };
        apply(&mut f, &config, &values);
        assert_eq!(*f.get([0]), 5.0 + 1.0 * 2.0);
        assert_eq!(*f.get([2]), 5.0 + 0.0 * 2.0);
    }

    #[test]
    fn periodic_averages_endpoints() {
        let grid = Arc::new(Grid::new([3], [1.0], [0.0]));
        let mut f: Field<f64, 1> = Field::zeros(grid);
        f.set([0], 2.0);
        f.set([1], 100.0);
        f.set([2], 6.0);
        let config = BoundaryConfig::periodic();
        apply(&mut f, &config, &BoundaryValues::zeros());
        assert_eq!(*f.get([0]), 4.0);
        assert_eq!(*f.get([2]), 4.0);
        assert_eq!(*f.get([1]), 100.0);
    }

    #[test]
    fn values_constructors() {
        let u = BoundaryValues::<2>::uniform(3.0);
        assert_eq!(u.lower, [3.0, 3.0]);
        assert_eq!(u.upper, [3.0, 3.0]);
        let z = BoundaryValues::<2>::zeros();
        assert_eq!(z.lower, [0.0, 0.0]);
    }
    #[test]
    fn neumann() {
        let boundary = BoundaryConfig::<3>::neumann();
        assert!(boundary.lower.iter().all(|&k| k == BoundaryKind::Neumann));
        assert!(boundary.upper.iter().all(|&k| k == BoundaryKind::Neumann));
    }
    #[test]
    fn dirichlet() {
        let boundary = BoundaryConfig::<3>::dirichlet();
        assert!(boundary.lower.iter().all(|&k| k == BoundaryKind::Dirichlet));
        assert!(boundary.upper.iter().all(|&k| k == BoundaryKind::Dirichlet));
    }
}