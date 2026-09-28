#[cfg(test)]
mod tests {
    use einstein_engine::grid::{Field, Grid};
    use std::sync::Arc;

    fn grid3() -> Arc<Grid<3>> {
        Arc::new(Grid::new([2, 3, 4], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]))
    }

    #[test]
    fn field_new_fills_value() {
        let f = Field::new(grid3(), 7.0);
        assert_eq!(f.data.len(), 24);
        assert!(f.data.iter().all(|&v| v == 7.0));
    }

    #[test]
    fn field_zeros() {
        let f: Field<f64, 3> = Field::zeros(grid3());
        assert_eq!(f.data.len(), 24);
        assert!(f.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn field_get_set() {
        let mut f = Field::zeros(grid3());
        f.set([1, 2, 3], 42.0);
        assert_eq!(*f.get([1, 2, 3]), 42.0);
        assert_eq!(*f.get([0, 0, 0]), 0.0);
    }

    #[test]
    #[should_panic]
    fn field_get_out_of_bounds_panics() {
        let f: Field<f64, 3> = Field::zeros(grid3());
        let _ = f.get([2, 0, 0]);
    }

    #[test]
    fn field_holds_vector_values() {
        let mut f: Field<[f64; 3], 3> = Field::zeros(grid3());
        f.set([0, 0, 0], [1.0, 2.0, 3.0]);
        assert_eq!(*f.get([0, 0, 0]), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn fields_share_one_grid() {
        let grid = grid3();
        let a: Field<f64, 3> = Field::zeros(grid.clone());
        let b: Field<f64, 3> = Field::zeros(grid.clone());
        assert!(Arc::ptr_eq(&a.grid, &b.grid));
    }

    #[test]
    fn field_holds_matrix_values() {        let mut f: Field<[[f64; 3]; 3], 3> = Field::zeros(grid3());
        let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        f.set([1, 1, 1], m);
        assert_eq!(*f.get([1, 1, 1]), m);
        assert_eq!(*f.get([0, 0, 0]), [[0.0; 3]; 3]);
    }
}
