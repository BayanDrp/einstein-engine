#[cfg(test)]
mod tests {
    use einstein_engine::grid::Grid;

    #[test]
    fn grid_new() {
        let grid = Grid::new([2, 3, 4], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]);
        assert_eq!(grid.size(), 24);
    }
    #[test]
    fn grid_index() {
        let grid = Grid::new([2, 3, 4], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]);
        assert_eq!(grid.index([0, 0, 0]), 0);
        assert_eq!(grid.index([1, 2, 3]), 23);
    }
    #[test]
    fn grid_coords() {
        let grid = Grid::new([2, 3, 4], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]);
        assert_eq!(grid.coords([0, 0, 0]), [0.0, 0.0, 0.0]);
        assert_eq!(grid.coords([1, 2, 3]), [1.0, 4.0, 9.0]);
    } 
    #[test]
    fn grid_contains() {
        let grid = Grid::new([2, 3, 4], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]);
        assert!(grid.contains([0, 0, 0]));
        assert!(!grid.contains([2, 3, 4]));
    }
    #[test]
    fn grid_dim() {
        let grid = Grid::new([2, 3, 4], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]);
        assert_eq!(grid.dim(), 3);
    }
    #[test]
    fn grid_index_row_major() {
        let grid = Grid::new([2, 3, 4], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]);
        assert_eq!(grid.index([0, 0, 1]), 1);
        assert_eq!(grid.index([0, 1, 0]), 4);
        assert_eq!(grid.index([1, 0, 0]), 12);
    }
    #[test]
    fn grid_coords_nonzero_origin() {
        let grid = Grid::new([2, 3, 4], [1.0, 2.0, 3.0], [10.0, -5.0, 0.5]);
        assert_eq!(grid.coords([0, 0, 0]), [10.0, -5.0, 0.5]);
        assert_eq!(grid.coords([1, 2, 3]), [11.0, -1.0, 9.5]);
    }
    #[test]
    fn grid_contains_boundary() {
        let grid = Grid::new([2, 3, 4], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]);
        assert!(grid.contains([1, 2, 3]));
        assert!(!grid.contains([2, 0, 0]));
        assert!(!grid.contains([0, 3, 0]));
        assert!(!grid.contains([0, 0, 4]));
    }
    #[test]
    #[should_panic]
    fn grid_rejects_empty_shape() {
        let _ = Grid::new([0, 3, 4], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]);
    }
    #[test]
    #[should_panic]
    fn grid_rejects_bad_spacing() {
        let _ = Grid::new([2, 3, 4], [1.0, 0.0, 3.0], [0.0, 0.0, 0.0]);
    }                                                                                                                  
}                                                                                                                                                                               


