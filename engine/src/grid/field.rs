use super::Grid;
use std::sync::Arc;


#[derive(Debug, Clone)]
pub struct Field<T, const NDIM: usize> {
    pub grid: Arc<Grid<NDIM>>,
    pub data: Vec<T>,
}

impl<T: Clone, const NDIM: usize> Field<T, NDIM> {
    /// Field filled with `value` on the shared `grid`.
    pub fn new(grid: Arc<Grid<NDIM>>, value: T) -> Self {
        let data = vec![value; grid.size()];
        Self { grid, data }
    }

    /// Field of zeros on the shared `grid`.
    pub fn zeros(grid: Arc<Grid<NDIM>>) -> Self
    where
        T: Default,
    {
        Self::new(grid, T::default())
    }

    pub fn get(&self, idx: [usize; NDIM]) -> &T {
        let flat = self.grid.index(idx);
        &self.data[flat]
    }

    pub fn set(&mut self, idx: [usize; NDIM], value: T) {
        let flat = self.grid.index(idx);
        self.data[flat] = value;
    }
}