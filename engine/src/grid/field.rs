use super::Grid;
use std::sync::Arc;


pub struct Field<T, const D: usize> {
    pub grid: Arc<Grid<D>>,
    pub data: Vec<T>,
}

impl<T: Clone, const D: usize> Field<T, D> {
    /// Field filled with `value` on the shared `grid`.
    pub fn new(grid: Arc<Grid<D>>, value: T) -> Self {
        let data = vec![value; grid.size()];
        Self { grid, data }
    }

    /// Field of zeros on the shared `grid`.
    pub fn zeros(grid: Arc<Grid<D>>) -> Self
    where
        T: Default,
    {
        Self::new(grid, T::default())
    }

    pub fn get(&self, idx: [usize; D]) -> &T {
        let flat = self.grid.index(idx);
        &self.data[flat]
    }

    pub fn set(&mut self, idx: [usize; D], value: T) {
        let flat = self.grid.index(idx);
        self.data[flat] = value;
    }
}