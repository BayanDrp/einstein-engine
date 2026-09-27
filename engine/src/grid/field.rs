use super::Grid;

pub struct Field<T, const D: usize> {
    pub grid: Grid<D>,
    pub data: Vec<T>,
}

impl<T: Clone, const D: usize> Field<T, D> {
    /// Field filled with `value` on `grid`.
    pub fn new(grid: Grid<D>, value: T) -> Self {
        let data = vec![value; grid.size()];
        Self { grid, data }
    }

    /// Field of zeros on `grid`.
    pub fn zeros(grid: Grid<D>) -> Self
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