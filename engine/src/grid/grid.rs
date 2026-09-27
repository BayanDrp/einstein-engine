#[derive(Debug, Clone)]
pub struct Grid<const D: usize> {
    pub shape: [usize; D],
    pub spacing: [f64; D],
    pub origin: [f64; D],
}

impl<const D: usize> Grid<D> {
    pub fn new(
        shape: [usize; D],
        spacing: [f64; D],
        origin: [f64; D],
    ) -> Self {
        assert!(shape.iter().all(|&n| n > 0));
        assert!(spacing.iter().all(|&dx| dx > 0.0));

        Self {
            shape,
            spacing,
            origin,
        }
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn dim(&self) -> usize {
        D
    }

    /// True if `idx` is inside the grid bounds.
    pub fn contains(&self, idx: [usize; D]) -> bool {
        idx.iter().zip(self.shape.iter()).all(|(&i, &n)| i < n)
    }

    /// Row-major flat offset for a multi-index.
    pub fn index(&self, idx: [usize; D]) -> usize {
        assert!(self.contains(idx));
        let mut flat = 0;
        for d in 0..D {
            flat = flat * self.shape[d] + idx[d];
        }
        flat
    }

    /// Physical coordinates of a multi-index.
    pub fn coords(&self, idx: [usize; D]) -> [f64; D] {
        assert!(self.contains(idx));
        let mut out = [0.0; D];
        for d in 0..D {
            out[d] = self.origin[d] + idx[d] as f64 * self.spacing[d];
        }
        out
    }
}