#[derive(Debug, Clone)]
pub struct Grid<const NDIM: usize> {
    pub shape: [usize; NDIM],
    pub spacing: [f64; NDIM],
    pub origin: [f64; NDIM],
}

impl<const NDIM: usize> Grid<NDIM> {
    pub fn new(
        shape: [usize; NDIM],
        spacing: [f64; NDIM],
        origin: [f64; NDIM],
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
        NDIM
    }

    /// True if `idx` is inside the grid bounds.
    pub fn contains(&self, idx: [usize; NDIM]) -> bool {
        idx.iter().zip(self.shape.iter()).all(|(&i, &n)| i < n)
    }

    /// Row-major flat offset for a multi-index.
    pub fn index(&self, idx: [usize; NDIM]) -> usize {
        assert!(self.contains(idx));
        let mut flat = 0;
        for d in 0..NDIM {
            flat = flat * self.shape[d] + idx[d];
        }
        flat
    }

    /// Physical coordinates of a multi-index.
    pub fn coords(&self, idx: [usize; NDIM]) -> [f64; NDIM] {
        assert!(self.contains(idx));
        let mut out = [0.0; NDIM];
        for d in 0..NDIM {
            out[d] = self.origin[d] + idx[d] as f64 * self.spacing[d];
        }
        out
    }

    /// walk every grid point once, row-major
    pub fn for_each_index(&self, mut f: impl FnMut([usize; NDIM])) {
        let shape = self.shape;
        let mut idx = [0usize; NDIM];
        loop {
            f(idx);
            let mut carry = true;
            for d in (0..NDIM).rev() {
                if carry {
                    idx[d] += 1;
                    if idx[d] < shape[d] {
                        carry = false;
                    } else {
                        idx[d] = 0;
                    }
                }
            }
            if carry {
                break;
            }
        }
    }
}