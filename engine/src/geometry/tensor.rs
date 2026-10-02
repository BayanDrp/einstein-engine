use std::ops::{Index, IndexMut};

// fixed-size vector, indexed by usize
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector<const D: usize> {
    data: [f64; D],
}

impl<const D: usize> Vector<D> {
    pub fn new(data: [f64; D]) -> Self {
        Self { data }
    }

    pub fn zero() -> Self {
        Self { data: [0.0; D] }
    }
}

impl<const D: usize> Index<usize> for Vector<D> {
    type Output = f64;

    fn index(&self, i: usize) -> &f64 {
        &self.data[i]
    }
}

impl<const D: usize> IndexMut<usize> for Vector<D> {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.data[i]
    }
}

// second-rank tensor (matrix), indexed by a (row, col) pair
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tensor2<const D: usize> {
    data: [[f64; D]; D],
}

impl<const D: usize> Tensor2<D> {
    pub fn new(data: [[f64; D]; D]) -> Self {
        Self { data }
    }

    pub fn zero() -> Self {
        Self { data: [[0.0; D]; D] }
    }
}

impl<const D: usize> Index<(usize, usize)> for Tensor2<D> {
    type Output = f64;

    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i][j]
    }
}

impl<const D: usize> IndexMut<(usize, usize)> for Tensor2<D> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i][j]
    }
}

/// invert an NxN matrix by Gauss-Jordan, panics if singular
pub fn invert<const N: usize>(m: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut a = *m;
    let mut inv = [[0.0f64; N]; N];
    for i in 0..N {
        inv[i][i] = 1.0;
    }
    for col in 0..N {
        let pivot = (col..N)
            .max_by(|&x, &y| a[x][col].abs().partial_cmp(&a[y][col].abs()).unwrap())
            .unwrap();
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let d = a[col][col];
        assert!(d.abs() > 1e-15, "singular matrix");
        for j in 0..N {
            a[col][j] /= d;
            inv[col][j] /= d;
        }
        for row in 0..N {
            if row != col && a[row][col] != 0.0 {
                let factor = a[row][col];
                for j in 0..N {
                    a[row][j] -= factor * a[col][j];
                    inv[row][j] -= factor * inv[col][j];
                }
            }
        }
    }
    inv
}