use crate::grid::boundary::{BoundaryConfig, BoundaryValues};
use crate::grid::Field;
use crate::numerics::derivatives::central_partial;
use super::tensor::{Tensor2, Vector};

/// Christoffel symbols of the 4-metric. data[a][b][c] = Gamma^a_bc,
/// with a, b, c spacetime indices (0 for time, 1..=D spatial).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Christoffel<const N: usize> {
    pub data: [[[f64; N]; N]; N],
}

impl<const N: usize> Default for Christoffel<N> {
    fn default() -> Self {
        Self::zero()
    }
}

pub type ChristoffelField<const D: usize> = Field<Christoffel<4>, D>;

impl<const N: usize> Christoffel<N> {
    /// all symbols zero
    pub fn zero() -> Self {
        Self {
            data: [[[0.0; N]; N]; N],
        }
    }
}

/// walk every grid point once
fn for_each_index<const D: usize>(shape: [usize; D], mut f: impl FnMut([usize; D])) {
    let mut idx = [0usize; D];
    loop {
        f(idx);
        let mut carry = true;
        for d in (0..D).rev() {
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

/// copy one component of a vector field into a scalar field
fn shift_component<const D: usize>(shift: &Field<Vector<D>, D>, m: usize) -> Field<f64, D> {
    let shape = shift.grid.shape;
    let mut out = Field::zeros(shift.grid.clone());
    for_each_index(shape, |idx| out.set(idx, (*shift.get(idx))[m]));
    out
}

/// copy one component of a tensor field into a scalar field
fn spatial_component<const D: usize>(
    spatial: &Field<Tensor2<D>, D>,
    i: usize,
    j: usize,
) -> Field<f64, D> {
    let shape = spatial.grid.shape;
    let mut out = Field::zeros(spatial.grid.clone());
    for_each_index(shape, |idx| out.set(idx, spatial.get(idx)[(i, j)]));
    out
}

/// invert a 4x4 matrix, panics if singular
fn invert4(m: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut a = [[0.0f64; 8]; 4];
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = m[i][j];
        }
        a[i][i + 4] = 1.0;
    }
    for col in 0..4 {
        let pivot = (col..4)
            .max_by(|&x, &y| a[x][col].abs().partial_cmp(&a[y][col].abs()).unwrap())
            .unwrap();
        a.swap(col, pivot);
        let d = a[col][col];
        assert!(d.abs() > 1e-15, "singular metric");
        for j in 0..8 {
            a[col][j] /= d;
        }
        for row in 0..4 {
            if row != col && a[row][col] != 0.0 {
                let factor = a[row][col];
                for j in 0..8 {
                    a[row][j] -= factor * a[col][j];
                }
            }
        }
    }
    let mut inv = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            inv[i][j] = a[i][j + 4];
        }
    }
    inv
}

/// Christoffel symbols of the 3+1 metric on the whole grid.
/// Gamma = 1/2 g^(ad) (dg_bcd + dg_cbd - dg_dbc), using central
/// differences off the lapse, shift and spatial metric fields.
pub fn christoffel_symbols<const D: usize>(
    lapse: &Field<f64, D>,
    shift: &Field<Vector<D>, D>,
    spatial: &Field<Tensor2<D>, D>,
    boundary: &BoundaryConfig<D>,
    boundary_values: &BoundaryValues<D>,
) -> ChristoffelField<D> {
    let shape = lapse.grid.shape;

    // first derivatives d/dx^k of each metric variable
    let dalpha: Vec<Field<f64, D>> = (0..D)
        .map(|k| central_partial(lapse, k, boundary, boundary_values))
        .collect();
    let mut dbeta: Vec<Vec<Field<f64, D>>> = Vec::with_capacity(D);
    for m in 0..D {
        let beta_m = shift_component(shift, m);
        dbeta.push(
            (0..D)
                .map(|k| central_partial(&beta_m, k, boundary, boundary_values))
                .collect(),
        );
    }
    let mut dgamma: Vec<Vec<Vec<Field<f64, D>>>> = Vec::with_capacity(D);
    for i in 0..D {
        dgamma.push(Vec::with_capacity(D));
        for j in 0..D {
            let gamma_ij = spatial_component(spatial, i, j);
            dgamma[i].push(
                (0..D)
                    .map(|k| central_partial(&gamma_ij, k, boundary, boundary_values))
                    .collect(),
            );
        }
    }

    let mut result: ChristoffelField<D> = Field::zeros(lapse.grid.clone());
    let mut dg = [[0.0f64; 4]; 4];
    let mut dg_list = vec![[[0.0f64; 4]; 4]; 4];
    let mut beta_lower = [0.0f64; 4];
    let mut d_beta_lower = [0.0f64; 4];
    let mut g = [[0.0f64; 4]; 4];
    let mut symbols = Christoffel::<4>::zero();

    for_each_index(shape, |idx| {
        let alpha = *lapse.get(idx);
        let beta = shift.get(idx);
        let gamma = spatial.get(idx);

        // lower the shift with the spatial metric: beta_i = gamma_ij beta^j
        for i in 0..D {
            let mut s = 0.0;
            for j in 0..D {
                s += gamma[(i, j)] * beta[j];
            }
            beta_lower[i] = s;
        }
        let beta_sq = (0..D).fold(0.0, |acc, i| acc + beta_lower[i] * beta[i]);

        // the 3+1 four-metric g_ab
        for a in 0..=D {
            for b in 0..=D {
                g[a][b] = 0.0;
            }
        }
        g[0][0] = -alpha * alpha + beta_sq;
        for i in 1..=D {
            g[0][i] = beta_lower[i - 1];
            g[i][0] = beta_lower[i - 1];
        }
        for i in 1..=D {
            for j in 1..=D {
                g[i][j] = gamma[(i - 1, j - 1)];
            }
        }

        // derivatives of g_ab; index 0 is time and stays zero
        dg_list[0] = [[0.0f64; 4]; 4];
        for k in 0..D {
            let d_alpha = *dalpha[k].get(idx);
            // d/dx^k of the lower shift: gamma_mj d beta^j + d gamma_mj beta^j
            for m in 0..D {
                let mut s = 0.0;
                for j in 0..D {
                    let d_gamma_mj = *dgamma[m][j][k].get(idx);
                    let d_beta_j = *dbeta[j][k].get(idx);
                    s += d_gamma_mj * beta[j] + gamma[(m, j)] * d_beta_j;
                }
                d_beta_lower[m] = s;
            }
            let d_beta_sq = (0..D).fold(0.0, |acc, m| acc + beta[m] * d_beta_lower[m] * 2.0);

            for a in 0..=D {
                for b in 0..=D {
                    dg[a][b] = 0.0;
                }
            }
            dg[0][0] = -2.0 * alpha * d_alpha + d_beta_sq;
            for i in 1..=D {
                dg[0][i] = d_beta_lower[i - 1];
                dg[i][0] = d_beta_lower[i - 1];
            }
            for i in 1..=D {
                for j in 1..=D {
                    dg[i][j] = *dgamma[i - 1][j - 1][k].get(idx);
                }
            }
            dg_list[k + 1] = dg;
        }

        let ginv = invert4(&g);

        // Gamma^a_bc = 1/2 g^(ad) (dg_bcd + dg_cbd - dg_dbc)
        for a in 0..=D {
            for b in 0..=D {
                for c in 0..=D {
                    let mut s = 0.0;
                    for d in 0..=D {
                        s += ginv[a][d] * (dg_list[b][d][c] + dg_list[c][b][d] - dg_list[d][b][c]);
                    }
                    symbols.data[a][b][c] = 0.5 * s;
                }
            }
        }
        result.set(idx, symbols);
    });

    result
}