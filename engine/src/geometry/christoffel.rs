use super::metric::{LapseField, ShiftField, SliceMetric, SpacetimeMetric};
use super::tensor::{invert, Tensor2, Vector};
use crate::grid::boundary::{BoundaryConfig, BoundaryValues};
use crate::grid::Field;
use crate::numerics::derivatives::central_partial;

/// Christoffel symbols of the 4-metric. data[a][b][c] = Gamma^a_bc,
/// with a, b, c spacetime indices (0 for time, 1..=D spatial).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Christoffel {
    pub data: [[[f64; 4]; 4]; 4],
}

pub type ChristoffelField<const D: usize> = Field<Christoffel, D>;

impl Christoffel {
    /// all symbols zero
    pub fn zero() -> Self {
        Self {
            data: [[[0.0; 4]; 4]; 4],
        }
    }
}

/// copy one component of a vector field into a scalar field
fn shift_component<const D: usize>(shift: &Field<Vector<D>, D>, m: usize) -> Field<f64, D> {
    let mut out = Field::zeros(shift.grid.clone());
    shift
        .grid
        .for_each_index(|idx| out.set(idx, (*shift.get(idx))[m]));
    out
}

/// copy one component of a tensor field into a scalar field
fn spatial_component<const D: usize>(
    spatial: &Field<Tensor2<D>, D>,
    i: usize,
    j: usize,
) -> Field<f64, D> {
    let mut out = Field::zeros(spatial.grid.clone());
    spatial
        .grid
        .for_each_index(|idx| out.set(idx, spatial.get(idx)[(i, j)]));
    out
}

/// Christoffel symbols of the 3+1 metric on the whole grid.
/// Gamma = 1/2 g^(ad) (dg_bcd + dg_cbd - dg_dbc), using central
/// differences off the lapse, shift and spatial metric fields.
pub fn christoffel_symbols<const D: usize>(
    lapse: &LapseField<D>,
    shift: &ShiftField<D>,
    spatial: &SliceMetric<D>,
    boundary: &BoundaryConfig<D>,
    boundary_values: &BoundaryValues<D>,
) -> ChristoffelField<D> {
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
    let mut d_beta_lower = [0.0f64; 4];
    let mut symbols = Christoffel::zero();

    lapse.grid.for_each_index(|idx| {
        let alpha = *lapse.get(idx);
        let beta = shift.get(idx);
        let gamma = spatial.get(idx);

        // the 3+1 four-metric g_ab, assembled by the metric module
        let g = SpacetimeMetric::from_3plus1(alpha, beta, gamma).as_4x4();

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

        let ginv = invert(&g);

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