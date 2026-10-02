use super::christoffel::ChristoffelField;
use crate::grid::boundary::{BoundaryConfig, BoundaryValues};
use crate::grid::Field;
use crate::numerics::derivatives::central_partial;

/// Riemann curvature tensor of the spacetime metric.
/// data[mu][nu][rho][sigma] = R^mu_nu rho sigma. N is the number
/// of spacetime dimensions (3 for 2+1, 4 for 3+1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Riemann<const N: usize> {
    pub data: [[[[f64; N]; N]; N]; N],
}

pub type RiemannField<const N: usize, const D: usize> = Field<Riemann<N>, D>;

impl<const N: usize> Default for Riemann<N> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const N: usize> Riemann<N> {
    /// all components zero
    pub fn zero() -> Self {
        Self {
            data: [[[[0.0; N]; N]; N]; N],
        }
    }
}

/// copy one Christoffel component Gamma^mu_nu rho into a scalar field,
/// so it can be differentiated like a field itself
fn gamma_component<const D: usize, const N: usize>(
    gamma: &ChristoffelField<N, D>,
    mu: usize,
    nu: usize,
    rho: usize,
) -> Field<f64, D> {
    let mut out = Field::zeros(gamma.grid.clone());
    gamma
        .grid
        .for_each_index(|idx| out.set(idx, gamma.get(idx).data[mu][nu][rho]));
    out
}

/// Riemann tensor on the whole grid, built from its Christoffel symbols:
///
///   R^mu_nu rho sigma = d_rho Gamma^mu_nu sigma - d_sigma Gamma^mu_nu rho
///                     + Gamma^mu_lambda rho Gamma^lambda_nu sigma
///                     - Gamma^mu_lambda sigma Gamma^lambda_nu rho
///
/// index 0 is time; the metric is assumed static, so the time
/// derivative of any Christoffel symbol is zero.
pub fn riemann_from_christoffel<const D: usize, const N: usize>(
    gamma: &ChristoffelField<N, D>,
    boundary: &BoundaryConfig<D>,
    boundary_values: &BoundaryValues<D>,
) -> RiemannField<N, D> {
    // d_gamma[mu][nu][rho][k] = d/dx^k of Gamma^mu_nu rho, k over spatial dims
    let mut d_gamma = Vec::new();
    for mu in 0..N {
        let mut by_nu = Vec::new();
        for nu in 0..N {
            let mut by_rho = Vec::new();
            for rho in 0..N {
                let comp = gamma_component(gamma, mu, nu, rho);
                by_rho.push(
                    (0..D)
                        .map(|k| central_partial(&comp, k, boundary, boundary_values))
                        .collect::<Vec<_>>(),
                );
            }
            by_nu.push(by_rho);
        }
        d_gamma.push(by_nu);
    }

    let mut result: RiemannField<N, D> = Field::zeros(gamma.grid.clone());
    gamma.grid.for_each_index(|idx| {
        let g = gamma.get(idx);
        let mut r = Riemann::<N>::zero();
        for mu in 0..N {
            for nu in 0..N {
                for rho in 0..N {
                    for sigma in 0..N {
                        // d_rho Gamma^mu_nu sigma - d_sigma Gamma^mu_nu rho
                        let mut s = 0.0;
                        if rho > 0 {
                            s += *d_gamma[mu][nu][sigma][rho - 1].get(idx);
                        }
                        if sigma > 0 {
                            s -= *d_gamma[mu][nu][rho][sigma - 1].get(idx);
                        }
                        // Gamma^mu_lambda rho Gamma^lambda_nu sigma
                        // - Gamma^mu_lambda sigma Gamma^lambda_nu rho
                        for lambda in 0..N {
                            s += g.data[mu][lambda][rho] * g.data[lambda][nu][sigma]
                                - g.data[mu][lambda][sigma] * g.data[lambda][nu][rho];
                        }
                        r.data[mu][nu][rho][sigma] = s;
                    }
                }
            }
        }
        result.set(idx, r);
    });

    result
}