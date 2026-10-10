
use std::sync::Arc;

use crate::geometry::christoffel::spatial_christoffel_symbols;
use crate::geometry::metric::{
    LapseField,
    ShiftField,
};
use crate::geometry::tensor::{det, invert, Tensor2, Vector};
use crate::grid::boundary::{BoundaryConfig, BoundaryValues};
use crate::grid::{Field, Grid};
use crate::numerics::rk4::EvolutionState;

/// copy a slice tensor into plain matrix form for `invert` and `det`
fn matrix<const D: usize>(t: &Tensor2<D>) -> [[f64; D]; D] {
    let mut m = [[0.0; D]; D];
    for i in 0..D {
        for j in 0..D {
            m[i][j] = t[(i, j)];
        }
    }
    m
}

/// State variables for the standard 3D BSSN formulation.
///
/// Algebraic constraints to maintain:
/// det(conformal_metric) = 1
/// conformal_metric^ij * conformal_a_ij = 0
#[derive(Debug, Clone)]
pub struct BssnState<const D: usize> {
    pub grid: Arc<Grid<D>>,

    /// Conformal exponent phi.
    pub phi: Field<f64, D>,

    /// Conformal spatial metric: gamma_ij = exp(4 phi) * gtilde_ij.
    pub conformal_metric: Field<Tensor2<D>, D>,

    /// Trace of the extrinsic curvature: K = gamma^ij K_ij.
    pub trace_k: Field<f64, D>,

    /// Conformal trace-free extrinsic curvature.
    pub conformal_a: Field<Tensor2<D>, D>,

    /// Conformal connection functions Gamma^i.
    pub conformal_gamma: Field<Vector<D>, D>,

    /// Lapse alpha.
    pub lapse: LapseField<D>,

    /// Shift beta^i.
    pub shift: ShiftField<D>,
}

impl<const D: usize> BssnState<D> {
    /// Construct flat-space BSSN data.
    ///
    /// phi = 0
    /// gtilde_ij = delta_ij
    /// K = 0
    /// Atilde_ij = 0
    /// Gamma^i = 0
    /// alpha = 1
    /// beta^i = 0
    pub fn minkowski(grid: Arc<Grid<D>>) -> Self {
        let mut identity = Tensor2::<D>::zero();

        for i in 0..D {
            identity[(i, i)] = 1.0;
        }

        Self {
            phi: Field::new(grid.clone(), 0.0),
            conformal_metric: Field::new(
                grid.clone(),
                identity,
            ),
            trace_k: Field::new(grid.clone(), 0.0),
            conformal_a: Field::new(
                grid.clone(),
                Tensor2::<D>::zero(),
            ),
            conformal_gamma: Field::new(
                grid.clone(),
                Vector::<D>::zero(),
            ),
            lapse: Field::new(grid.clone(), 1.0),
            shift: Field::new(
                grid.clone(),
                Vector::<D>::zero(),
            ),
            grid,
        }
    }

    /// ADM -> BSSN conversion.
    ///
    /// The conformal split of the spatial metric is exact:
    ///
    ///   phi = ln(det(gamma)) / 12
    ///   gtilde_ij = exp(-4 phi) gamma_ij = det^{-1/3} gamma_ij
    ///
    /// so det(gtilde) = 1 by construction. The trace and trace-free part of
    /// the extrinsic curvature, and the conformal connection functions:
    ///
    ///   K = gamma^{ij} K_ij
    ///   A tilde_ij = K_ij - (1/3) gamma_ij K      (trace-free wrt gamma)
    ///   Gamma^i = gtilde^{jk} Gamma^i_jk(gtilde)
    ///
    /// `boundary` is needed only for the Christoffel symbols of gtilde, on
    /// which the pointwise split itself does not depend.
    pub fn from_adm(
        adm: &EvolutionState<D>,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> Self {
        let grid = adm.grid.clone();

        let mut phi = Field::new(grid.clone(), 0.0);
        let mut conformal_metric = Field::new(grid.clone(), Tensor2::zero());
        let mut trace_k = Field::new(grid.clone(), 0.0);
        let mut conformal_a = Field::new(grid.clone(), Tensor2::zero());

        grid.for_each_index(|idx| {
            let gamma = *adm.gamma.get(idx);
            let k = *adm.k.get(idx);

            let g = matrix(&gamma);
            let ginv = invert(&g);
            let gdet = det(&g);
            assert!(gdet > 0.0, "the spatial metric must be positive definite");

            let p = gdet.ln() / 12.0;
            let sc = gdet.powf(-1.0 / 3.0);
            phi.set(idx, p);

            let mut gtilde = Tensor2::zero();
            let mut at = Tensor2::zero();
            let mut trk = 0.0;
            for i in 0..D {
                for j in 0..D {
                    gtilde[(i, j)] = sc * gamma[(i, j)];
                    trk += ginv[i][j] * k[(i, j)];
                }
            }
            for i in 0..D {
                for j in 0..D {
                    at[(i, j)] = k[(i, j)] - gamma[(i, j)] * trk / 3.0;
                }
            }
            conformal_metric.set(idx, gtilde);
            trace_k.set(idx, trk);
            conformal_a.set(idx, at);
        });

        // Gamma^i = gtilde^{jk} Gamma^i_jk(gtilde)
        let gtilde_christoffel = spatial_christoffel_symbols(
            &conformal_metric,
            boundary,
            boundary_values,
        );
        let mut conformal_gamma = Field::new(grid.clone(), Vector::zero());
        grid.for_each_index(|idx| {
            let gtilde = *conformal_metric.get(idx);
            let ginvt = invert(&matrix(&gtilde));
            let syms = *gtilde_christoffel.get(idx);
            let mut g = Vector::zero();
            for i in 0..D {
                let mut s = 0.0;
                for j in 0..D {
                    for k in 0..D {
                        s += ginvt[j][k] * syms.data[i][j][k];
                    }
                }
                g[i] = s;
            }
            conformal_gamma.set(idx, g);
        });

        Self {
            grid,
            phi,
            conformal_metric,
            trace_k,
            conformal_a,
            conformal_gamma,
            lapse: adm.lapse.clone(),
            shift: adm.shift.clone(),
        }
    }

    /// BSSN -> ADM conversion.
    ///
    ///   gamma_ij = exp(4 phi) gtilde_ij
    ///   K_ij = A tilde_ij + (1/3) gamma_ij K
    pub fn to_adm(&self) -> EvolutionState<D> {
        let mut gamma = Field::new(self.grid.clone(), Tensor2::zero());
        let mut k = Field::new(self.grid.clone(), Tensor2::zero());

        self.grid.for_each_index(|idx| {
            let p = *self.phi.get(idx);
            let e4 = (4.0 * p).exp();
            let gtilde = *self.conformal_metric.get(idx);
            let trk = *self.trace_k.get(idx);
            let at = *self.conformal_a.get(idx);

            let mut g = Tensor2::zero();
            let mut kk = Tensor2::zero();
            for i in 0..D {
                for j in 0..D {
                    let gij = e4 * gtilde[(i, j)];
                    g[(i, j)] = gij;
                    kk[(i, j)] = at[(i, j)] + gij * trk / 3.0;
                }
            }
            gamma.set(idx, g);
            k.set(idx, kk);
        });

        EvolutionState {
            grid: self.grid.clone(),
            gamma,
            k,
            lapse: self.lapse.clone(),
            shift: self.shift.clone(),
        }
    }
}
