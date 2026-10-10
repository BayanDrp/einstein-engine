
use std::sync::Arc;

use crate::geometry::christoffel::spatial_christoffel_symbols;
use crate::geometry::metric::{
    LapseField,
    ShiftField,
};
use crate::geometry::ricci::{ricci_from_riemann, Ricci, RicciField};
use crate::geometry::riemann::riemann_from_christoffel;
use crate::geometry::tensor::{det, invert, Tensor2, Vector};
use crate::grid::boundary::{apply_nonperiodic, BoundaryConfig, BoundaryValues};
use crate::grid::{Field, Grid};
use crate::numerics::derivatives::central_partial;
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

/// copy one (i, j) component of a tensor field into a scalar field,
/// so it can be differentiated
fn tensor_component<const D: usize>(
    field: &Field<Tensor2<D>, D>,
    i: usize,
    j: usize,
) -> Field<f64, D> {
    let mut out = Field::zeros(field.grid.clone());
    field
        .grid
        .for_each_index(|idx| out.set(idx, field.get(idx)[(i, j)]));
    out
}

/// copy one k-th component of a vector field into a scalar field
fn vector_component<const D: usize>(field: &Field<Vector<D>, D>, k: usize) -> Field<f64, D> {
    let mut out = Field::zeros(field.grid.clone());
    field
        .grid
        .for_each_index(|idx| out.set(idx, (*field.get(idx))[k]));
    out
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
    ///   A tilde_ij = e^{-4 phi} (K_ij - (1/3) gamma_ij K)
    ///                (trace-free wrt gamma, conformally weighted)
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
                    // sc = gdet^{-1/3} = e^{-4 phi}
                    at[(i, j)] = sc * (k[(i, j)] - gamma[(i, j)] * trk / 3.0);
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
    ///   K_ij = exp(4 phi) A tilde_ij + (1/3) gamma_ij K
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
                    kk[(i, j)] = e4 * at[(i, j)] + gij * trk / 3.0;
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

    /// The split of the physical Ricci tensor into its conformal parts,
    ///
    ///   R_ij(gamma) = Rtilde_ij + Rphi_ij
    ///
    /// with gamma_ij = e^{4 phi} gtilde_ij. `Rtilde_ij` is the Ricci tensor
    /// of the conformal metric with the hyperbolic principal part: the
    /// direct Ricci of gtilde plus the evolved-connection correction
    ///
    ///   Rtilde_ij = R_ij(gtilde) + gtilde_k(i d_j) (Gamma^k - Gamma(n)^k),
    ///
    /// where `Gamma^k` is the evolved connection and `Gamma(n)^k` is the
    /// contracted Christoffel symbol recomputed from the current conformal
    /// metric. On the constraint surface the correction vanishes and this
    /// is exactly the Ricci of gtilde; off it the correction supplies the
    /// BSSN principal part. `Rphi_ij` carries the conformal factor:
    ///
    ///   Rphi_ij = -2 (d_i d_j phi - Gamma^k_ij d_k phi)
    ///             -2 gtilde_ij gtilde^kl (d_k d_l phi - Gamma^m_kl d_m phi)
    ///             + 4 d_i phi d_j phi
    ///             - 4 gtilde_ij gtilde^kl d_k phi d_l phi
    ///
    /// all partial derivatives carried by central differences.
    pub fn conformal_ricci(
        &self,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> (RicciField<D, D>, RicciField<D, D>) {
        assert!(D == 3, "the BSSN equations are implemented for 3+1 only");

        // d_k phi and d_i d_j phi
        let mut d_phi: Vec<Field<f64, D>> = Vec::with_capacity(D);
        for k in 0..D {
            d_phi.push(central_partial(&self.phi, k, boundary, boundary_values));
        }
        let mut hess_phi: Vec<Vec<Field<f64, D>>> = Vec::with_capacity(D);
        for i in 0..D {
            let mut row = Vec::with_capacity(D);
            for j in 0..D {
                row.push(central_partial(&d_phi[j], i, boundary, boundary_values));
            }
            hess_phi.push(row);
        }

        let gtilde_christoffel =
            spatial_christoffel_symbols(&self.conformal_metric, boundary, boundary_values);

        // Rtilde_ij is the Ricci tensor of the conformal metric, computed
        // directly from gtilde with the same Christoffel -> Riemann -> Ricci
        // pipeline the ADM equations use for the physical Ricci tensor. This
        // is exact by construction; the evolved Gamma^i is still evolved and
        // differentiated for the constraint/hyperbolicity structure.
        let riemann = riemann_from_christoffel::<D, D>(
            &gtilde_christoffel,
            boundary,
            boundary_values,
        );
        let ricc_direct: RicciField<D, D> = ricci_from_riemann::<D, D>(&riemann);

        // Gamma(n)^i recomputed from the current conformal metric
        let mut gamma_n_field: Field<Vector<D>, D> =
            Field::new(self.grid.clone(), Vector::zero());
        self.grid.for_each_index(|idx| {
            let gtilde = *self.conformal_metric.get(idx);
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
            gamma_n_field.set(idx, g);
        });

        // delta Gamma^i = evolved - recomputed; its symmetrized gradient is
        // the hyperbolic correction to the direct conformal Ricci
        let mut delta_field: Field<Vector<D>, D> =
            Field::new(self.grid.clone(), Vector::zero());
        self.grid.for_each_index(|idx| {
            let mut d = Vector::zero();
            let ev = *self.conformal_gamma.get(idx);
            let rn = *gamma_n_field.get(idx);
            for i in 0..D {
                d[i] = ev[i] - rn[i];
            }
            delta_field.set(idx, d);
        });
        let mut d_delta: Vec<Vec<Field<f64, D>>> = Vec::with_capacity(D);
        for a in 0..D {
            let comp = vector_component(&delta_field, a);
            let mut row = Vec::with_capacity(D);
            for k in 0..D {
                row.push(central_partial(&comp, k, boundary, boundary_values));
            }
            d_delta.push(row);
        }

        let mut ricc_t: RicciField<D, D> = Field::zeros(self.grid.clone());
        let mut ricc_p: RicciField<D, D> = Field::zeros(self.grid.clone());

        self.grid.for_each_index(|idx| {
            let gtilde = *self.conformal_metric.get(idx);
            let ginvt = invert(&matrix(&gtilde));
            let syms = *gtilde_christoffel.get(idx);
            let base = *ricc_direct.get(idx);

            let mut rt = Ricci::zero();
            for i in 0..D {
                for j in 0..D {
                    let mut corr = 0.0;
                    for k in 0..D {
                        corr += 0.5
                            * (gtilde[(k, i)] * *d_delta[k][j].get(idx)
                                + gtilde[(k, j)] * *d_delta[k][i].get(idx));
                    }
                    rt.data[i][j] = base.data[i][j] + corr;
                }
            }

            let mut rp = Ricci::zero();

            for i in 0..D {
                for j in 0..D {
                    // Rphi_ij
                    let mut dd_phi = *hess_phi[i][j].get(idx);
                    for k in 0..D {
                        dd_phi -= syms.data[k][i][j] * *d_phi[k].get(idx);
                    }
                    let mut lap_phi = 0.0;
                    for k in 0..D {
                        for l in 0..D {
                            let mut d2 = *hess_phi[k][l].get(idx);
                            for m in 0..D {
                                d2 -= syms.data[m][k][l] * *d_phi[m].get(idx);
                            }
                            lap_phi += ginvt[k][l] * d2;
                        }
                    }
                    let mut dot_phi = 0.0;
                    for k in 0..D {
                        for l in 0..D {
                            dot_phi += ginvt[k][l] * *d_phi[k].get(idx) * *d_phi[l].get(idx);
                        }
                    }
                    rp.data[i][j] = -2.0 * dd_phi - 2.0 * gtilde[(i, j)] * lap_phi
                        + 4.0 * *d_phi[i].get(idx) * *d_phi[j].get(idx)
                        - 4.0 * gtilde[(i, j)] * dot_phi;
                }
            }

            ricc_t.set(idx, rt);
            ricc_p.set(idx, rp);
        });

        (ricc_t, ricc_p)
    }

    /// Right-hand side of the vacuum BSSN equations in 3+1, in the standard
    /// form that takes the conformally weighted extrinsic curvature
    ///
    ///   A tilde_ij = e^{-4 phi} (K_ij - (1/3) gamma_ij K),
    ///
    /// as in e.g. Rosswog & Diener, arXiv:2404.15952:
    ///
    ///   d_t phi = -1/6 (alpha K - d_i beta^i) + beta^i d_i phi
    ///
    ///   d_t gtilde_ij = -2 alpha A tilde_ij
    ///                   + gtilde_ik d_j beta^k + gtilde_jk d_i beta^k
    ///                   - (2/3) gtilde_ij d_k beta^k + beta^k d_k gtilde_ij
    ///
    ///   d_t K = -e^{-4 phi} ( gtilde^ij (d_i d_j alpha + 2 d_i phi d_j alpha)
    ///                         - Gamma(n)^i d_i alpha )
    ///           + alpha (A tilde^a_b A tilde^b_a + K^2/3) + beta^i d_i K
    ///
    ///   d_t A tilde_ij = e^{-4 phi} [ -d_i d_j alpha + Gamma^k_ij d_k alpha
    ///                       + 2 (d_i alpha d_j phi + d_j alpha d_i phi)
    ///                       + alpha (Rtilde_ij + Rphi_ij) ]^TF
    ///           + alpha (K A tilde_ij - 2 A tilde_ik A tilde^k_j)
    ///           + A tilde_ik d_j beta^k + A tilde_jk d_i beta^k
    ///           - (2/3) A tilde_ij d_k beta^k + beta^k d_k A tilde_ij
    ///
    ///   d_t Gamma^i = -2 A tilde^ij d_j alpha
    ///           + 2 alpha (Gamma^i_jk A tilde^jk - (2/3) gtilde^ij d_j K
    ///                       + 6 A tilde^ij d_j phi)
    ///           + gtilde^jk d_j d_k beta^i + (1/3) gtilde^ij d_j d_k beta^k
    ///           - Gamma(n)^j d_j beta^i + (2/3) Gamma(n)^i d_j beta^j
    ///           + beta^j d_j Gamma^i
    ///
    /// Gauge: 1+log slicing for the lapse, frozen shift, exactly as in the
    /// ADM formulation. The "^TF" superscript is the trace-free part with
    /// respect to the conformal metric, and `Gamma(n)^i` is rebuilt from the
    /// current conformal metric while the evolved `Gamma^i` is only ever
    /// differentiated. Matter terms are omitted.
    pub fn rhs(
        &self,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> BssnState<D> {
        assert!(D == 3, "the BSSN equations are implemented for 3+1 only");

        let (ricc_t, ricc_p) = self.conformal_ricci(boundary, boundary_values);

        let mut d_phi: Vec<Field<f64, D>> = Vec::with_capacity(D);
        for k in 0..D {
            d_phi.push(central_partial(&self.phi, k, boundary, boundary_values));
        }
        let mut d_alpha: Vec<Field<f64, D>> = Vec::with_capacity(D);
        for k in 0..D {
            d_alpha.push(central_partial(&self.lapse, k, boundary, boundary_values));
        }
        let mut hess_alpha: Vec<Vec<Field<f64, D>>> = Vec::with_capacity(D);
        for i in 0..D {
            let mut row = Vec::with_capacity(D);
            for j in 0..D {
                row.push(central_partial(&d_alpha[j], i, boundary, boundary_values));
            }
            hess_alpha.push(row);
        }
        let mut d_tracek: Vec<Field<f64, D>> = Vec::with_capacity(D);
        for k in 0..D {
            d_tracek.push(central_partial(&self.trace_k, k, boundary, boundary_values));
        }
        let mut d_gtilde: Vec<Vec<Vec<Field<f64, D>>>> = Vec::with_capacity(D);
        for k in 0..D {
            let mut by_i = Vec::with_capacity(D);
            for i in 0..D {
                let mut by_j = Vec::with_capacity(D);
                for j in 0..D {
                    let comp = tensor_component(&self.conformal_metric, i, j);
                    by_j.push(central_partial(&comp, k, boundary, boundary_values));
                }
                by_i.push(by_j);
            }
            d_gtilde.push(by_i);
        }
        let mut d_at: Vec<Vec<Vec<Field<f64, D>>>> = Vec::with_capacity(D);
        for k in 0..D {
            let mut by_i = Vec::with_capacity(D);
            for i in 0..D {
                let mut by_j = Vec::with_capacity(D);
                for j in 0..D {
                    let comp = tensor_component(&self.conformal_a, i, j);
                    by_j.push(central_partial(&comp, k, boundary, boundary_values));
                }
                by_i.push(by_j);
            }
            d_at.push(by_i);
        }
        let mut d_gfn: Vec<Vec<Field<f64, D>>> = Vec::with_capacity(D);
        for a in 0..D {
            let comp = vector_component(&self.conformal_gamma, a);
            let mut row = Vec::with_capacity(D);
            for k in 0..D {
                row.push(central_partial(&comp, k, boundary, boundary_values));
            }
            d_gfn.push(row);
        }
        // d_k beta^i and d_l d_k beta^i
        let mut d_beta: Vec<Vec<Field<f64, D>>> = Vec::with_capacity(D);
        for i in 0..D {
            let mut row = Vec::with_capacity(D);
            let comp = vector_component(&self.shift, i);
            for k in 0..D {
                row.push(central_partial(&comp, k, boundary, boundary_values));
            }
            d_beta.push(row);
        }
        let mut d2_beta: Vec<Vec<Vec<Field<f64, D>>>> = Vec::with_capacity(D);
        for i in 0..D {
            let mut by_k = Vec::with_capacity(D);
            for k in 0..D {
                let mut by_l = Vec::with_capacity(D);
                for l in 0..D {
                    by_l.push(central_partial(&d_beta[i][k], l, boundary, boundary_values));
                }
                by_k.push(by_l);
            }
            d2_beta.push(by_k);
        }

        let gtilde_christoffel =
            spatial_christoffel_symbols(&self.conformal_metric, boundary, boundary_values);

        let mut phi_rhs = Field::new(self.grid.clone(), 0.0);
        let mut metric_rhs: Field<Tensor2<D>, D> = Field::new(self.grid.clone(), Tensor2::zero());
        let mut trace_k_rhs = Field::new(self.grid.clone(), 0.0);
        let mut conformal_a_rhs: Field<Tensor2<D>, D> =
            Field::new(self.grid.clone(), Tensor2::zero());
        let mut conformal_gamma_rhs: Field<Vector<D>, D> =
            Field::new(self.grid.clone(), Vector::zero());
        let mut lapse_rhs = Field::new(self.grid.clone(), 0.0);

        self.grid.for_each_index(|idx| {
            let phi = *self.phi.get(idx);
            let e4 = (4.0 * phi).exp();
            let em4 = 1.0 / e4;

            let gtilde = *self.conformal_metric.get(idx);
            let ginvt = invert(&matrix(&gtilde));
            let at = *self.conformal_a.get(idx);
            let trk = *self.trace_k.get(idx);
            let alpha = *self.lapse.get(idx);
            let beta = *self.shift.get(idx);
            let syms = *gtilde_christoffel.get(idx);
            let rt = ricc_t.get(idx);
            let rp = ricc_p.get(idx);

            // Gamma(n)^i: contracted connection of the current conformal metric
            let mut gamma_n = [0.0f64; D];
            for i in 0..D {
                for j in 0..D {
                    for k in 0..D {
                        gamma_n[i] += ginvt[j][k] * syms.data[i][j][k];
                    }
                }
            }

            let mut div_beta = 0.0;
            for d in 0..D {
                div_beta += *d_beta[d][d].get(idx);
            }

            // mixed and fully upper versions of A tilde
            let mut at_mixed = [[0.0f64; D]; D];
            let mut at_upper = [[0.0f64; D]; D];
            for a in 0..D {
                for b in 0..D {
                    for m in 0..D {
                        at_mixed[a][b] += ginvt[a][m] * at[(m, b)];
                    }
                    for p in 0..D {
                        for q in 0..D {
                            at_upper[a][b] += ginvt[a][p] * ginvt[b][q] * at[(p, q)];
                        }
                    }
                }
            }

            // ---- d_t phi ------------------------------------------------
            let mut advect_phi = 0.0;
            for k in 0..D {
                advect_phi += beta[k] * *d_phi[k].get(idx);
            }
            phi_rhs.set(idx, advect_phi - (alpha * trk - div_beta) / 6.0);

            // ---- d_t gtilde_ij ------------------------------------------
            let mut g_rhs = Tensor2::zero();
            for i in 0..D {
                for j in 0..D {
                    let mut v = -2.0 * alpha * at[(i, j)];
                    for k in 0..D {
                        v += beta[k] * *d_gtilde[k][i][j].get(idx);
                        v += gtilde[(i, k)] * *d_beta[j][k].get(idx);
                        v += gtilde[(j, k)] * *d_beta[i][k].get(idx);
                    }
                    v -= (2.0 / 3.0) * gtilde[(i, j)] * div_beta;
                    g_rhs[(i, j)] = v;
                }
            }
            metric_rhs.set(idx, g_rhs);

            // ---- d_t K ----------------------------------------------------
            let mut lap_alpha = 0.0;
            for i in 0..D {
                for j in 0..D {
                    lap_alpha += ginvt[i][j]
                        * (*hess_alpha[i][j].get(idx) + 2.0 * *d_phi[i].get(idx) * *d_alpha[j].get(idx));
                }
            }
            let mut gamma_n_da = 0.0;
            for i in 0..D {
                gamma_n_da += gamma_n[i] * *d_alpha[i].get(idx);
            }
            let mut aa_trace = 0.0;
            for a in 0..D {
                for b in 0..D {
                    aa_trace += at_mixed[a][b] * at_mixed[b][a];
                }
            }
            let mut advect_k = 0.0;
            for k in 0..D {
                advect_k += beta[k] * *d_tracek[k].get(idx);
            }
            trace_k_rhs.set(
                idx,
                -em4 * (lap_alpha - gamma_n_da) + alpha * (aa_trace + trk * trk / 3.0) + advect_k,
            );

            // ---- d_t A tilde_ij ------------------------------------------
            let mut bracket = [[0.0f64; D]; D];
            for i in 0..D {
                for j in 0..D {
                    let mut b = -(*hess_alpha[i][j].get(idx)) + alpha * (rt.data[i][j] + rp.data[i][j]);
                    for k in 0..D {
                        b += syms.data[k][i][j] * *d_alpha[k].get(idx);
                    }
                    b += 2.0 * (*d_alpha[i].get(idx) * *d_phi[j].get(idx)
                        + *d_alpha[j].get(idx) * *d_phi[i].get(idx));
                    bracket[i][j] = b;
                }
            }
            let mut br_trace = 0.0;
            for p in 0..D {
                for q in 0..D {
                    br_trace += ginvt[p][q] * bracket[p][q];
                }
            }
            // A tilde_ik A tilde^k_j
            let mut at_sq = [[0.0f64; D]; D];
            for i in 0..D {
                for j in 0..D {
                    for k in 0..D {
                        at_sq[i][j] += at[(i, k)] * at_mixed[k][j];
                    }
                }
            }
            let mut at_out = Tensor2::zero();
            for i in 0..D {
                for j in 0..D {
                    let tf = bracket[i][j] - (1.0 / 3.0) * gtilde[(i, j)] * br_trace;
                    let mut v = em4 * tf + alpha * (trk * at[(i, j)] - 2.0 * at_sq[i][j]);
                    for k in 0..D {
                        v += at[(i, k)] * *d_beta[j][k].get(idx);
                        v += at[(j, k)] * *d_beta[i][k].get(idx);
                    }
                    v -= (2.0 / 3.0) * at[(i, j)] * div_beta;
                    for k in 0..D {
                        v += beta[k] * *d_at[k][i][j].get(idx);
                    }
                    at_out[(i, j)] = v;
                }
            }
            conformal_a_rhs.set(idx, at_out);

            // ---- d_t Gamma^i ---------------------------------------------
            let mut gamma_rhs = Vector::zero();
            for i in 0..D {
                let mut v = 0.0;
                for j in 0..D {
                    v += -2.0 * at_upper[i][j] * *d_alpha[j].get(idx);
                }
                let mut ch_at = 0.0;
                for j in 0..D {
                    for k in 0..D {
                        ch_at += syms.data[i][j][k] * at_upper[j][k];
                    }
                }
                let mut gamma_dk = 0.0;
                for j in 0..D {
                    gamma_dk += ginvt[i][j] * *d_tracek[j].get(idx);
                }
                let mut at_dphi = 0.0;
                for j in 0..D {
                    at_dphi += at_upper[i][j] * *d_phi[j].get(idx);
                }
                v += 2.0 * alpha * (ch_at - (2.0 / 3.0) * gamma_dk + 6.0 * at_dphi);
                for j in 0..D {
                    for k in 0..D {
                        v += ginvt[j][k] * *d2_beta[i][j][k].get(idx);
                    }
                }
                for j in 0..D {
                    for k in 0..D {
                        v += (1.0 / 3.0) * ginvt[i][j] * *d2_beta[k][j][k].get(idx);
                    }
                }
                for j in 0..D {
                    v += -gamma_n[j] * *d_beta[i][j].get(idx)
                        + (2.0 / 3.0) * gamma_n[i] * *d_beta[j][j].get(idx)
                        + beta[j] * *d_gfn[i][j].get(idx);
                }
                gamma_rhs[i] = v;
            }
            conformal_gamma_rhs.set(idx, gamma_rhs);

            // ---- d_t alpha, 1+log slicing --------------------------------
            let mut advect_alpha = 0.0;
            for k in 0..D {
                advect_alpha += beta[k] * *d_alpha[k].get(idx);
            }
            lapse_rhs.set(idx, advect_alpha - 2.0 * alpha * trk);
        });

        BssnState {
            grid: self.grid.clone(),
            phi: phi_rhs,
            conformal_metric: metric_rhs,
            trace_k: trace_k_rhs,
            conformal_a: conformal_a_rhs,
            conformal_gamma: conformal_gamma_rhs,
            lapse: lapse_rhs,
            shift: Field::new(self.grid.clone(), Vector::zero()),
        }
    }

    /// Time derivative of the shift: frozen, exactly as in ADM.
    pub fn shift_rhs(&self) -> ShiftField<D> {
        Field::new(self.grid.clone(), Vector::zero())
    }

    /// Re-impose the boundary conditions on the evolved state.
    ///
    /// Mirrors `EvolutionState`: after an RK4 step the face values have
    /// drifted, and only the scalar lapse can be held with the current
    /// scalar-per-face `BoundaryValues`. A uniform Dirichlet value over all
    /// components of a tensor or vector field would silently overwrite the
    /// geometry, so those fields are left alone. The geometric BSSN
    /// constraints (det gtilde = 1, trace A tilde = 0) drift only at the
    /// truncation level for the short runs the current tests cover.
    fn enforce_boundaries(
        &mut self,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) {
        apply_nonperiodic(&mut self.lapse, boundary, boundary_values);
    }

    /// self + a * rhs
    pub fn axpy(&self, a: f64, rhs: &Self) -> Self {
        assert_eq!(self.grid.shape, rhs.grid.shape);
        let mut out = self.clone();
        out.add_scaled_in_place(a, rhs);
        out
    }

    /// Advance the full state by one RK4 timestep.
    pub fn step_rk4(
        &self,
        dt: f64,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> Self {
        assert!(dt.is_finite() && dt > 0.0);

        // as in EvolutionState: start from boundary-compliant data, re-impose
        // the boundary conditions after every intermediate stage
        let mut y0 = self.clone();
        y0.enforce_boundaries(boundary, boundary_values);

        let k1 = y0.rhs(boundary, boundary_values);

        let mut y2 = y0.axpy(0.5 * dt, &k1);
        y2.enforce_boundaries(boundary, boundary_values);
        let k2 = y2.rhs(boundary, boundary_values);

        let mut y3 = y0.axpy(0.5 * dt, &k2);
        y3.enforce_boundaries(boundary, boundary_values);
        let k3 = y3.rhs(boundary, boundary_values);

        let mut y4 = y0.axpy(dt, &k3);
        y4.enforce_boundaries(boundary, boundary_values);
        let k4 = y4.rhs(boundary, boundary_values);

        let mut next = y0;
        next.add_scaled_in_place(dt / 6.0, &k1);
        next.add_scaled_in_place(dt / 3.0, &k2);
        next.add_scaled_in_place(dt / 3.0, &k3);
        next.add_scaled_in_place(dt / 6.0, &k4);
        next.enforce_boundaries(boundary, boundary_values);
        next
    }

    fn add_scaled_in_place(&mut self, a: f64, rhs: &Self) {
        self.grid.for_each_index(|idx| {
            self.phi.set(idx, *self.phi.get(idx) + a * *rhs.phi.get(idx));

            let mut g = *self.conformal_metric.get(idx);
            let gr = *rhs.conformal_metric.get(idx);
            for i in 0..D {
                for j in 0..D {
                    g[(i, j)] += a * gr[(i, j)];
                }
            }
            self.conformal_metric.set(idx, g);

            self.trace_k
                .set(idx, *self.trace_k.get(idx) + a * *rhs.trace_k.get(idx));

            let mut at = *self.conformal_a.get(idx);
            let atr = *rhs.conformal_a.get(idx);
            for i in 0..D {
                for j in 0..D {
                    at[(i, j)] += a * atr[(i, j)];
                }
            }
            self.conformal_a.set(idx, at);

            let mut gn = *self.conformal_gamma.get(idx);
            let gnr = *rhs.conformal_gamma.get(idx);
            for i in 0..D {
                gn[i] += a * gnr[i];
            }
            self.conformal_gamma.set(idx, gn);

            self.lapse
                .set(idx, *self.lapse.get(idx) + a * *rhs.lapse.get(idx));

            let mut sh = *self.shift.get(idx);
            let shr = *rhs.shift.get(idx);
            for i in 0..D {
                sh[i] += a * shr[i];
            }
            self.shift.set(idx, sh);
        });
    }
}
