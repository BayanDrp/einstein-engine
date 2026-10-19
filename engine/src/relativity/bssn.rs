
use std::f64::consts::PI;
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
use crate::numerics::rk4::AdmState;
use crate::physics::dust::{dust_sources, DustState};
use crate::physics::dust_evolution::{dust_rhs, DustRk4Stages};
use crate::relativity::matter::{FourVelocityError, MatterSources};

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
        adm: &AdmState<D>,
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
    pub fn to_adm(&self) -> AdmState<D> {
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

        AdmState {
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
    ///           + 4 pi alpha (rho + s)
    ///
    ///   d_t A tilde_ij = e^{-4 phi} [ -d_i d_j alpha + Gamma^k_ij d_k alpha
    ///                       + 2 (d_i alpha d_j phi + d_j alpha d_i phi)
    ///                       + alpha (Rtilde_ij + Rphi_ij) ]^TF
    ///           + alpha (K A tilde_ij - 2 A tilde_ik A tilde^k_j)
    ///           + A tilde_ik d_j beta^k + A tilde_jk d_i beta^k
    ///           - (2/3) A tilde_ij d_k beta^k + beta^k d_k A tilde_ij
    ///           - e^{-4 phi} alpha 8 pi (S_ij - (1/3) gamma_ij s)
    ///
    ///   d_t Gamma^i = -2 A tilde^ij d_j alpha
    ///           + 2 alpha (Gamma^i_jk A tilde^jk - (2/3) gtilde^ij d_j K
    ///                       + 6 A tilde^ij d_j phi)
    ///           + gtilde^jk d_j d_k beta^i + (1/3) gtilde^ij d_j d_k beta^k
    ///           - Gamma(n)^j d_j beta^i + (2/3) Gamma(n)^i d_j beta^j
    ///           + beta^j d_j Gamma^i
    ///           - 16 pi alpha gtilde^ij s_j
    ///
    /// Gauge: 1+log slicing for the lapse, frozen shift, exactly as in the
    /// ADM formulation. The "^TF" superscript is the trace-free part with
    /// respect to the conformal metric, and `Gamma(n)^i` is rebuilt from the
    /// current conformal metric while the evolved `Gamma^i` is only ever
    /// differentiated. `rhs` is the vacuum specialization (identically
    /// zero sources).
    /// Vacuum specialization: no matter sources. See `rhs_with_matter`.
    pub fn rhs(
        &self,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> BssnState<D> {
        self.rhs_with_matter(
            &MatterSources::vacuum(&self.grid),
            boundary,
            boundary_values,
        )
    }

    /// Right-hand side of the BSSN equations above, with matter sources.
    pub fn rhs_with_matter(
        &self,
        matter: &MatterSources<D>,
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
            let rho = *matter.rho.get(idx);
            let s_tr = *matter.stress_trace.get(idx);
            let sij = *matter.stress.get(idx);
            let mom = *matter.momentum.get(idx);
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
            // + 4 pi alpha (rho + s): matter sources
            trace_k_rhs.set(
                idx,
                -em4 * (lap_alpha - gamma_n_da)
                    + alpha * (aa_trace + trk * trk / 3.0)
                    + advect_k
                    + alpha * 4.0 * PI * (rho + s_tr),
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
                    // -e^{-4 phi} alpha 8 pi (S_ij - (1/3) gamma_ij s),
                    // gamma_ij = e^{4 phi} gtilde_ij physical
                    v += -em4 * alpha * 8.0 * PI
                        * (sij[(i, j)] - e4 * gtilde[(i, j)] * s_tr / 3.0);
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
            // s_j = gamma_jk S^k, gamma physical
            let mut s_cov = [0.0f64; D];
            for j in 0..D {
                for k in 0..D {
                    s_cov[j] += e4 * gtilde[(j, k)] * mom[k];
                }
            }
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
                // -16 pi alpha gtilde^ij s_j: matter sources
                for j in 0..D {
                    v += -16.0 * alpha * PI * ginvt[i][j] * s_cov[j];
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
    /// Mirrors `AdmState`: after an RK4 step the face values have
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

    /// Re-impose the algebraic BSSN constraints pointwise:
    /// det(gtilde) = 1 and gtilde^ij A tilde_ij = 0.
    /// Standard practice, enforced at every integrator substep, to stop
    /// constraint violations feeding back into the evolution.
    fn enforce_constraints(&mut self) {
        self.grid.for_each_index(|idx| {
            let gtilde = *self.conformal_metric.get(idx);
            let d = det(&matrix(&gtilde));
            let s = d.powf(-1.0 / 3.0);
            let mut g = Tensor2::zero();
            for i in 0..D {
                for j in 0..D {
                    g[(i, j)] = s * gtilde[(i, j)];
                }
            }
            self.conformal_metric.set(idx, g);

            let gi = invert(&matrix(&g));
            let at = *self.conformal_a.get(idx);
            let mut tr = 0.0;
            for i in 0..D {
                for j in 0..D {
                    tr += gi[i][j] * at[(i, j)];
                }
            }
            let mut a = Tensor2::zero();
            for i in 0..D {
                for j in 0..D {
                    a[(i, j)] = at[(i, j)] - g[(i, j)] * tr / 3.0;
                }
            }
            self.conformal_a.set(idx, a);
        });
    }

    /// self + a * rhs
    pub fn axpy(&self, a: f64, rhs: &Self) -> Self {
        assert_eq!(self.grid.shape, rhs.grid.shape);
        let mut out = self.clone();
        out.add_scaled_in_place(a, rhs);
        out
    }

    /// Advance the full state by one RK4 timestep (vacuum).
    pub fn step_rk4(
        &self,
        dt: f64,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> Self {
        self.step_rk4_with_matter(
            dt,
            &MatterSources::vacuum(&self.grid),
            boundary,
            boundary_values,
        )
    }

    /// Advance the full state by one RK4 timestep (vacuum), returning the
    /// stage data instead of the collapsed next state.
    pub fn step_rk4_stages(
        &self,
        dt: f64,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> BssnRk4Stages<D> {
        self.step_rk4_with_matter_stages(
            dt,
            &MatterSources::vacuum(&self.grid),
            boundary,
            boundary_values,
        )
    }

    /// Advance the full state by one RK4 timestep with matter sources,
    /// returning the stage data instead of the collapsed next state.
    ///
    /// The stage sequence is identical to `step_rk4_with_matter`: y0 is
    /// boundary- and constraint-enforced before its rate is taken, every
    /// intermediate stage is re-enforced before it is advanced from, and
    /// the classic RK4 combination is assembled from (y0, k1..k4). This
    /// exists so geodesic couplings can sample the BSSN stage states and
    /// BSSN right-hand sides the integrator really stepped through (see
    /// [`stage_physical_adm`]), at stage times 0, dt/2, dt/2, dt.
    pub fn step_rk4_with_matter_stages(
        &self,
        dt: f64,
        matter: &MatterSources<D>,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> BssnRk4Stages<D> {
        assert!(dt.is_finite() && dt > 0.0);

        // as in AdmState: start from boundary-compliant data, re-impose
        // the boundary conditions after every intermediate stage
        let mut y0 = self.clone();
        y0.enforce_boundaries(boundary, boundary_values);
        y0.enforce_constraints();

        let k1 = y0.rhs_with_matter(matter, boundary, boundary_values);

        let mut y2 = y0.axpy(0.5 * dt, &k1);
        y2.enforce_boundaries(boundary, boundary_values);
        y2.enforce_constraints();
        let k2 = y2.rhs_with_matter(matter, boundary, boundary_values);

        let mut y3 = y0.axpy(0.5 * dt, &k2);
        y3.enforce_boundaries(boundary, boundary_values);
        y3.enforce_constraints();
        let k3 = y3.rhs_with_matter(matter, boundary, boundary_values);

        let mut y4 = y0.axpy(dt, &k3);
        y4.enforce_boundaries(boundary, boundary_values);
        y4.enforce_constraints();
        let k4 = y4.rhs_with_matter(matter, boundary, boundary_values);

        BssnRk4Stages {
            dt,
            y0,
            y2,
            y3,
            y4,
            k1,
            k2,
            k3,
            k4,
        }
    }

    /// Advance the full state by one RK4 timestep with matter sources.
    pub fn step_rk4_with_matter(
        &self,
        dt: f64,
        matter: &MatterSources<D>,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> Self {
        self.step_rk4_with_matter_stages(dt, matter, boundary, boundary_values)
            .collapse(boundary, boundary_values)
    }

    /// Advance geometry and dust together by one RK4 timestep, exposing
    /// the stage data instead of the collapsed states.
    ///
    /// Every stage rebuilds the matter sources from the intermediate dust
    /// state (m_i) and the intermediate geometry (y_i), so neither system
    /// ever sees the other's stale values, exactly as in
    /// [`BssnState::step_rk4_with_dust`]. Dust evolution itself lives in
    /// `physics::dust_evolution`. The return pairs are consistent with
    /// what RK4 evaluated: geometry stage i pairs (y_i, k_g i), dust
    /// stage i pairs (m_i, k_m i), and the classic stepper below collapses
    /// both halves of the same stage set -- so geodesic couplings
    /// (`bssn_stage_connections`) and matter consumers see one shared,
    /// stage-consistent step.
    pub fn step_rk4_with_dust_stages(
        &self,
        dust: &DustState<D>,
        dt: f64,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> Result<(BssnRk4Stages<D>, DustRk4Stages<D>), FourVelocityError> {
        assert!(D == 3, "the BSSN equations are implemented for 3+1 only");
        assert!(dt.is_finite() && dt > 0.0);

        let mut y0 = self.clone();
        y0.enforce_boundaries(boundary, boundary_values);
        y0.enforce_constraints();
        let m0 = dust.clone();

        // k1 = F(Y_n)
        let a0 = y0.to_adm();
        let s1 = dust_sources::<4, D>(&m0, &a0.lapse, &a0.shift, &a0.gamma)?;
        let k1g = y0.rhs_with_matter(&s1, boundary, boundary_values);
        let k1m = dust_rhs(&m0, &a0.lapse, &a0.shift, &a0.gamma, boundary, boundary_values);

        // k2 = F(Y_n + dt/2 * k1)
        let mut y2 = y0.axpy(0.5 * dt, &k1g);
        let m2 = m0.axpy(0.5 * dt, &k1m);
        y2.enforce_boundaries(boundary, boundary_values);
        y2.enforce_constraints();
        let a2 = y2.to_adm();
        let s2 = dust_sources::<4, D>(&m2, &a2.lapse, &a2.shift, &a2.gamma)?;
        let k2g = y2.rhs_with_matter(&s2, boundary, boundary_values);
        let k2m = dust_rhs(&m2, &a2.lapse, &a2.shift, &a2.gamma, boundary, boundary_values);

        // k3 = F(Y_n + dt/2 * k2)
        let mut y3 = y0.axpy(0.5 * dt, &k2g);
        let m3 = m0.axpy(0.5 * dt, &k2m);
        y3.enforce_boundaries(boundary, boundary_values);
        y3.enforce_constraints();
        let a3 = y3.to_adm();
        let s3 = dust_sources::<4, D>(&m3, &a3.lapse, &a3.shift, &a3.gamma)?;
        let k3g = y3.rhs_with_matter(&s3, boundary, boundary_values);
        let k3m = dust_rhs(&m3, &a3.lapse, &a3.shift, &a3.gamma, boundary, boundary_values);

        // k4 = F(Y_n + dt * k3)
        let mut y4 = y0.axpy(dt, &k3g);
        let m4 = m0.axpy(dt, &k3m);
        y4.enforce_boundaries(boundary, boundary_values);
        y4.enforce_constraints();
        let a4 = y4.to_adm();
        let s4 = dust_sources::<4, D>(&m4, &a4.lapse, &a4.shift, &a4.gamma)?;
        let k4g = y4.rhs_with_matter(&s4, boundary, boundary_values);
        let k4m = dust_rhs(&m4, &a4.lapse, &a4.shift, &a4.gamma, boundary, boundary_values);

        Ok((
            BssnRk4Stages {
                dt,
                y0,
                y2,
                y3,
                y4,
                k1: k1g,
                k2: k2g,
                k3: k3g,
                k4: k4g,
            },
            DustRk4Stages {
                dt,
                m0,
                m2,
                m3,
                m4,
                k1: k1m,
                k2: k2m,
                k3: k3m,
                k4: k4m,
            },
        ))
    }

    /// Advance geometry and dust together by one RK4 timestep.
    ///
    /// The collapse of [`BssnState::step_rk4_with_dust_stages`]: geometry
    /// via [`BssnRk4Stages::collapse`] (boundary conditions and algebraic
    /// constraints re-imposed, exactly the sequential accumulation the
    /// classic stepper always did), dust via [`DustRk4Stages::collapse`]
    /// (no enforcement -- conserved densities carry no algebraic
    /// constraints). Bit-for-bit the same numbers as this method always
    /// produced; the only change is that the stages are now reachable.
    pub fn step_rk4_with_dust(
        &self,
        dust: &DustState<D>,
        dt: f64,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> Result<(Self, DustState<D>), FourVelocityError> {
        let (stages, dust_stages) = self.step_rk4_with_dust_stages(
            dust,
            dt,
            boundary,
            boundary_values,
        )?;

        Ok((
            stages.collapse(boundary, boundary_values),
            dust_stages.collapse(),
        ))
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

/// The stage data of one classical RK4 BSSN step: the four stage states
/// y0, y2 = y0 + dt/2 k1, y3 = y0 + dt/2 k2, y4 = y0 + dt k3 and the
/// rates k1..k4 evaluated on them (ki is the right-hand side of yi-1 and
/// of yi for i = 2..4). Boundary conditions and the algebraic BSSN
/// constraints are re-imposed on y0 and on every intermediate stage,
/// exactly as inside `step_rk4_with_matter`.
///
/// This exists so advection-style couplings (e.g. geodesic test particles)
/// can sample the same BSSN geometry the integrator saw at each stage
/// time 0, dt/2, dt/2, dt; see [`stage_physical_adm`] below and
/// `bssn_stage_connections` in `physics::geodesics` for the coupling.
pub struct BssnRk4Stages<const D: usize> {
    pub dt: f64,
    pub y0: BssnState<D>,
    pub y2: BssnState<D>,
    pub y3: BssnState<D>,
    pub y4: BssnState<D>,
    pub k1: BssnState<D>,
    pub k2: BssnState<D>,
    pub k3: BssnState<D>,
    pub k4: BssnState<D>,
}

impl<const D: usize> BssnRk4Stages<D> {
    /// Collapse the stage data into Y_{n+1} = y0 + dt/6 k1 + dt/3 k2 +
    /// dt/3 k3 + dt/6 k4, with the boundary conditions and the algebraic
    /// BSSN constraints re-imposed afterwards. Reproduces exactly the
    /// return value of `step_rk4_with_matter` on the same inputs.
    pub fn collapse(
        &self,
        boundary: &BoundaryConfig<D>,
        boundary_values: &BoundaryValues<D>,
    ) -> BssnState<D> {
        let mut next = self.y0.clone();

        next.add_scaled_in_place(self.dt / 6.0, &self.k1);
        next.add_scaled_in_place(self.dt / 3.0, &self.k2);
        next.add_scaled_in_place(self.dt / 3.0, &self.k3);
        next.add_scaled_in_place(self.dt / 6.0, &self.k4);

        next.enforce_boundaries(boundary, boundary_values);
        next.enforce_constraints();

        next
    }
}

/// The physical ADM geometry of one BSSN stage pair (y, k), plus the
/// temporal metric derivatives the time-aware Christoffel constructor
/// needs, all derived from the engine's own BSSN conformal convention
///
///   gamma_ij = exp(4 phi) gtilde_ij.
///
/// The product rule gives the spatial metric rate
///
///   d_t gamma_ij = exp(4 phi) (4 d_t phi gtilde_ij + d_t gtilde_ij),
///
/// where `d_t phi = k.phi` and `d_t gtilde_ij = k.conformal_metric` are
/// the BSSN right-hand-side fields of the very same stage pair -- never
/// ADM-rate substitutes. The lapse rate is the BSSN gauge field `k.lapse`
/// (1+log slicing over the BSSN trace K) and the shift rate is `k.shift`,
/// which the BSSN gauge freezes to zero exactly like the ADM one. The
/// returned `AdmState` is the honest physical stage state via `to_adm`:
/// gamma_ij = exp(4 phi) gtilde_ij and
/// K_ij = exp(4 phi) A tilde_ij + (1/3) gamma_ij K.
pub fn stage_physical_adm<const D: usize>(
    y: &BssnState<D>,
    k: &BssnState<D>,
) -> (
    AdmState<D>,
    Field<Tensor2<D>, D>,
    Field<Vector<D>, D>,
    Field<f64, D>,
) {
    let adm = y.to_adm();

    let mut dgamma_dt: Field<Tensor2<D>, D> = Field::new(y.grid.clone(), Tensor2::zero());
    y.grid.for_each_index(|idx| {
        let p = *y.phi.get(idx);
        let e4 = (4.0 * p).exp();
        let gtilde = *y.conformal_metric.get(idx);
        let phi_dot = *k.phi.get(idx);
        let gtilde_dot = *k.conformal_metric.get(idx);

        let mut a = Tensor2::<D>::zero();
        for i in 0..D {
            for j in 0..D {
                a[(i, j)] = e4 * (4.0 * phi_dot * gtilde[(i, j)] + gtilde_dot[(i, j)]);
            }
        }
        dgamma_dt.set(idx, a);
    });

    (adm, dgamma_dt, k.shift.clone(), k.lapse.clone())
}
