use crate::geometry::christoffel::spatial_christoffel_symbols;
use crate::geometry::metric::{ExtrinsicCurvature, LapseField, ShiftField, SliceMetric};
use crate::geometry::ricci::ricci_from_riemann;
use crate::geometry::riemann::riemann_from_christoffel;
use crate::geometry::tensor::{invert, Tensor2, Vector};
use crate::grid::boundary::{BoundaryConfig, BoundaryValues};
use crate::grid::Field;
use crate::numerics::derivatives::{central_partial, shift_derivative};

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

/// right-hand side of the ADM evolution equation for the spatial metric:
///
///   d_t gamma_ij = -2 alpha K_ij + L_beta gamma_ij
///
/// with the Lie derivative of the metric written out in partials:
///
///   L_beta gamma_ij = beta^k d_k gamma_ij
///                     + gamma_ik d_j beta^k
///                     + gamma_jk d_i beta^k
///
/// This form needs no Christoffel symbols: a coordinate change slides
/// the metric values along beta and tilts the axes, both captured by
/// partial derivatives. Note the result is symmetric in (i, j) because
/// gamma_ij and K_ij are symmetric tensors, so we fill the upper
/// triangle and mirror it.
pub fn spatial_metric_rhs<const D: usize>(
    lapse: &LapseField<D>,
    shift: &ShiftField<D>,
    spatial: &SliceMetric<D>,
    extrinsic: &ExtrinsicCurvature<D>,
    boundary: &BoundaryConfig<D>,
    boundary_values: &BoundaryValues<D>,
) -> Field<Tensor2<D>, D> {
    // d_k gamma_ij: one scalar field per (k, i, j)
    let mut d_gamma: Vec<Vec<Vec<Field<f64, D>>>> = Vec::with_capacity(D);
    for k in 0..D {
        let mut by_i = Vec::with_capacity(D);
        for i in 0..D {
            let mut by_j = Vec::with_capacity(D);
            for j in 0..D {
                let comp = tensor_component(spatial, i, j);
                by_j.push(central_partial(&comp, k, boundary, boundary_values));
            }
            by_i.push(by_j);
        }
        d_gamma.push(by_i);
    }

    // d_i beta^k: one scalar field per (i, k)
    let mut d_beta: Vec<Vec<Field<f64, D>>> = Vec::with_capacity(D);
    for i in 0..D {
        let mut row = Vec::with_capacity(D);
        for k in 0..D {
            let comp = vector_component(shift, k);
            row.push(central_partial(&comp, i, boundary, boundary_values));
        }
        d_beta.push(row);
    }

    let mut out: Field<Tensor2<D>, D> = Field::new(spatial.grid.clone(), Tensor2::<D>::zero());

    spatial.grid.for_each_index(|idx| {
        let alpha = *lapse.get(idx);
        let k_ij = extrinsic.get(idx);
        let beta = shift.get(idx);
        let gamma = spatial.get(idx);

        let mut rhs = Tensor2::<D>::zero();

        for i in 0..D {
            for j in i..D {
                let mut value = -2.0 * alpha * k_ij[(i, j)];

                // beta^k d_k gamma_ij: advection along the shift
                for k in 0..D {
                    value += beta[k] * *d_gamma[k][i][j].get(idx);
                }

                // gamma_ik d_j beta^k + gamma_jk d_i beta^k
                for k in 0..D {
                    value += gamma[(i, k)] * *d_beta[j][k].get(idx);
                    value += gamma[(j, k)] * *d_beta[i][k].get(idx);
                }

                rhs[(i, j)] = value;
                rhs[(j, i)] = value;
            }
        }

        out.set(idx, rhs);
    });

    out
}
/// right-hand side of the ADM evolution equation for the extrinsic
/// curvature, in vacuum:
///
///   d_t K_ij = -alpha nabla_i nabla_j alpha
///              + alpha (R_ij - 2 K_ik K^k_j + K K_ij)
///              + L_beta K_ij
///
/// where the second covariant derivative of the lapse is
///
///   nabla_i nabla_j alpha = d_i d_j alpha - Gamma^k_ij d_k alpha
///
/// and the Lie derivative of the two-tensor K along the shift is
///
///   L_beta K_ij = beta^k d_k K_ij
///                 + K_ik d_j beta^k
///                 + K_jk d_i beta^k
///
/// Unlike the metric equation, this one does need the connection: both
/// nabla nabla alpha and R_ij are built from the spatial Christoffel
/// symbols. As before the result is symmetric in (i, j), so only the
/// upper triangle is filled and then mirrored.
pub fn extrinsic_curvature_rhs<const D: usize>(
    lapse: &LapseField<D>,
    shift: &ShiftField<D>,
    spatial: &SliceMetric<D>,
    extrinsic: &ExtrinsicCurvature<D>,
    boundary: &BoundaryConfig<D>,
    boundary_values: &BoundaryValues<D>,
) -> Field<Tensor2<D>, D> {
    // the connection: needed for nabla nabla alpha and for R_ij
    let gamma_c = spatial_christoffel_symbols(spatial, boundary, boundary_values);

    // R_ij of the spatial slice: Christoffel -> Riemann -> Ricci
    let riemann = riemann_from_christoffel::<D, D>(&gamma_c, boundary, boundary_values);
    let ricci = ricci_from_riemann::<D, D>(&riemann);

    // d_k alpha, one scalar field per direction
    let mut grad_alpha: Vec<Field<f64, D>> = Vec::with_capacity(D);
    for k in 0..D {
        grad_alpha.push(central_partial(lapse, k, boundary, boundary_values));
    }

    // d_i d_j alpha: differentiate each gradient component once more
    let mut hessian: Vec<Vec<Field<f64, D>>> = Vec::with_capacity(D);
    for i in 0..D {
        let mut row = Vec::with_capacity(D);
        for j in 0..D {
            row.push(central_partial(&grad_alpha[j], i, boundary, boundary_values));
        }
        hessian.push(row);
    }

    // d_i beta^k: one scalar field per (i, k)
    let mut d_beta: Vec<Vec<Field<f64, D>>> = Vec::with_capacity(D);
    for i in 0..D {
        let mut row = Vec::with_capacity(D);
        for k in 0..D {
            let comp = vector_component(shift, k);
            row.push(central_partial(&comp, i, boundary, boundary_values));
        }
        d_beta.push(row);
    }

    // beta^k d_k K_ij: advection of each K component along the shift
    let mut d_k_k: Vec<Vec<Field<f64, D>>> = Vec::with_capacity(D);
    for i in 0..D {
        let mut row = Vec::with_capacity(D);
        for j in 0..D {
            let comp = tensor_component(extrinsic, i, j);
            row.push(shift_derivative(&comp, shift, boundary, boundary_values));
        }
        d_k_k.push(row);
    }

    let mut out: Field<Tensor2<D>, D> = Field::new(spatial.grid.clone(), Tensor2::<D>::zero());

    spatial.grid.for_each_index(|idx| {
        let alpha = *lapse.get(idx);
        let k_ij = extrinsic.get(idx);
        let beta = shift.get(idx);
        let gamma = spatial.get(idx);
        let gamma_c = gamma_c.get(idx);
        let r_ij = ricci.get(idx);

        let mut gamma_matrix = [[0.0f64; D]; D];
        for i in 0..D {
            for j in 0..D {
                gamma_matrix[i][j] = gamma[(i, j)];
            }
        }
        let gamma_inv = invert(&gamma_matrix);

        // K = gamma^kl K_kl
        let mut trace_k = 0.0;
        for i in 0..D {
            for j in 0..D {
                trace_k += gamma_inv[i][j] * k_ij[(i, j)];
            }
        }

        // K^k_j = gamma^km K_mj
        let mut k_mixed = Tensor2::<D>::zero();
        for upper in 0..D {
            for lower in 0..D {
                for m in 0..D {
                    k_mixed[(upper, lower)] += gamma_inv[upper][m] * k_ij[(m, lower)];
                }
            }
        }

        // K_ik K^k_j
        let mut k_squared = Tensor2::<D>::zero();
        for i in 0..D {
            for j in 0..D {
                for k in 0..D {
                    k_squared[(i, j)] += k_ij[(i, k)] * k_mixed[(k, j)];
                }
            }
        }

        let mut rhs = Tensor2::<D>::zero();

        for i in 0..D {
            for j in i..D {
                // nabla_i nabla_j alpha = d_i d_j alpha - Gamma^k_ij d_k alpha
                let mut nabla_nabla_alpha = *hessian[i][j].get(idx);
                for k in 0..D {
                    nabla_nabla_alpha -= gamma_c.data[k][i][j] * *grad_alpha[k].get(idx);
                }

                // -alpha nabla_i nabla_j alpha
                let mut value = -alpha * nabla_nabla_alpha;

                // + alpha (R_ij - 2 K_ik K^k_j + K K_ij)
                value += alpha
                    * (r_ij.data[i][j] - 2.0 * k_squared[(i, j)] + trace_k * k_ij[(i, j)]);

                // + beta^k d_k K_ij
                for k in 0..D {
                    value += beta[k] * *d_k_k[i][j].get(idx);
                }

                // + K_ik d_j beta^k + K_jk d_i beta^k
                for k in 0..D {
                    value += k_ij[(i, k)] * *d_beta[j][k].get(idx);
                    value += k_ij[(j, k)] * *d_beta[i][k].get(idx);
                }

                rhs[(i, j)] = value;
                rhs[(j, i)] = value;
            }
        }

        out.set(idx, rhs);
    });

    out
}
