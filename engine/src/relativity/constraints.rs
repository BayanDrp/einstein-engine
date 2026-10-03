use crate::geometry::christoffel::spatial_christoffel_symbols;
use crate::geometry::covariant_derivative::covariant_divergence_mixed;
use crate::geometry::curvature::spatial_ricci_scalar;
use crate::geometry::metric::{ExtrinsicCurvature, SliceMetric};
use crate::geometry::tensor::{invert, Tensor2, Vector};
use crate::grid::boundary::{BoundaryConfig, BoundaryValues};
use crate::grid::Field;

pub type HamiltonianConstraintField<const D: usize> = Field<f64, D>;

/// Hamiltonian constraint:
///
/// H = ³R + K² - K_ij K^ij - 16πρ
///
/// In an exact Einstein solution:
///
/// H = 0
pub fn hamiltonian_constraint<const D: usize>(
    spatial: &SliceMetric<D>,
    extrinsic: &ExtrinsicCurvature<D>,
    rho: &Field<f64, D>,
    boundary: &BoundaryConfig<D>,
    boundary_values: &BoundaryValues<D>,
) -> HamiltonianConstraintField<D> {
    let spatial_ricci = spatial_ricci_scalar(
        spatial,
        boundary,
        boundary_values,
    );

    let mut out =
        Field::zeros(spatial.grid.clone());

    spatial.grid.for_each_index(|idx| {
        let gamma = spatial.get(idx);
        let k = extrinsic.get(idx);
        let rho = *rho.get(idx);
        let r3 = *spatial_ricci.get(idx);

        // Build γ^{ij}
        let mut gamma_matrix = [[0.0; D]; D];

        for i in 0..D {
            for j in 0..D {
                gamma_matrix[i][j] = gamma[(i, j)];
            }
        }

        let gamma_inv = invert(&gamma_matrix);

        // K = γ^{ij} K_ij
        let mut trace_k = 0.0;

        for i in 0..D {
            for j in 0..D {
                trace_k += gamma_inv[i][j] * k[(i, j)];
            }
        }

        // K_ij K^ij
        let mut k_squared = 0.0;

        for i in 0..D {
            for j in 0..D {
                for p in 0..D {
                    for q in 0..D {
                        k_squared +=
                            k[(i, j)]
                            * gamma_inv[i][p]
                            * gamma_inv[j][q]
                            * k[(p, q)];
                    }
                }
            }
        }

        let h =
            r3
            + trace_k * trace_k
            - k_squared
            - 16.0 * std::f64::consts::PI * rho;

        out.set(idx, h);
    });

    out
}

fn mixed_extrinsic<const D: usize>(
    spatial: &SliceMetric<D>,
    extrinsic: &ExtrinsicCurvature<D>,
) -> Field<Tensor2<D>, D> {
    let mut out: Field<Tensor2<D>, D> = Field::new(spatial.grid.clone(), Tensor2::<D>::zero());

    spatial.grid.for_each_index(|idx| {
        let gamma = spatial.get(idx);
        let k = extrinsic.get(idx);

        let mut gamma_matrix = [[0.0; D]; D];

        for i in 0..D {
            for j in 0..D {
                gamma_matrix[i][j] = gamma[(i, j)];
            }
        }

        let gamma_inv = invert(&gamma_matrix);

        let mut mixed = Tensor2::<D>::zero();

        for upper in 0..D {
            for lower in 0..D {
                for m in 0..D {
                    mixed[(upper, lower)] +=
                        gamma_inv[upper][m] * k[(m, lower)];
                }
            }
        }

        out.set(idx, mixed);
    });

    out
}


fn extrinsic_trace<const D: usize>(
    spatial: &SliceMetric<D>,
    extrinsic: &ExtrinsicCurvature<D>,
) -> Field<f64, D> {
    let mut out = Field::zeros(spatial.grid.clone());

    spatial.grid.for_each_index(|idx| {
        let gamma = spatial.get(idx);
        let k = extrinsic.get(idx);

        let mut gamma_matrix = [[0.0; D]; D];

        for i in 0..D {
            for j in 0..D {
                gamma_matrix[i][j] = gamma[(i, j)];
            }
        }

        let gamma_inv = invert(&gamma_matrix);

        let mut trace = 0.0;

        for i in 0..D {
            for j in 0..D {
                trace += gamma_inv[i][j] * k[(i, j)];
            }
        }

        out.set(idx, trace);
    });

    out
}

/// V^i_j = K^i_j - delta^i_j K: the trace-free part of the mixed
/// extrinsic curvature, whose divergence the momentum constraint sees.
fn tracefree_mixed<const D: usize>(
    spatial: &SliceMetric<D>,
    extrinsic: &ExtrinsicCurvature<D>,
) -> Field<Tensor2<D>, D> {
    let mixed = mixed_extrinsic(spatial, extrinsic);
    let trace = extrinsic_trace(spatial, extrinsic);

    let mut out: Field<Tensor2<D>, D> = Field::new(spatial.grid.clone(), Tensor2::<D>::zero());
    spatial.grid.for_each_index(|idx| {
        let m = mixed.get(idx);
        let k = *trace.get(idx);
        let mut v = Tensor2::<D>::zero();
        for i in 0..D {
            for j in 0..D {
                v[(i, j)] = m[(i, j)] - if i == j { k } else { 0.0 };
            }
        }
        out.set(idx, v);
    });

    out
}

/// momentum constraint:
///
///   M^i = div_j (K^i_j - delta^i_j K) - 8 pi S^i = 0
///
/// S^i is the momentum density; dust at rest has S^i = 0. the covariant
/// divergence lives in geometry/covariant_derivative.rs. In an exact
/// Einstein solution M^i = 0.
pub fn momentum_constraint<const D: usize>(
    spatial: &SliceMetric<D>,
    extrinsic: &ExtrinsicCurvature<D>,
    momentum: &Field<Vector<D>, D>,
    boundary: &BoundaryConfig<D>,
    boundary_values: &BoundaryValues<D>,
) -> Field<Vector<D>, D> {
    // V^i_j = K^i_j - delta^i_j K
    let v = tracefree_mixed(spatial, extrinsic);
    let gamma_c = spatial_christoffel_symbols(spatial, boundary, boundary_values);

    // divergence of the trace-free part, then balance it with matter
    let div = covariant_divergence_mixed(&v, &gamma_c, boundary, boundary_values);

    let mut out: Field<Vector<D>, D> = Field::new(spatial.grid.clone(), Vector::<D>::zero());
    spatial.grid.for_each_index(|idx| {
        let d = div.get(idx);
        let s = momentum.get(idx);
        let mut out_v = Vector::<D>::zero();
        for i in 0..D {
            out_v[i] = d[i] - 8.0 * std::f64::consts::PI * s[i];
        }
        out.set(idx, out_v);
    });

    out
}