use super::christoffel::ChristoffelField;
use super::tensor::{Tensor2, Vector};
use crate::grid::boundary::{BoundaryConfig, BoundaryValues};
use crate::grid::Field;
use crate::numerics::derivatives::central_partial;

// covariant derivatives on the spatial slice: partial derivatives
// plus connection (Christoffel) terms. the connection field has to
// match the slice metric, e.g. from spatial_christoffel_symbols.

/// copy one component of a vector field into a scalar field
fn vector_component<const D: usize>(v: &Field<Vector<D>, D>, m: usize) -> Field<f64, D> {
    let mut out = Field::zeros(v.grid.clone());
    v.grid.for_each_index(|idx| out.set(idx, (*v.get(idx))[m]));
    out
}

/// copy one (i, j) component of a tensor field into a scalar field
fn tensor_component<const D: usize>(
    t: &Field<Tensor2<D>, D>,
    i: usize,
    j: usize,
) -> Field<f64, D> {
    let mut out = Field::zeros(t.grid.clone());
    t.grid.for_each_index(|idx| out.set(idx, t.get(idx)[(i, j)]));
    out
}

/// covariant derivative of a vector field V^i:
///
///   D_k V^i = d_k V^i + Gamma^i_km V^m
///
/// the result holds D_k V^i as data[k][i].
pub fn covariant_deriv_vector<const D: usize>(
    vec: &Field<Vector<D>, D>,
    gamma: &ChristoffelField<D, D>,
    boundary: &BoundaryConfig<D>,
    boundary_values: &BoundaryValues<D>,
) -> Field<Tensor2<D>, D> {
    // d_k V^m for every component, one scalar field per (k, m)
    let mut d_v: Vec<Vec<Field<f64, D>>> = Vec::with_capacity(D);
    for k in 0..D {
        let mut row = Vec::with_capacity(D);
        for m in 0..D {
            let comp = vector_component(vec, m);
            row.push(central_partial(&comp, k, boundary, boundary_values));
        }
        d_v.push(row);
    }

    let mut out: Field<Tensor2<D>, D> = Field::new(vec.grid.clone(), Tensor2::<D>::zero());
    vec.grid.for_each_index(|idx| {
        let g = gamma.get(idx);
        let v = vec.get(idx);
        let mut t = Tensor2::<D>::zero();
        for k in 0..D {
            for i in 0..D {
                let mut s = *d_v[k][i].get(idx);
                for m in 0..D {
                    s += g.data[i][k][m] * v[m];
                }
                t[(k, i)] = s;
            }
        }
        out.set(idx, t);
    });

    out
}

/// covariant derivative of a covector field W_i:
///
///   D_k W_i = d_k W_i - Gamma^m_ki W_m
///
/// the result holds D_k W_i as data[k][i].
pub fn covariant_deriv_covector<const D: usize>(
    covec: &Field<Vector<D>, D>,
    gamma: &ChristoffelField<D, D>,
    boundary: &BoundaryConfig<D>,
    boundary_values: &BoundaryValues<D>,
) -> Field<Tensor2<D>, D> {
    let mut d_w: Vec<Vec<Field<f64, D>>> = Vec::with_capacity(D);
    for k in 0..D {
        let mut row = Vec::with_capacity(D);
        for m in 0..D {
            let comp = vector_component(covec, m);
            row.push(central_partial(&comp, k, boundary, boundary_values));
        }
        d_w.push(row);
    }

    let mut out: Field<Tensor2<D>, D> = Field::new(covec.grid.clone(), Tensor2::<D>::zero());
    covec.grid.for_each_index(|idx| {
        let g = gamma.get(idx);
        let w = covec.get(idx);
        let mut t = Tensor2::<D>::zero();
        for k in 0..D {
            for i in 0..D {
                let mut s = *d_w[k][i].get(idx);
                for m in 0..D {
                    s -= g.data[m][k][i] * w[m];
                }
                t[(k, i)] = s;
            }
        }
        out.set(idx, t);
    });

    out
}

/// covariant divergence of a mixed (1, 1) tensor T^i_j:
///
///   div_j T^i_j = d_j T^i_j + Gamma^i_jm T^m_j - Gamma^m_jj T^i_m
///
/// that is sum over the lower index j, so the result is a vector field.
pub fn covariant_divergence_mixed<const D: usize>(
    mixed: &Field<Tensor2<D>, D>,
    gamma: &ChristoffelField<D, D>,
    boundary: &BoundaryConfig<D>,
    boundary_values: &BoundaryValues<D>,
) -> Field<Vector<D>, D> {
    // d_j T^i_j for every (i, j), one scalar field per component
    let mut d_t: Vec<Vec<Field<f64, D>>> = Vec::with_capacity(D);
    for i in 0..D {
        let mut row = Vec::with_capacity(D);
        for j in 0..D {
            let comp = tensor_component(mixed, i, j);
            row.push(central_partial(&comp, j, boundary, boundary_values));
        }
        d_t.push(row);
    }

    let mut out: Field<Vector<D>, D> = Field::new(mixed.grid.clone(), Vector::<D>::zero());
    mixed.grid.for_each_index(|idx| {
        let g = gamma.get(idx);
        let t = mixed.get(idx);
        let mut v = Vector::<D>::zero();
        for i in 0..D {
            let mut s = 0.0;
            for j in 0..D {
                s += *d_t[i][j].get(idx);
            }
            for j in 0..D {
                for m in 0..D {
                    s += g.data[i][j][m] * t[(m, j)];
                }
            }
            for j in 0..D {
                for m in 0..D {
                    s -= g.data[m][j][j] * t[(i, m)];
                }
            }
            v[i] = s;
        }
        out.set(idx, v);
    });

    out
}