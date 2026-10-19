//! Trilinear (tensor-product linear) interpolation of grid fields.
//!
//! Domain: the closed box spanned by grid nodes, per axis
//! `[origin, origin + (shape - 1) * spacing]`. Positions inside are blended
//! from the surrounding cell vertices with the standard weights; positions
//! outside return [`InterpolationError::OutsideDomain`]. Nothing is clamped
//! or wrapped, so callers on periodic grids must wrap positions into the
//! domain first. Degenerate axes (`shape == 1`) sample the single node
//! exactly. NaN coordinates fail the bounds check and are rejected.

use crate::geometry::christoffel::{Christoffel, ChristoffelField};
use crate::geometry::tensor::{Tensor2, Vector};
use crate::grid::{Field, Grid};

/// Interpolation failure: the position lies outside the closed box of grid
/// nodes (this includes NaN coordinates, which fail every bounds check).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationError {
    OutsideDomain,
}

/// Corner indices and weights of the cell containing `pos`.
///
/// Exactly `2^D` entries; weights sum to one. The upper grid boundary maps
/// onto the last cell with weight one on the far vertex, so no index ever
/// leaves the grid.
fn interpolation_corners<const D: usize>(
    grid: &Grid<D>,
    pos: [f64; D],
) -> Result<Vec<([usize; D], f64)>, InterpolationError> {
    let mut base = [0usize; D];
    let mut frac = [0.0f64; D];
    for d in 0..D {
        let lo = grid.origin[d];
        let hi = lo + (grid.shape[d] - 1) as f64 * grid.spacing[d];
        let x = pos[d];
        if !(x >= lo && x <= hi) {
            return Err(InterpolationError::OutsideDomain);
        }
        if grid.shape[d] == 1 {
            base[d] = 0;
            frac[d] = 0.0;
            continue;
        }
        let t = (x - lo) / grid.spacing[d];
        let i0 = t.floor() as usize;
        if i0 >= grid.shape[d] - 1 {
            // exactly on the upper boundary (or its float neighborhood
            // inside the domain): use the last cell, full weight forward
            base[d] = grid.shape[d] - 2;
            frac[d] = 1.0;
        } else {
            base[d] = i0;
            frac[d] = t - i0 as f64;
        }
    }

    let mut corners = Vec::with_capacity(1 << D);
    for mask in 0..(1 << D) {
        let mut idx = [0usize; D];
        let mut w = 1.0;
        for d in 0..D {
            if (mask >> d) & 1 == 1 {
                idx[d] = (base[d] + 1).min(grid.shape[d] - 1);
                w *= frac[d];
            } else {
                idx[d] = base[d];
                w *= 1.0 - frac[d];
            }
        }
        corners.push((idx, w));
    }
    Ok(corners)
}

/// Interpolate a scalar field at an arbitrary position.
pub fn interpolate_scalar<const D: usize>(
    field: &Field<f64, D>,
    pos: [f64; D],
) -> Result<f64, InterpolationError> {
    let mut out = 0.0;
    for (idx, w) in interpolation_corners(&field.grid, pos)? {
        out += w * field.get(idx);
    }
    Ok(out)
}

/// Interpolate a vector field component-wise at an arbitrary position.
pub fn interpolate_vector<const D: usize>(
    field: &Field<Vector<D>, D>,
    pos: [f64; D],
) -> Result<Vector<D>, InterpolationError> {
    let mut out = Vector::<D>::zero();
    for (idx, w) in interpolation_corners(&field.grid, pos)? {
        let v = field.get(idx);
        for i in 0..D {
            out[i] += w * v[i];
        }
    }
    Ok(out)
}

/// Interpolate a rank-2 tensor field component-wise at an arbitrary position.
pub fn interpolate_tensor<const D: usize>(
    field: &Field<Tensor2<D>, D>,
    pos: [f64; D],
) -> Result<Tensor2<D>, InterpolationError> {
    let mut out = Tensor2::<D>::zero();
    for (idx, w) in interpolation_corners(&field.grid, pos)? {
        let t = field.get(idx);
        for i in 0..D {
            for j in 0..D {
                out[(i, j)] += w * t[(i, j)];
            }
        }
    }
    Ok(out)
}

/// Interpolate a full spacetime Christoffel table component-wise.
///
/// Used to sample Gamma^a_bc at off-grid particle positions from a
/// `ChristoffelField` computed on the whole grid. Same domain rules as
/// [`interpolate_scalar`]: closed node box, no wrapping, no extrapolation.
pub fn interpolate_christoffel<const N: usize, const D: usize>(
    field: &ChristoffelField<N, D>,
    pos: [f64; D],
) -> Result<Christoffel<N>, InterpolationError> {
    let mut out = Christoffel::<N>::zero();
    for (idx, w) in interpolation_corners(&field.grid, pos)? {
        let t = field.get(idx);
        for a in 0..N {
            for b in 0..N {
                for c in 0..N {
                    out.data[a][b][c] += w * t.data[a][b][c];
                }
            }
        }
    }
    Ok(out)
}
