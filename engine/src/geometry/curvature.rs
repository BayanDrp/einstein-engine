use super::christoffel::spatial_christoffel_symbols;
use super::metric::{LapseField, ShiftField, SliceMetric, SpacetimeMetric};
use super::ricci::{ricci_from_riemann, RicciField};
use super::riemann::riemann_from_christoffel;
use super::tensor::invert;
use crate::grid::boundary::{BoundaryConfig, BoundaryValues};
use crate::grid::Field;

/// the Ricci scalar R = g^ab R_ab: one number per grid point.
/// the Ricci tensor measures curvature, and this is the whole
/// curvature collapsed into a single value (like a "how curved
/// is it here" number).
pub type RicciScalarField<const D: usize> = Field<f64, D>;

/// trace the Ricci tensor with the inverse spacetime metric:
/// R = sum over a,b of g^ab R_ab, at every grid point.
pub fn ricci_scalar<const D: usize, const N: usize>(
    lapse: &LapseField<D>,
    shift: &ShiftField<D>,
    spatial: &SliceMetric<D>,
    ricci: &RicciField<N, D>,
) -> RicciScalarField<D> {
    let mut out: RicciScalarField<D> = Field::zeros(lapse.grid.clone());

    lapse.grid.for_each_index(|idx| {
        // g_ab from the 3+1 data, then inverse g^ab
        let g_ab = SpacetimeMetric::from_3plus1(*lapse.get(idx), &shift.get(idx), &spatial.get(idx));
        let g_ab = g_ab.as_matrix::<N>();
        let g_ab_up = invert(&g_ab);

        let r = ricci.get(idx);
        let mut s = 0.0;
        for a in 0..N {
            for b in 0..N {
                s += g_ab_up[a][b] * r.data[a][b];
            }
        }
        out.set(idx, s);
    });

    out
}

/// the Ricci scalar of the spatial slice metric alone: R^(3) = gamma^ij R_ij.
/// runs the full spatial pipeline (Christoffel -> Riemann -> Ricci -> trace)
/// in D dimensions; this is the curvature the Hamiltonian constraint uses.
pub fn spatial_ricci_scalar<const D: usize>(
    spatial: &SliceMetric<D>,
    boundary: &BoundaryConfig<D>,
    boundary_values: &BoundaryValues<D>,
) -> RicciScalarField<D> {
    let gamma_c = spatial_christoffel_symbols::<D>(spatial, boundary, boundary_values);
    let riemann = riemann_from_christoffel::<D, D>(&gamma_c, boundary, boundary_values);
    let ricci = ricci_from_riemann::<D, D>(&riemann);

    let mut out: RicciScalarField<D> = Field::zeros(spatial.grid.clone());
    spatial.grid.for_each_index(|idx| {
        let g = spatial.get(idx);
        let mut gmat = [[0.0f64; D]; D];
        for i in 0..D {
            for j in 0..D {
                gmat[i][j] = g[(i, j)];
            }
        }
        let ginv = invert(&gmat);

        let r = ricci.get(idx);
        let mut s = 0.0;
        for i in 0..D {
            for j in 0..D {
                s += ginv[i][j] * r.data[i][j];
            }
        }
        out.set(idx, s);
    });

    out
}