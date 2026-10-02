use super::metric::{LapseField, ShiftField, SliceMetric, SpacetimeMetric};
use super::ricci::RicciField;
use super::tensor::invert;
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