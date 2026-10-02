use crate::grid::Field;
use crate::geometry::curvature::RicciScalarField;
use crate::geometry::metric::{
    LapseField,
    ShiftField,
    SliceMetric,
    SpacetimeMetric,
};
use crate::geometry::ricci::RicciField;



#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EinsteinTensor<const N: usize> {
    pub data: [[f64; N]; N],
}

pub type EinsteinTensorField<const N: usize, const D: usize> =
    Field<EinsteinTensor<N>, D>;

impl<const N: usize> EinsteinTensor<N> {
    pub fn zero() -> Self {
        Self {
            data: [[0.0; N]; N],
        }
    }
}

impl<const N: usize> Default for EinsteinTensor<N> {
    fn default() -> Self {
        Self::zero()
    }
}

pub fn einstein_tensor<const D: usize, const N: usize>(
    lapse: &LapseField<D>,
    shift: &ShiftField<D>,
    spatial: &SliceMetric<D>,
    ricci: &RicciField<N, D>,
    scalar: &RicciScalarField<D>,
) -> EinsteinTensorField<N, D> {
    let mut out: EinsteinTensorField<N, D> =
        Field::zeros(lapse.grid.clone());

    lapse.grid.for_each_index(|idx| {
        let metric =
            SpacetimeMetric::from_3plus1(
                *lapse.get(idx),
                &shift.get(idx),
                &spatial.get(idx),
            );

        let g = metric.as_matrix::<N>();

        let r = ricci.get(idx);
        let scalar_r = *scalar.get(idx);

        let mut einstein = EinsteinTensor::<N>::zero();

        for mu in 0..N {
            for nu in 0..N {
                einstein.data[mu][nu] =
                    r.data[mu][nu]
                    - 0.5 * scalar_r * g[mu][nu];
            }
        }

        out.set(idx, einstein);
    });

    out
}