use crate::grid::Field;
use super::riemann::RiemannField;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ricci<const N: usize> {
    pub data: [[f64; N]; N],
}

pub type RicciField<const N: usize, const D: usize> =
    Field<Ricci<N>, D>;

impl<const N: usize> Ricci<N> {
    pub fn zero() -> Self {
        Self {
            data: [[0.0; N]; N],
        }
    }
}

impl<const N: usize> Default for Ricci<N> {
    fn default() -> Self {
        Self::zero()
    }
}

pub fn ricci_from_riemann<const N: usize, const D: usize>(
    riemann: &RiemannField<N, D>,
) -> RicciField<N, D> {
    let mut result = Field::zeros(riemann.grid.clone());

    riemann.grid.for_each_index(|idx| {
        let r = riemann.get(idx);
        let mut ricci = Ricci::<N>::zero();

        for nu in 0..N {
            for sigma in 0..N {
                let mut value = 0.0;

                for mu in 0..N {
                    value += r.data[mu][nu][mu][sigma];
                }

                ricci.data[nu][sigma] = value;
            }
        }

        result.set(idx, ricci);
    });

    result
}

