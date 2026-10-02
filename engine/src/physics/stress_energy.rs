use crate::grid::Field;

/// T_mu nu: where the energy lives, at every grid point.
/// index 0 is time, indices 1..N are space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StressEnergy<const N: usize> {
    pub data: [[f64; N]; N],
}

pub type StressEnergyField<const N: usize, const D: usize> = Field<StressEnergy<N>, D>;

impl<const N: usize> StressEnergy<N> {
    /// every component zero
    pub fn zero() -> Self {
        Self {
            data: [[0.0; N]; N],
        }
    }
}

impl<const N: usize> Default for StressEnergy<N> {
    fn default() -> Self {
        Self::zero()
    }
}

/// dust at rest: each grid point keeps only its density rho
/// in the tt slot, every other component is zero.
pub fn dust<const N: usize, const D: usize>(rho: &Field<f64, D>) -> StressEnergyField<N, D> {
    let mut out: StressEnergyField<N, D> = Field::zeros(rho.grid.clone());
    rho.grid.for_each_index(|idx| {
        let mut t = StressEnergy::zero();
        t.data[0][0] = *rho.get(idx);
        out.set(idx, t);
    });
    out
}