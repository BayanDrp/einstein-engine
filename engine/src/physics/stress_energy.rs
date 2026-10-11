use crate::geometry::tensor::Vector;
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

/// Compute the dust stress-energy tensor at one grid point.
///
/// T_mu_nu = rho * u_mu * u_nu

pub fn dust_stress_energy<const N: usize>(
    rho: f64,
    u: [f64; N],
) -> StressEnergy<N> {
    let mut t = StressEnergy::zero();

    for mu in 0..N {
        for nu in 0..N {
            t.data[mu][nu] = rho * u[mu] * u[nu];
        }
    }

    t
}

/// moving dust on the whole grid: T_mu_nu = rho * u_mu * u_nu from the split
/// covariant four-velocity (u_0, u_i) of `four_velocity_covariant_field.
pub fn dust_stress_energy_field<const N: usize, const D: usize>(
    rho: &Field<f64, D>,
    u_0: &Field<f64, D>,
    u_cov: &Field<Vector<D>, D>,
) -> StressEnergyField<N, D> {
    assert!(N == D + 1, "spacetime dim N must be D + 1, got N = {N}, D = {D}");
    assert_eq!(u_0.grid.shape, rho.grid.shape);
    assert_eq!(u_cov.grid.shape, rho.grid.shape);

    let mut out: StressEnergyField<N, D> = Field::zeros(rho.grid.clone());
    rho.grid.for_each_index(|idx| {
        let mut u = [0.0f64; N];
        u[0] = *u_0.get(idx);
        let sp = *u_cov.get(idx);
        for i in 0..D {
            u[i + 1] = sp[i];
        }
        out.set(idx, dust_stress_energy(*rho.get(idx), u));
    });
    out
}