use einstein_engine::grid::{Field, Grid};
use einstein_engine::physics::stress_energy::{dust, StressEnergy , dust_stress_energy};
use std::sync::Arc;

// no matter anywhere: every component of T_mu nu is zero.
#[test]
fn stress_energy_vacuum_zero() {
    let grid = Arc::new(Grid::new([5, 5, 5], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
    let rho: Field<f64, 3> = Field::new(grid.clone(), 0.0);
    let t: einstein_engine::grid::Field<StressEnergy<4>, 3> = dust::<4, 3>(&rho);
    let c = *t.get([2, 2, 2]);
    let expected = StressEnergy::<4>::zero();
    assert_eq!(c.data, expected.data);
}

// dust with density rho: only the tt slot holds rho, rest zero.
#[test]
fn stress_energy_dust_tt_only() {
    let grid = Arc::new(Grid::new([5, 5, 5], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
    let rho: Field<f64, 3> = Field::new(grid.clone(), 2.5);
    let t: einstein_engine::grid::Field<StressEnergy<4>, 3> = dust::<4, 3>(&rho);
    let c = *t.get([2, 2, 2]);
    assert_eq!(c.data[0][0], 2.5); // T_tt = rho
    for a in 0..4 {
        for b in 0..4 {
            if a == 0 && b == 0 {
                continue;
            }
            assert_eq!(c.data[a][b], 0.0);
        }
    }
}

#[test]
fn dust_stress_energy_is_symmetric() {
    let rho = 2.0;
    let u = [-1.25, 0.75, 0.0, 0.0];

    let t = dust_stress_energy::<4>(rho, u);

    for mu in 0..4 {
        for nu in 0..4 {
            assert!(
                (t.data[mu][nu] - t.data[nu][mu]).abs() < 1e-12,
                "T[{mu}][{nu}] != T[{nu}][{mu}]"
            );
        }
    }
}

#[test]
fn dust_stress_energy_scales_linearly_with_density() {
    let u = [-1.25, 0.75, 0.0, 0.0];

    let t1 = dust_stress_energy::<4>(1.0, u);
    let t2 = dust_stress_energy::<4>(2.0, u);

    for mu in 0..4 {
        for nu in 0..4 {
            assert!(
                (t2.data[mu][nu] - 2.0 * t1.data[mu][nu]).abs() < 1e-12
            );
        }
    }
}

#[test]
fn dust_stress_energy_has_zero_components_for_zero_velocity_components() {
    let rho = 2.0;
    let u = [-1.25, 0.75, 0.0, 0.0];

    let t = dust_stress_energy::<4>(rho, u);

    for mu in 0..4 {
        for nu in 2..4 {
            assert_eq!(t.data[mu][nu], 0.0);
            assert_eq!(t.data[nu][mu], 0.0);
        }
    }
}
