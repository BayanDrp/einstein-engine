use einstein_engine::grid::{Field, Grid};
use einstein_engine::physics::stress_energy::{dust, StressEnergy};
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