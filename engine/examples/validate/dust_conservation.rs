//! V6 Dust conservation: the dust update conserves rest mass on a closed
//! periodic domain.
//!
//! Two runs share one smooth density bump:
//!
//!   1. a moving run at uniform `v^x = 0.1`, long enough to translate the
//!      profile across several cells, and
//!   2. a zero velocity run, where the momentum must stay exactly zero.
//!
//! Geometry is held fixed at the flat metric with unit lapse and zero shift.
//! That isolates the dust update that the coupled steppers reuse underneath.
//!
//! Pass criteria and tolerances follow VALIDATION.md, V6.

use einstein_engine::geometry::metric::{LapseField, ShiftField, SliceMetric};
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::physics::dust::DustState;
use einstein_engine::physics::dust_evolution::{dust_rhs, dust_step_rk4};
use std::sync::Arc;

const D: usize = 3;
const H: f64 = 0.5;

const RHO0: f64 = 1.0;
const A: f64 = 0.5;
const SIGMA: f64 = 2.0;

const V0: f64 = 0.1;

/// `v = 0.1` with `dt = 0.25` moves the profile `1.5`, three cells.
const STEPS: usize = 60;
const DT: f64 = 0.25;

const MASS_TOL: f64 = 1.0e-14;
const CONTINUITY_TOL: f64 = 1.0e-10;

pub fn run() -> Result<(), String> {
    let grid = Arc::new(Grid::new([4, 4, 4], [H; D], [0.0; D]));
    let config = BoundaryConfig::<D>::periodic();
    let values = BoundaryValues::<D>::zeros();

    // Flat geometry, unit lapse, zero shift: the dust update is exercised
    // on its own, on a closed domain.
    let mut identity = Tensor2::<D>::zero();
    for i in 0..D {
        identity[(i, i)] = 1.0;
    }

    let spatial = SliceMetric::new(grid.clone(), identity);
    let lapse = LapseField::new(grid.clone(), 1.0);
    let shift = ShiftField::new(grid.clone(), Vector::<D>::zero());

    let density = bump(&grid);

    // Moving run: the conserved density must stay constant even as the
    // profile crosses the periodic boundary.
    let initial = dust_state(&grid, &density, &spatial, V0);
    let initial_mass = dust_total(&initial);

    let mut moving = initial.clone();
    let mut first_step = None;
    for step in 1..=STEPS {
        let next = dust_step_rk4(&moving, DT, &lapse, &shift, &spatial, &config, &values);
        check_density(&next, "moving", step)?;

        if step == 1 {
            first_step = Some(next.clone());
        }

        moving = next;
    }

    let first_step = first_step.ok_or_else(|| "moving run produced no step".to_string())?;
    let final_mass = dust_total(&moving);
    let mass_drift = ((final_mass - initial_mass) / initial_mass).abs();

    if mass_drift > MASS_TOL {
        return Err(format!(
            "moving run: relative rest-mass drift {mass_drift:.3e} exceeds {MASS_TOL:.1e}"
        ));
    }

    // Continuity residual of the first fully discrete step: the RK4 update
    // must equal the stage average of the continuity right-hand side.
    let k1 = dust_rhs(&initial, &lapse, &shift, &spatial, &config, &values);
    let y2 = initial.axpy(0.5 * DT, &k1);
    let k2 = dust_rhs(&y2, &lapse, &shift, &spatial, &config, &values);
    let y3 = initial.axpy(0.5 * DT, &k2);
    let k3 = dust_rhs(&y3, &lapse, &shift, &spatial, &config, &values);
    let y4 = initial.axpy(DT, &k3);
    let k4 = dust_rhs(&y4, &lapse, &shift, &spatial, &config, &values);

    let mut continuity = 0.0f64;
    grid.for_each_index(|idx| {
        let discrete_rate =
            (first_step.densitized_rest_mass.get(idx) - initial.densitized_rest_mass.get(idx)) / DT;
        let stage_average = (k1.densitized_rest_mass.get(idx)
            + 2.0 * k2.densitized_rest_mass.get(idx)
            + 2.0 * k3.densitized_rest_mass.get(idx)
            + k4.densitized_rest_mass.get(idx))
            / 6.0;
        continuity = continuity.max((discrete_rate - stage_average).abs());
    });

    if continuity > CONTINUITY_TOL {
        return Err(format!(
            "moving run: continuity residual {continuity:.3e} exceeds {CONTINUITY_TOL:.1e}"
        ));
    }

    // Zero velocity run: nothing may move, and the momentum must stay
    // exactly zero rather than merely small.
    let rest = dust_state(&grid, &density, &spatial, 0.0);
    let rest_mass = dust_total(&rest);

    let mut stationary = rest;
    for step in 1..=STEPS {
        stationary = dust_step_rk4(&stationary, DT, &lapse, &shift, &spatial, &config, &values);
        check_density(&stationary, "zero velocity", step)?;
        check_zero_momentum(&stationary, step)?;
    }

    if dust_total(&stationary) != rest_mass {
        return Err("zero velocity run: rest mass changed".to_string());
    }

    println!(
        "V6 Dust conservation: mass drift {mass_drift:.3e}, momentum 0.000e0, continuity {continuity:.3e}"
    );

    Ok(())
}

fn bump(grid: &Arc<Grid<D>>) -> Field<f64, D> {
    let mut rho = Field::new(grid.clone(), 0.0);

    grid.for_each_index(|idx| {
        let coords = grid.coords(idx);

        let x = coords[0];
        let y = coords[1];
        let z = coords[2];

        let r_squared = x * x + y * y + z * z;
        let exponent = -r_squared / (2.0 * SIGMA * SIGMA);
        let rho_value = RHO0 * (1.0 + A * exponent.exp());

        rho.set(idx, rho_value);
    });

    rho
}

fn dust_state(
    grid: &Arc<Grid<D>>,
    density: &Field<f64, D>,
    spatial: &SliceMetric<D>,
    velocity: f64,
) -> DustState<D> {
    let velocity = Field::new(grid.clone(), Vector::<D>::new([velocity, 0.0, 0.0]));

    DustState::from_primitives(density, &velocity, spatial)
}

fn dust_total(dust: &DustState<D>) -> f64 {
    let cell_volume: f64 = dust.grid.spacing.iter().product();
    let mut total = 0.0f64;

    dust.grid.for_each_index(|idx| {
        total += *dust.densitized_rest_mass.get(idx);
    });

    total * cell_volume
}

fn check_density(dust: &DustState<D>, label: &str, step: usize) -> Result<(), String> {
    let mut failure = None;

    dust.grid.for_each_index(|idx| {
        let density = *dust.densitized_rest_mass.get(idx);

        if !density.is_finite() || density <= 0.0 {
            failure = Some(*dust.densitized_rest_mass.get(idx));
        }
    });

    match failure {
        Some(density) => Err(format!(
            "{label} run, step {step}: densitized rest mass is {density:.3e}"
        )),
        None => Ok(()),
    }
}

fn check_zero_momentum(dust: &DustState<D>, step: usize) -> Result<(), String> {
    let mut worst = 0.0f64;

    dust.grid.for_each_index(|idx| {
        let momentum = *dust.densitized_momentum.get(idx);
        for i in 0..D {
            worst = worst.max(momentum[i].abs());
        }
    });

    if worst != 0.0 {
        return Err(format!(
            "zero velocity run, step {step}: momentum {worst:.3e} is not exactly zero"
        ));
    }

    Ok(())
}
