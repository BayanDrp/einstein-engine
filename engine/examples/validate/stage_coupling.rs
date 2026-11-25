//! V8 Stage coupled particles: stage-aware connections improve a particle
//! trajectory when the geometry changes inside a time step.
//!
//! The scenario uses the prescribed exponential FLRW background
//! `a(t) = A0 exp(H0 t)` from `tests/physics/geodesics_time.rs`. It builds
//! four connection bundles at `t`, `t + dt/2`, `t + dt/2`, `t + dt` and
//! compares the stage-aware particle against a frozen bundle taken at `t`.
//!
//! For this background the momentum
//! `p = A0^2 V0 / sqrt(1 - A0^2 V0^2)` is conserved exactly and
//! `v(t) = p / (a sqrt(a^2 + p^2))` is known in closed form, so both runs
//! are checked against the analytic velocity.
//!
//! Pass criteria and tolerances follow VALIDATION.md, V8.

use einstein_engine::geometry::christoffel::christoffel_symbols_with_time_derivatives;
use einstein_engine::geometry::metric::{LapseField, ShiftField, SliceMetric};
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::physics::geodesics::{GeodesicState, StageConnections};
use einstein_engine::physics::particles::Particles;
use std::sync::Arc;

const D: usize = 3;
const N: usize = 4;

/// Grid spacing.
const H: f64 = 0.5;

/// Hubble rate of the prescribed scale factor, plus its amplitude and the
/// particle's initial velocity.
const H0: f64 = 0.5;
const A0: f64 = 1.0;
const V0: f64 = 0.28;

const DT: f64 = 0.025;
const STEPS: usize = 20;

const VELOCITY_TOL: f64 = 1.0e-5;
const MOMENTUM_TOL: f64 = 1.0e-5;
const CLOCK_TOL: f64 = 1.0e-9;

pub fn run() -> Result<(), String> {
    let grid = Arc::new(Grid::new([16; D], [H; D], [0.0; D]));
    let config = BoundaryConfig::<D>::periodic();
    let values = BoundaryValues::<D>::zeros();

    let lapse = LapseField::new(grid.clone(), 1.0);
    let shift = ShiftField::new(grid.clone(), Vector::<D>::zero());
    let dshift_dt = ShiftField::new(grid.clone(), Vector::<D>::zero());
    let dlapse_dt = LapseField::new(grid.clone(), 0.0);

    // One connection bundle for the prescribed scale factor at time `t`.
    let bundle_at = |t: f64| {
        let a = A0 * (H0 * t).exp();
        let a2 = a * a;

        let mut gamma: SliceMetric<D> = Field::new(grid.clone(), Tensor2::<D>::zero());
        let mut dgamma_dt: SliceMetric<D> = Field::new(grid.clone(), Tensor2::<D>::zero());

        grid.for_each_index(|idx| {
            let mut g = Tensor2::<D>::zero();
            let mut dg = Tensor2::<D>::zero();

            for i in 0..D {
                g[(i, i)] = a2;
                dg[(i, i)] = 2.0 * H0 * a2;
            }

            gamma.set(idx, g);
            dgamma_dt.set(idx, dg);
        });

        christoffel_symbols_with_time_derivatives::<D, N>(
            &lapse, &shift, &gamma, &dgamma_dt, &dshift_dt, &dlapse_dt, &config, &values,
        )
    };

    let start = GeodesicState {
        pos: [0.5, 0.5, 0.5],
        vel: [V0, 0.0, 0.0],
    };

    // Stage-aware run: each RK4 stage samples the geometry at its own time.
    let mut staged = Particles::<D>::new();
    let staged_id = staged
        .spawn(0.0, start)
        .map_err(|error| format!("stage-aware spawn failed: {error:?}"))?;

    let mut t = 0.0;
    for step in 0..STEPS {
        let connections = StageConnections {
            dt: DT,
            connections: [
                bundle_at(t),
                bundle_at(t + 0.5 * DT),
                bundle_at(t + 0.5 * DT),
                bundle_at(t + DT),
            ],
        };

        staged
            .step_all_with_connections(t, &connections)
            .map_err(|error| format!("stage-aware step {step} failed: {error:?}"))?;

        t = staged
            .get(staged_id)
            .ok_or_else(|| "stage-aware particle missing".to_string())?
            .time;
    }

    // Frozen run: all four stages reuse the bundle from the start of the step.
    let mut frozen = Particles::<D>::new();
    let frozen_id = frozen
        .spawn(0.0, start)
        .map_err(|error| format!("frozen spawn failed: {error:?}"))?;

    let mut t = 0.0;
    for step in 0..STEPS {
        let bundle = bundle_at(t);
        let connections = StageConnections {
            dt: DT,
            connections: [bundle.clone(), bundle.clone(), bundle.clone(), bundle],
        };

        frozen
            .step_all_with_connections(t, &connections)
            .map_err(|error| format!("frozen step {step} failed: {error:?}"))?;

        t = frozen
            .get(frozen_id)
            .ok_or_else(|| "frozen particle missing".to_string())?
            .time;
    }

    let staged_particle = staged
        .get(staged_id)
        .ok_or_else(|| "stage-aware particle missing".to_string())?;
    let frozen_particle = frozen
        .get(frozen_id)
        .ok_or_else(|| "frozen particle missing".to_string())?;

    let t_end = STEPS as f64 * DT;
    let a_end = A0 * (H0 * t_end).exp();
    let momentum = A0 * A0 * V0 / (1.0 - A0 * A0 * V0 * V0).sqrt();
    let velocity_exact = momentum / (a_end * (a_end * a_end + momentum * momentum).sqrt());

    let staged_error = (staged_particle.state.vel[0] - velocity_exact).abs();
    let frozen_error = (frozen_particle.state.vel[0] - velocity_exact).abs();

    let staged_velocity = staged_particle.state.vel[0];
    let staged_momentum = a_end * a_end * staged_velocity
        / (1.0 - a_end * a_end * staged_velocity * staged_velocity).sqrt();
    let momentum_drift = (staged_momentum - momentum).abs();
    let clock_error = (staged_particle.time - t_end).abs();

    if staged_error >= VELOCITY_TOL {
        return Err(format!(
            "stage-aware velocity error {staged_error:.3e} is not below {VELOCITY_TOL:.1e}"
        ));
    }

    if momentum_drift >= MOMENTUM_TOL {
        return Err(format!(
            "stage-aware momentum drift {momentum_drift:.3e} is not below {MOMENTUM_TOL:.1e}"
        ));
    }

    if staged_error >= frozen_error {
        return Err(format!(
            "stage-aware error {staged_error:.3e} is not strictly below frozen error {frozen_error:.3e}"
        ));
    }

    if clock_error > CLOCK_TOL {
        return Err(format!(
            "particle clock error {clock_error:.3e} exceeds {CLOCK_TOL:.1e}"
        ));
    }

    println!(
        "V8 Stage coupling: staged v error {staged_error:.3e}, frozen {frozen_error:.3e}, momentum drift {momentum_drift:.3e}, clock {clock_error:.3e}"
    );

    Ok(())
}
