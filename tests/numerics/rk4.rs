use einstein_engine::geometry::metric::{ExtrinsicCurvature, LapseField, ShiftField, SliceMetric};
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::numerics::rk4::EvolutionState;
use std::sync::Arc;

const D: usize = 3;

const NX: usize = 64;
const H: f64 = 1.0 / NX as f64;

// Wavelength 0.5 on a unit domain, so k*L is a multiple of 2*pi and the
// profile is exactly periodic: the wrap stencil sees the same function as
// the interior, which matters because central differences reach across it.
const K: f64 = 4.0 * std::f64::consts::PI;
const LAMBDA: f64 = 0.5;

// The wave crest starts here and is expected to arrive at X0 + c*T.
const X0: f64 = 0.125;

// A*k must stay far below 1. The K_ij K^ij term in the K equation is
// quadratic in A*k, and it has nothing to balance it: the Hamiltonian
// constraint for this slice is R^(3) + K_ij K^ij - K^2 = 0 with R^(3) = 0
// because the polarization is traceless, so a large A*k means the initial
// data violates that constraint and the wave no longer travels at c.
const AMP: f64 = 0.002;

fn wave_grid() -> Arc<Grid<D>> {
    Arc::new(Grid::new([NX, 4, 4], [H, 0.25, 0.25], [0.0, 0.0, 0.0]))
}

fn small_grid() -> Arc<Grid<D>> {
    Arc::new(Grid::new([8, 4, 4], [0.25, 0.25, 0.25], [0.0, 0.0, 0.0]))
}

fn periodic() -> (BoundaryConfig<D>, BoundaryValues<D>) {
    (BoundaryConfig::<D>::periodic(), BoundaryValues::<D>::zeros())
}

/// Transversely traceless profile: h_yy = g(x) with h_zz = -g(x), so the
/// trace vanishes and R^(3) = 0 on this slice.
fn g(x: f64) -> f64 {
    AMP * (K * (x - X0)).cos()
}

fn g_prime(x: f64) -> f64 {
    -AMP * K * (K * (x - X0)).sin()
}

/// A wave travelling in +x at light speed. The ADM metric equation gives
/// K_ij = -1/2 d_t h_ij at first order, and a right-moving profile has
/// d_t = -d_x, so K_yy = +1/2 g'(x). The opposite sign silently builds a
/// left-moving wave instead.
fn wave_state(grid: Arc<Grid<D>>) -> EvolutionState<D> {
    let mut gamma: SliceMetric<D> = Field::new(grid.clone(), Tensor2::<D>::zero());
    let mut k: ExtrinsicCurvature<D> = Field::new(grid.clone(), Tensor2::<D>::zero());

    grid.for_each_index(|idx| {
        let x = grid.coords(idx)[0];

        let mut gij = Tensor2::<D>::zero();
        for i in 0..D {
            gij[(i, i)] = 1.0;
        }
        gij[(1, 1)] += g(x);
        gij[(2, 2)] -= g(x);
        gamma.set(idx, gij);

        let mut kij = Tensor2::<D>::zero();
        kij[(1, 1)] = 0.5 * g_prime(x);
        kij[(2, 2)] = -0.5 * g_prime(x);
        k.set(idx, kij);
    });

    let lapse: LapseField<D> = Field::new(grid.clone(), 1.0);
    let shift: ShiftField<D> = Field::new(grid.clone(), Vector::<D>::zero());

    EvolutionState {
        grid: grid.clone(),
        gamma,
        k,
        lapse,
        shift,
    }
}

fn advance(state: &EvolutionState<D>, dt: f64, steps: usize) -> EvolutionState<D> {
    let (config, values) = periodic();
    let mut current = state.clone();
    for _ in 0..steps {
        current = current.step_rk4(dt, &config, &values);
    }
    current
}

/// x position of the wave crest, found within a window so the second crest
/// on the periodic domain cannot win, then refined parabolically so the
/// measurement is not quantised to whole cells.
///
/// A parabolic fit through three samples straddling the peak of a cosine
/// locates its vertex far more precisely than one cell, which matters when
/// the expected travel is only a handful of cells.
fn crest_x(state: &EvolutionState<D>, center: f64, window: f64) -> f64 {
    let n = state.grid.shape[0];
    let mut best = 0usize;
    let mut best_value = f64::NEG_INFINITY;

    for j in 0..n {
        let x = state.grid.coords([j, 1, 1])[0];
        if (x - center).abs() > window {
            continue;
        }
        let value = state.gamma.get([j, 1, 1])[(1, 1)];
        if value > best_value {
            best_value = value;
            best = j;
        }
    }

    let before = state.gamma.get([(best + n - 1) % n, 1, 1])[(1, 1)];
    let after = state.gamma.get([(best + 1) % n, 1, 1])[(1, 1)];

    let denominator = before - 2.0 * best_value + after;
    let offset = if denominator.abs() < 1.0e-300 {
        0.0
    } else {
        0.5 * (before - after) / denominator
    };

    state.grid.coords([best, 1, 1])[0] + offset * H
}

/// largest absolute difference of one component between two states
fn component_error(a: &EvolutionState<D>, b: &EvolutionState<D>, i: usize, j: usize) -> f64 {
    let mut worst = 0.0f64;
    a.grid.for_each_index(|idx| {
        worst = worst.max((a.gamma.get(idx)[(i, j)] - b.gamma.get(idx)[(i, j)]).abs());
    });
    worst
}

// Flat spacetime is an exact solution, so every RK4 stage must see zero and
// the state must return bit-identical rather than merely close.
#[test]
fn flat_slice_is_a_fixed_point_over_many_steps() {
    let grid = small_grid();
    let state = EvolutionState::<D>::minkowski(grid.clone());
    let evolved = advance(&state, 0.01, 200);

    assert_eq!(evolved.gamma.data, state.gamma.data, "gamma drifted");
    assert_eq!(evolved.k.data, state.k.data, "K drifted");
}

// 1+log gives ∂t α = -2αK. The wave is traceless with respect to δ^{ij}, so
// the trace only shows up through γ^{ij} ≠ δ^{ij} and is second order in the
// amplitude: the lapse responds to K, but only barely. The shift has no
// equation yet and stays bit for bit.
#[test]
fn gauge_evolves_the_lapse_and_freezes_the_shift() {
    let grid = wave_grid();
    let (config, values) = periodic();
    let state = wave_state(grid.clone());

    let rhs = state.rhs(&config, &values);
    assert!(
        rhs.gamma.data.iter().any(|t| t[(1, 1)].abs() > 1.0e-9),
        "gamma derivative should be nonzero"
    );
    assert!(
        rhs.lapse.data.iter().any(|v| v.abs() > 1.0e-6),
        "1+log should see the trace of K"
    );
    assert!(
        rhs.shift.data.iter().all(|v| v == &Vector::<D>::zero()),
        "shift has no equation yet"
    );

    let stepped = state.step_rk4(0.004, &config, &values);
    assert_eq!(stepped.shift.data, state.shift.data, "shift moved");

    let moved = stepped
        .lapse
        .data
        .iter()
        .zip(&state.lapse.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(moved > 1.0e-9, "lapse should move, moved {moved}");
    assert!(moved < 1.0e-6, "traceless wave should barely move the lapse: {moved}");
}

#[test]
fn axpy_at_zero_scale_leaves_state_identical() {
    let grid = small_grid();
    let (config, values) = periodic();
    let state = EvolutionState::<D>::minkowski(grid.clone());

    let unchanged = state.axpy(0.0, &state.rhs(&config, &values));
    assert_eq!(unchanged.gamma.data, state.gamma.data);
    assert_eq!(unchanged.k.data, state.k.data);
    assert_eq!(unchanged.lapse.data, state.lapse.data);
    assert_eq!(unchanged.shift.data, state.shift.data);
}

// The headline check: data built to travel at c = 1 must actually be
// transported at c = 1. A flipped sign in K, a wrong RK4 weight, or a stale
// term in either equation all show up here as a crest moving at the wrong
// speed or in the wrong direction.
//
// Expect slightly below 1: second-order central differences give a phase
// speed of 1 - (k*h)^2/24, which here is a 0.2% deficit.
#[test]
fn wave_crest_moves_at_speed_of_light() {
    let grid = wave_grid();
    let state = wave_state(grid.clone());

    let start = crest_x(&state, X0, LAMBDA / 2.0);
    assert!((start - X0).abs() < 0.02, "crest should begin near x0, got {start}");

    let dt = 0.5 * H;
    let steps = 8;
    let elapsed = steps as f64 * dt;
    let evolved = advance(&state, dt, steps);

    let center = X0 + 0.5 * elapsed;
    let end = crest_x(&evolved, center, LAMBDA / 2.0);
    let speed = (end - start) / elapsed;

    assert!(
        (speed - 1.0).abs() < 0.05,
        "crest travelled at speed {speed}, expected 1.0 (dt = {dt}, {steps} steps)"
    );
}

// Pin down the integrator's order in time. The grid is held fixed so the
// spatial discretisation error cancels out of the comparison, and both
// errors are measured against a dt/200 reference whose own error is
// ~(1/20)^4 of the finer solution, i.e. about 6e-6 of it.
//
// RK4 is 4th order, so error ~ dt^4 and shrinking dt by a factor of 10 must
// shrink the error by 10^4 = 10000. For contrast: 1st order would give 10,
// 2nd order 100, 3rd order 1000.
#[test]
fn rk4_temporal_error_falls_as_fourth_power() {
    let grid = wave_grid();
    let state = wave_state(grid.clone());

    let dt = 0.5 * H;
    let steps = 4;

    let coarse = advance(&state, dt, steps);
    let fine = advance(&state, dt / 10.0, steps * 10);
    let reference = advance(&state, dt / 200.0, steps * 200);

    let coarse_error = component_error(&coarse, &reference, 1, 1);
    let fine_error = component_error(&fine, &reference, 1, 1);

    assert!(coarse_error > 0.0, "coarse run should differ from the reference");

    let ratio = coarse_error / fine_error;
    assert!(
        (2000.0..50_000.0).contains(&ratio),
        "error ratio was {ratio}, expected near 10000 for a 4th-order scheme"
    );
}

// The step refuses a non-positive or non-finite dt rather than silently
// returning the input state or producing NaNs.
#[test]
#[should_panic]
fn step_rk4_rejects_zero_timestep() {
    let grid = small_grid();
    let (config, values) = periodic();
    EvolutionState::<D>::minkowski(grid).step_rk4(0.0, &config, &values);
}

#[test]
#[should_panic]
fn step_rk4_rejects_negative_timestep() {
    let grid = small_grid();
    let (config, values) = periodic();
    EvolutionState::<D>::minkowski(grid).step_rk4(-0.01, &config, &values);
}
