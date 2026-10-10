use einstein_engine::geometry::metric::{ExtrinsicCurvature, SliceMetric};
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::numerics::rk4::EvolutionState;
use einstein_engine::validation::constraints::{residuals, ConstraintResidual};
use std::sync::Arc;

const D: usize = 3;

const NX: usize = 64;
const H: f64 = 1.0 / NX as f64;
const K: f64 = 4.0 * std::f64::consts::PI;
const X0: f64 = 0.125;
const AMP: f64 = 0.002;

fn wave_grid() -> Arc<Grid<D>> {
    Arc::new(Grid::new([NX, 4, 4], [H, 0.25, 0.25], [0.0, 0.0, 0.0]))
}

fn periodic() -> (BoundaryConfig<D>, BoundaryValues<D>) {
    (BoundaryConfig::<D>::periodic(), BoundaryValues::<D>::zeros())
}

fn zero_rho(grid: &Arc<Grid<D>>) -> Field<f64, D> {
    Field::new(grid.clone(), 0.0)
}

fn zero_momentum(grid: &Arc<Grid<D>>) -> Field<Vector<D>, D> {
    Field::new(grid.clone(), Vector::<D>::zero())
}

/// Same transversely traceless plane wave as tests/numerics/rk4.rs.
fn g(x: f64) -> f64 {
    AMP * (K * (x - X0)).cos()
}

fn g_prime(x: f64) -> f64 {
    -AMP * K * (K * (x - X0)).sin()
}

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

    EvolutionState {
        grid: grid.clone(),
        gamma,
        k,
        lapse: Field::new(grid.clone(), 1.0),
        shift: Field::new(grid.clone(), Vector::<D>::zero()),
    }
}

fn measure(state: &EvolutionState<D>) -> ConstraintResidual {
    let (boundary, boundary_values) = periodic();
    residuals(
        &state.gamma,
        &state.k,
        &zero_rho(&state.grid),
        &zero_momentum(&state.grid),
        &boundary,
        &boundary_values,
    )
}

// Baseline: the monitor itself has to be right before anything it says can
// be trusted. Flat slice, no extrinsic curvature, no matter, so both terms
// are exactly zero rather than merely small.
#[test]
fn minkowski_vacuum_residual_is_zero() {
    let grid = wave_grid();
    let state = EvolutionState::<D>::minkowski(grid.clone());

    let r = measure(&state);
    assert!(
        r.hamiltonian_max < 1.0e-12,
        "hamiltonian residual was {}",
        r.hamiltonian_max
    );
    assert!(
        r.momentum_max < 1.0e-12,
        "momentum residual was {}",
        r.momentum_max
    );
}

// The wave from tests/numerics/rk4.rs is transversely traceless, so its
// trace vanishes and 3R = 0 at linear order. That leaves
//     H = 3R + K^2 - K_ij K^ij - 16 pi rho = -0.5 (g')^2
// which is O((A k)^2): not an exact vacuum solution. The test asserts the
// residual is present at all, and bounded, but reports the value rather
// than pinning it -- what it actually is, is the interesting part.
#[test]
fn wave_initial_data_is_not_exactly_on_shell() {
    let grid = wave_grid();
    let state = wave_state(grid);

    let r = measure(&state);
    println!(
        "wave initial data: |H|max = {:.6e}, |M|max = {:.6e}",
        r.hamiltonian_max, r.momentum_max
    );

    assert!(
        r.hamiltonian_max > 0.0,
        "expected a nonzero hamiltonian residual, got 0: the data is traceless K with 3R = 0"
    );
    assert!(
        r.hamiltonian_max < 1.0e-2,
        "hamiltonian residual {} is far larger than the O((A k)^2) expected here",
        r.hamiltonian_max
    );
    assert!(
        r.momentum_max < 1.0e-2,
        "momentum residual {} should be small for transversely traceless data",
        r.momentum_max
    );
}

// The headline: watch the constraints while the slice evolves. The gauge is
// 1+log, but the wave data is transversely traceless, so its K trace only
// survives through γ^{ij} ≠ δ^{ij} and the lapse stays within ~2e-7 of 1
// over this run. The evolution is therefore indistinguishable from geodesic
// slicing at this horizon, and that is what the numbers below show.
//
// Measured over 8 steps the residual oscillates between 1.218e-3 and 1.243e-3
// and grows by only 0.6%. The envelope rises monotonically while the values
// cycle, so drift is present but far too slow to assert on this horizon.
// What the bound catches is instability, and 2x leaves no doubt about that.
// A longer run is what would make the drift claim itself testable.
#[test]
fn constraint_residual_stays_bounded_over_evolution() {
    let grid = wave_grid();
    let (config, values) = periodic();

    let mut state = wave_state(grid);
    let initial = measure(&state);

    assert!(
        initial.hamiltonian_max > 0.0,
        "nothing to measure: the initial residual is exactly zero, so a growth ratio is meaningless"
    );

    println!(
        "t = 0:0            |H|max = {:.6e} |M|max = {:.6e}",
        initial.hamiltonian_max, initial.momentum_max
    );

    let mut worst = initial.hamiltonian_max;
    for step in 1..=8 {
        state = state.step_rk4(0.5 * H, &config, &values);

        let r = measure(&state);
        worst = worst.max(r.hamiltonian_max);
        println!(
            "t = {:.5}  |H|max = {:.6e} |M|max = {:.6e}",
            step as f64 * 0.5 * H,
            r.hamiltonian_max,
            r.momentum_max
        );

        assert!(r.hamiltonian_max.is_finite(), "hamiltonian residual went non-finite");
        assert!(r.momentum_max.is_finite(), "momentum residual went non-finite");
    }

    let final_h = measure(&state).hamiltonian_max;
    let ratio = worst / initial.hamiltonian_max;
    println!(
        "worst / initial = {:.3}, final / initial = {:.3}",
        ratio,
        final_h / initial.hamiltonian_max
    );

    assert!(
        ratio < 2.0,
        "constraint residual grew by {ratio}x over 8 steps, which is instability rather than drift"
    );
}
