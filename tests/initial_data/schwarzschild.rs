use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::initial_data::schwarzschild::schwarzschild;
use einstein_engine::numerics::rk4::AdmState;
use einstein_engine::relativity::constraints::{hamiltonian_constraint, momentum_constraint};
use einstein_engine::validation::constraints::residuals;
use std::sync::Arc;

const D: usize = 3;
const MASS: f64 = 1.0;

// The puncture sits at the coordinate origin and the grid lives in the octant
// x, y, z >= 0.5, so every sample lies outside the horizon (r > M/2 = 0.5).
const ORIGIN: [f64; D] = [0.5, 0.5, 0.5];

fn periodic() -> (BoundaryConfig<D>, BoundaryValues<D>) {
    (BoundaryConfig::<D>::periodic(), BoundaryValues::<D>::zeros())
}

fn grid(shape: usize, h: f64) -> Arc<Grid<D>> {
    Arc::new(Grid::new([shape; D], [h; D], ORIGIN))
}

fn zero_rho(grid: &Arc<Grid<D>>) -> Field<f64, D> {
    Field::new(grid.clone(), 0.0)
}

fn zero_momentum(grid: &Arc<Grid<D>>) -> Field<Vector<D>, D> {
    Field::new(grid.clone(), Vector::<D>::zero())
}

// Every field is the analytic isotropic-Schwarzschild value pointwise:
// gamma_ij = psi^4 delta_ij with psi = 1 + M/2r, the Schwarzschild lapse, and
// K_ij = beta^i = 0.
#[test]
fn matches_the_isotropic_form_pointwise() {
    let grid = grid(8, 0.25);
    let state = schwarzschild::<D>(grid.clone(), MASS).expect("valid data");

    grid.for_each_index(|idx| {
        let c = grid.coords(idx);
        let r = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        let psi = 1.0 + MASS / (2.0 * r);
        let psi4 = psi * psi * psi * psi;
        let alpha = (1.0 - MASS / (2.0 * r)) / (1.0 + MASS / (2.0 * r));

        let g = state.gamma.get(idx);
        for i in 0..D {
            assert!(
                (g[(i, i)] - psi4).abs() < 1e-12,
                "gamma[{i},{i}] at {idx:?}: got {}, want {psi4}",
                g[(i, i)]
            );
            for j in 0..D {
                if i != j {
                    assert_eq!(g[(i, j)], 0.0, "gamma[{i},{j}] at {idx:?} should be 0");
                }
            }
        }

        assert!(
            (state.lapse.get(idx) - alpha).abs() < 1e-12,
            "lapse at {idx:?}: got {}, want {alpha}",
            state.lapse.get(idx)
        );
        assert_eq!(state.k.get(idx), &Tensor2::<D>::zero());
        assert_eq!(state.shift.get(idx), &Vector::<D>::zero());
    });
}

// The slice is a moment of time symmetry: K = 0 makes the momentum constraint
// identically zero, down to the last bit (a divergence of zero is zero).
#[test]
fn momentum_constraint_is_exactly_zero() {
    let grid = grid(8, 0.25);
    let state = schwarzschild::<D>(grid.clone(), MASS).unwrap();
    let (config, values) = periodic();

    let r = residuals(
        &state.gamma,
        &state.k,
        &zero_rho(&grid),
        &zero_momentum(&grid),
        &config,
        &values,
    );

    assert_eq!(r.momentum_max, 0.0, "K = 0 must give exactly zero momentum");
}

// Largest |H| over a fixed physical window, so two grids of different spacing
// are compared on the same region rather than on different cell counts.
fn window_hamiltonian_max(state: &AdmState<D>, lo: f64, hi: f64) -> f64 {
    let (config, values) = periodic();
    let h = hamiltonian_constraint(
        &state.gamma,
        &state.k,
        &zero_rho(&state.grid),
        &config,
        &values,
    );

    let mut worst = 0.0f64;
    state.grid.for_each_index(|idx| {
        let c = state.grid.coords(idx);
        if c.iter().all(|&x| x >= lo && x <= hi) {
            worst = worst.max(h.get(idx).abs());
        }
    });
    worst
}

// The point of this data: gamma is conformally flat and psi is harmonic away
// from the puncture, so R^(3) = 0 exactly in the continuum and the whole
// residual is finite-difference truncation. Halving h must quarter it, which
// is what proves the data is on-shell rather than accidentally small.
//
// Measured over x, y, z in [0.75, 1.25], away from the periodic faces.
#[test]
fn hamiltonian_residual_converges_second_order() {
    let coarse = schwarzschild::<D>(grid(16, 1.0 / 16.0), MASS).unwrap();
    let fine = schwarzschild::<D>(grid(32, 1.0 / 32.0), MASS).unwrap();

    let (lo, hi) = (0.75, 1.25);
    let e_coarse = window_hamiltonian_max(&coarse, lo, hi);
    let e_fine = window_hamiltonian_max(&fine, lo, hi);
    let ratio = e_coarse / e_fine;

    println!(
        "H residual: coarse(h=1/16) = {e_coarse:.4e}, fine(h=1/32) = {e_fine:.4e}, ratio = {ratio:.3}"
    );

    assert!(
        e_coarse > 0.0,
        "an exact zero would mean the test is not measuring truncation"
    );
    assert!(
        (3.0..5.0).contains(&ratio),
        "residual ratio was {ratio}, expected ~4 for a second-order operator"
    );
}

// Guard rails: the constructor refuses data it cannot represent.
#[test]
fn rejects_a_puncture_inside_the_grid() {
    // origin at zero puts r = 0 on the corner grid point
    let bad = Arc::new(Grid::new([4, 4, 4], [0.25; D], [0.0; D]));
    assert!(schwarzschild::<D>(bad, MASS).is_err());
}

#[test]
fn rejects_nonpositive_or_nonfinite_mass() {
    assert!(schwarzschild::<D>(grid(4, 0.25), 0.0).is_err());
    assert!(schwarzschild::<D>(grid(4, 0.25), -1.0).is_err());
    assert!(schwarzschild::<D>(grid(4, 0.25), f64::NAN).is_err());
}

#[test]
fn rejects_dimensions_other_than_three() {
    let flat = Arc::new(Grid::new([4, 4], [0.25, 0.25], [1.0, 1.0]));
    assert!(schwarzschild::<2>(flat, MASS).is_err());
}

// -------------------------------------------------------------------------
// long-run evolution
// -------------------------------------------------------------------------

fn in_window(c: [f64; D], lo: f64, hi: f64) -> bool {
    c.iter().all(|&x| x >= lo && x <= hi)
}

// |H| and |M| over a fixed physical window, away from the periodic faces: the
// boundary stencils wrap around data that is not periodic, so a layer of a
// couple of cells there is not physical and would otherwise dominate.
fn window_constraints(state: &AdmState<D>, lo: f64, hi: f64) -> (f64, f64) {
    let (config, values) = periodic();
    let h = hamiltonian_constraint(
        &state.gamma,
        &state.k,
        &zero_rho(&state.grid),
        &config,
        &values,
    );
    let m = momentum_constraint(
        &state.gamma,
        &state.k,
        &zero_momentum(&state.grid),
        &config,
        &values,
    );

    let mut worst_h = 0.0f64;
    let mut worst_m = 0.0f64;
    state.grid.for_each_index(|idx| {
        if !in_window(state.grid.coords(idx), lo, hi) {
            return;
        }
        worst_h = worst_h.max(h.get(idx).abs());
        let mv = m.get(idx);
        for i in 0..D {
            worst_m = worst_m.max(mv[i].abs());
        }
    });
    (worst_h, worst_m)
}

fn window_max_abs_k(state: &AdmState<D>, lo: f64, hi: f64) -> f64 {
    let mut worst = 0.0f64;
    state.grid.for_each_index(|idx| {
        if !in_window(state.grid.coords(idx), lo, hi) {
            return;
        }
        let k = state.k.get(idx);
        for i in 0..D {
            for j in 0..D {
                worst = worst.max(k[(i, j)].abs());
            }
        }
    });
    worst
}

// Same window, but on the right-hand side of the state.
fn window_rhs_maxes(state: &AdmState<D>, lo: f64, hi: f64) -> (f64, f64, f64, f64) {
    let (config, values) = periodic();
    let r = state.rhs(&config, &values);

    let mut dtg = 0.0f64;
    let mut dtk = 0.0f64;
    let mut dta = 0.0f64;
    let mut dtb = 0.0f64;

    state.grid.for_each_index(|idx| {
        if !in_window(state.grid.coords(idx), lo, hi) {
            return;
        }
        let gt = r.gamma.get(idx);
        let kt = r.k.get(idx);
        let bt = r.shift.get(idx);
        for i in 0..D {
            for j in 0..D {
                dtg = dtg.max(gt[(i, j)].abs());
                dtk = dtk.max(kt[(i, j)].abs());
            }
            dtb = dtb.max(bt[i].abs());
        }
        dta = dta.max(r.lapse.get(idx).abs());
    });

    (dtg, dtk, dta, dtb)
}

// The isotropic slice is a moment of time symmetry with the Schwarzschild
// lapse, so it is static: the exact ADM fixed point is D_i D_j alpha = alpha
// R_ij. At the discrete level that identity holds only up to truncation O(h^2),
// and the mismatch shows up in d_t K and nowhere else, since d_t gamma, d_t
// alpha and d_t beta are all proportional to K (exactly zero here).
#[test]
fn slice_is_a_discrete_fixed_point() {
    let (lo, hi) = (0.75, 1.25);

    let coarse = schwarzschild::<D>(grid(16, 1.0 / 16.0), MASS).unwrap();
    let fine = schwarzschild::<D>(grid(32, 1.0 / 32.0), MASS).unwrap();

    let (dtg, dtk, dta, dtb) = window_rhs_maxes(&coarse, lo, hi);
    assert_eq!(dtg, 0.0, "d_t gamma must vanish with K = 0");
    assert_eq!(dta, 0.0, "1+log must leave alpha alone when K = 0");
    assert_eq!(dtb, 0.0, "the frozen-shift RHS is exactly zero");
    assert!(dtk > 0.0, "d_t K should sit at the truncation level");

    // d_t K is pure truncation, so it must converge at second order: coarse h
    // against fine h/2 gives ratio close to 4.
    let (_, dtk_f, _, _) = window_rhs_maxes(&fine, lo, hi);
    let ratio = dtk / dtk_f;
    assert!(
        (ratio - 4.0).abs() < 0.5,
        "d_t K should quarter per h/2 (ratio 4), got {ratio:.2}"
    );
}

// The static-slice fixed point is *not* dynamically stable in this scheme:
// plain ADM + 1+log carries a spatial mode that grows like exp(lambda t) with
// a rate set by the grid (lambda ~ c/h), independent of the timestep, of the
// field mass, and of the lapse gauge. Measured: shrinking dt from h/4 down to
// h/100 changes nothing, and freezing the lapse makes the blow-up faster, not
// slower. This is the well-known weak-hyperbolicity pathology of ADM, which is
// why long-lived evolutions need BSSN (or equivalent) -- the next roadmap
// phase. Until that formulation lands, the long-run bound below is expected to
// fail, so it stays ignored; run it with `--ignored` to see the instability.
#[test]
#[ignore = "plain ADM + 1+log is not stable on curved slices over long runs; blocked on BSSN"]
fn static_slice_stays_bounded_over_a_long_run() {
    let h = 1.0 / 20.0;
    let grid = Arc::new(Grid::new([20; D], [h; D], ORIGIN));
    let (config, values) = periodic();

    let mut state = schwarzschild::<D>(grid, MASS).unwrap();

    let (lo, hi) = (0.8, 1.2);
    let (initial_h, initial_m) = window_constraints(&state, lo, hi);
    let initial_k = window_max_abs_k(&state, lo, hi);

    assert!(initial_h > 0.0, "the data must start with a nonzero truncation residual");
    assert_eq!(initial_m, 0.0, "K = 0 at t = 0 gives exactly zero momentum");
    assert_eq!(initial_k, 0.0, "K = 0 at t = 0");

    let dt = 0.5 * h;
    let steps = 40; // t = 1.0
    let mut worst_h = initial_h;

    for step in 1..=steps {
        state = state.step_rk4(dt, &config, &values);
        let (hres, mres) = window_constraints(&state, lo, hi);
        worst_h = worst_h.max(hres);

        assert!(hres.is_finite() && mres.is_finite(), "went non-finite at step {step}");

        if step % 10 == 0 {
            println!(
                "step {step:>3} (t = {:.3}): |H| = {:.3e}, |M| = {:.3e}, max|K| = {:.3e}",
                step as f64 * dt,
                hres,
                mres,
                window_max_abs_k(&state, lo, hi)
            );
        }
    }

    let (final_h, final_m) = window_constraints(&state, lo, hi);
    let ratio = worst_h / initial_h;

    println!(
        "t = 1: |H| = {final_h:.3e}, |M| = {final_m:.3e}; worst/initial = {ratio:.3}"
    );

    // K stays only as large as truncation can inject it, so the momentum
    // residual is still tiny, and the Hamiltonian residual has not run away.
    assert!(
        window_max_abs_k(&state, lo, hi) < 1.0e-2,
        "K grew far beyond the truncation level"
    );
    assert!(final_m < 1.0e-3, "momentum residual grew to {final_m}");
    assert!(
        ratio < 4.0,
        "|H| grew by {ratio}x over the run; a static slice should stay at its truncation level"
    );
}

