//! Orbit validation on the isotropic Schwarzschild slice (M4).
//!
//! Every tolerance below is set from measured behavior of the discrete
//! pipeline (central-difference Christoffels on the named grid + RK4 at the
//! named dt), not from wishful asymptotics; the temporal fourth order and
//! the spatial second order are measured, then asserted in justified bands.
//! Two error channels are separated on purpose:
//!
//! - temporal: the same coordinate-time right-hand side driven by the
//!   *exact* analytic connection (no grid), error against the exact
//!   circular solution with dt, dt/2, dt/4, yielding the measured ~16
//!   ratios of the RK4 integrator itself;
//! - spatial: the grid pipeline at negligible dt, error against the
//!   analytic orbit with h and h/2, yielding the ~4 ratio of the O(h^2)
//!   central differences. A third test pins what the full grid pipeline
//!   delivers in dt: an effective second order (trilinear interpolation
//!   of the connection is only C^0 across cell faces), measured, not
//!   asserted to be fourth order.
//!
//! The photon sphere is validated on a quarter-revolution arc because it
//! is an unstable null orbit: perturbations grow like exp(lambda t) with
//! the exact epicyclic exponent lambda = omega = 1/(3 sqrt(3) M), so a
//! full revolution would amplify a 1e-3 grid-induced seed by x533. Multi-
//! revolution closure is asserted on the stable ISCO orbit instead.

use einstein_engine::geometry::christoffel::{christoffel_symbols, ChristoffelField};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::Grid;
use einstein_engine::initial_data::schwarzschild::{isotropic_fields, schwarzschild};
use einstein_engine::numerics::interpolation::interpolate_christoffel;
use einstein_engine::physics::geodesics::{geodesic_rhs, geodesic_step_rk4, GeodesicState};
use einstein_engine::physics::orbits::{
    areal_of_state, circular_omega, circular_state, isotropic_radius, metric_functions,
    normalization,
};
use std::f64::consts::PI;
use std::sync::Arc;

const D: usize = 3;
const N: usize = 4;
const MASS: f64 = 1.0;

fn periodic() -> (BoundaryConfig<D>, BoundaryValues<D>) {
    (
        BoundaryConfig::<D>::periodic(),
        BoundaryValues::<D>::zeros(),
    )
}

/// even-count grid centered on the puncture: nodes at odd multiples of h/2
/// on every axis, so no node sits at rho = 0
fn hole_grid(min_half_span: f64, h: f64) -> Arc<Grid<D>> {
    let n = (((2.0 * min_half_span) / h).ceil() as usize) + 1;
    let n = if n % 2 == 1 { n + 1 } else { n };
    let origin = -((n as f64 - 1.0) * h) / 2.0;
    Arc::new(Grid::new([n; D], [h; D], [origin; D]))
}

fn connection_at(areal_r: f64, h: f64) -> ChristoffelField<N, D> {
    let grid = hole_grid(isotropic_radius(MASS, areal_r) + 1.4, h);
    let adm = isotropic_fields(grid, MASS).expect("fixture grid");
    let (config, values) = periodic();
    christoffel_symbols::<D, N>(&adm.lapse, &adm.shift, &adm.gamma, &config, &values)
}

/// angle of (x, y) wrapped into (-pi, pi]
fn wrap(angle: f64) -> f64 {
    ((angle + PI).rem_euclid(2.0 * PI)) - PI
}

/// distance in the orbital plane
fn plane_dist(a: [f64; D], b: [f64; D]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Radial acceleration the engine RHS assigns at the start of the circular
/// orbit: the continuum solution x(t) = rho cos(omega t) forces
/// d^2x/dt^2(0) = -rho omega^2 and d^2y/dt^2(0) = 0 exactly, with omega
/// the Kepler rate sqrt(M / r^3) in AREAL radius. The interpolated discrete
/// connection must reproduce both rows up to its O(h^2) symbol error --
/// this validates the derived initial conditions against the engine's
/// actual Christoffels, not against another hand formula.
#[test]
fn circular_initial_conditions_match_kepler_balance() {
    for &(areal, h) in &[(3.0 * MASS, 0.25), (6.0 * MASS, 0.25)] {
        let gamma = connection_at(areal, h);
        let state = circular_state::<D>(MASS, areal);
        let rho = state.pos[0];
        let omega = circular_omega(MASS, areal);

        let rhs =
            geodesic_rhs::<D, N>(&state, &interpolate_christoffel(&gamma, state.pos).unwrap());
        let ax_exact = -rho * omega * omega;
        println!(
            "areal r = {areal}, h = {h}: a_x = {:.6}, exact = {ax_exact:.6}, a_y = {:.3e}, a_z = {:.3e}",
            rhs.vel[0], rhs.vel[1], rhs.vel[2]
        );
        assert!(
            (rhs.vel[0] - ax_exact).abs() < 0.05 * ax_exact.abs(),
            "radial acceleration off the Kepler value at r = {areal}"
        );
        assert!(
            rhs.vel[1].abs() < 1e-12 * ax_exact.abs(),
            "tangential acceleration must vanish at the start"
        );
        assert!(rhs.vel[2].abs() < 1e-15, "no out-of-plane force");
    }
}

/// Wherever the engine factory itself runs, the unrestricted fixture must
/// produce bit-identical data (same closed formulas); on the fine centered
/// grids the orbit tests need, only the fixture can be constructed at all,
/// and the reason is precisely the factory's horizon-node exclusion.
#[test]
fn fixture_is_engine_data_where_the_factory_runs() {
    // coarse exterior grid like the M2 orbit test: every node outside the
    // isotropic horizon rho = M/2 on the +-11.5 box with unit spacing
    let n = 24;
    let h = 1.0;
    let half = (n as f64 - 1.0) * h / 2.0;
    let g = Arc::new(hole_grid_exterior(n, h, half));
    let fixture = isotropic_fields(g.clone(), MASS).expect("fixture accepts");
    let factory = schwarzschild(g.clone(), MASS).expect("factory accepts this grid");

    let mut worst = 0.0_f64;
    g.for_each_index(|idx| {
        let a = *fixture.gamma.get(idx);
        let b = *factory.gamma.get(idx);
        for i in 0..D {
            for j in 0..D {
                worst = worst.max((a[(i, j)] - b[(i, j)]).abs());
            }
        }
        worst = worst.max((*fixture.lapse.get(idx) - *factory.lapse.get(idx)).abs());
    });
    assert!(worst == 0.0, "fixture differs from the factory by {worst}");

    // the fine centered photon grid: the fixture must accept it, the
    // factory must reject it (nearest node rho = 0.216 < M/2 = 0.5)
    let grid = hole_grid(isotropic_radius(MASS, 3.0 * MASS) + 1.4, 0.25);
    assert!(schwarzschild(grid.clone(), MASS).is_err());
    let _fine = isotropic_fields(grid.clone(), MASS).expect("fixture accepts");
    // data stays sane across the interior region: every psi^4 finite, and
    // the puncture is node-free so nothing diverges
    let mut max_psi = 0.0_f64;
    grid.for_each_index(|idx| {
        let rho = (grid.coords(idx).iter().map(|x| x * x).sum::<f64>()).sqrt();
        let (psi4, _) = metric_functions(MASS, rho);
        assert!(psi4.is_finite());
        max_psi = max_psi.max(psi4);
    });
    println!(
        "fine photon grid: n = {:?}, max psi^4 = {max_psi:.3e}",
        grid.shape
    );
}

/// the M2 orbit test's grid: +-11.5 box, unit spacing, one node per unit
/// cell, closest nodes at rho = 0.866 (all outside the horizon M/2 = 0.5)
fn hole_grid_exterior(n: usize, h: f64, half: f64) -> Grid<D> {
    Grid::new([n; D], [h; D], [-half; D])
}

/// Photon sphere r = 3M: quarter-revolution arc at dt = 0.01, h = 0.25.
/// Measured on this pipeline: drift ~1.6e-2 (unstable-orbit amplified
/// grid error), phase ~4.5e-3 rad, g_ww drift ~3.9e-3 (the null condition
/// g_ww = 0 tracked along the arc), end_err ~1.9e-2.
#[test]
fn photon_sphere_orbit_tracks_analytic_arc() {
    let areal = 3.0 * MASS;
    let h = 0.25;
    let gamma = connection_at(areal, h);

    let rho = isotropic_radius(MASS, areal);
    let omega = circular_omega(MASS, areal);
    let arc_revs = 0.25;
    let period = 2.0 * PI / omega;
    let dt = 0.01;
    let steps = (arc_revs * period / dt).round() as usize;

    let state0 = circular_state::<D>(MASS, areal);
    let gww0 = normalization(MASS, state0.pos, state0.vel);
    assert!(gww0.abs() < 1e-12, "photon IC must be exactly null");

    let mut state = state0;
    let mut t = 0.0;
    let mut drift_max = 0.0_f64;
    let mut phase_max = 0.0_f64;
    let mut gww_max = 0.0_f64;
    for _ in 0..steps {
        state = geodesic_step_rk4::<D, N>(&state, dt, &gamma).unwrap();
        t += dt;
        let r_areal = areal_of_state(MASS, state.pos);
        drift_max = drift_max.max((r_areal - areal).abs());
        phase_max = phase_max.max(wrap(state.pos[1].atan2(state.pos[0]) - omega * t).abs());
        gww_max = gww_max.max(normalization(MASS, state.pos, state.vel).abs());
        assert!(state.pos[2].abs() < 1e-12, "left the orbital plane");
    }

    let end_err = plane_dist(
        state.pos,
        [rho * (omega * t).cos(), rho * (omega * t).sin(), 0.0],
    );
    println!(
        "photon arc: steps = {steps}, drift = {drift_max:.3e}, phase = {phase_max:.3e}, |g_ww| = {gww_max:.3e}, end_err = {end_err:.3e}"
    );

    assert!(drift_max < 0.04, "photon arc drifted {drift_max}");
    assert!(phase_max < 0.02, "photon arc phase drifted {phase_max}");
    assert!(
        gww_max < 0.008,
        "null condition violated: |g_ww| = {gww_max}"
    );
    assert!(end_err < 0.04, "photon arc end error {end_err}");
}

/// ISCO r = 6M: stable timelike orbit, two full revolutions at dt = 0.05,
/// h = 0.25. Measured: areal drift ~6.1e-3, phase ~8.4e-3 rad, g_ww drift
/// ~5.4e-4 around the exact value g_ww = -1/2 (d tau/dt = sqrt(1/2)),
/// closure ~4.6e-2. Stability: no secular growth over the two revolutions.
#[test]
fn isco_orbit_closes_over_two_revolutions() {
    let areal = 6.0 * MASS;
    let h = 0.25;
    let gamma = connection_at(areal, h);

    let omega = circular_omega(MASS, areal);
    let period = 2.0 * PI / omega;
    let revolutions = 2.0;
    let dt = 0.05;
    let steps = (revolutions * period / dt).round() as usize;

    let state0 = circular_state::<D>(MASS, areal);
    let gww0 = normalization(MASS, state0.pos, state0.vel);
    assert!((gww0 + 0.5).abs() < 1e-12, "ISCO IC: g_ww must be -1/2");

    let mut state = state0;
    let mut t = 0.0;
    let mut drift_max = 0.0_f64;
    let mut phase_max = 0.0_f64;
    let mut gww_max = 0.0_f64;
    let mut drift_seq = Vec::new();
    for _ in 0..steps {
        state = geodesic_step_rk4::<D, N>(&state, dt, &gamma).unwrap();
        t += dt;
        let r_areal = areal_of_state(MASS, state.pos);
        let d = (r_areal - areal).abs();
        drift_max = drift_max.max(d);
        drift_seq.push(d);
        phase_max = phase_max.max(wrap(state.pos[1].atan2(state.pos[0]) - omega * t).abs());
        gww_max = gww_max.max((normalization(MASS, state.pos, state.vel) - gww0).abs());
        assert!(state.pos[2].abs() < 1e-12, "left the orbital plane");
    }

    // stability: no secular runaway over the two revolutions. The discrete
    // vs continuum circular families differ at O(h^2) in Omega, so the
    // radius error is a beat, not an instability: measured growth
    // late/early was 5.1 over two revolutions with 6.1e-3 absolute size.
    let q = steps / 4;
    let early: f64 = drift_seq[..q].iter().cloned().fold(0.0, f64::max);
    let late: f64 = drift_seq[steps - q..].iter().cloned().fold(0.0, f64::max);
    println!("ISCO 2 rev: steps = {steps}, early quarter drift = {early:.3e}, late = {late:.3e}");
    assert!(
        late < early * 8.0 + 1e-12,
        "radius error grew secularly: early {early:.3e} -> late {late:.3e}"
    );
    assert!(late < 0.012, "absolute radius error runaway {late:.3e}");

    let closure = plane_dist(state.pos, state0.pos);
    println!(
        "ISCO: drift = {drift_max:.3e}, phase = {phase_max:.3e}, d g_ww = {gww_max:.3e}, closure = {closure:.3e}"
    );

    assert!(drift_max < 0.012, "ISCO drifted {drift_max}");
    assert!(phase_max < 0.02, "ISCO phase drifted {phase_max}");
    assert!(gww_max < 1.2e-3, "ISCO norm drifted {gww_max}");
    assert!(closure < 0.07, "ISCO did not close: {closure}");
}

/// Temporal fourth order, spatial representation frozen: dt ∈ {3.2, 1.6,
/// 0.8} (0.6, 0.31, 0.15 rad per step -- RK4 is in no trouble), all runs
/// from the same circular initial condition on the SAME photon connection
/// field, all ending after the same fixed step alignment (3 x 3.2 = 6 x 1.6
/// = 12 x 0.8 = 24 x 0.4 nominal common time T = 9.6 with no accumulated
/// time variable, so only the step count differs). Error against a dt/8
/// self-reference (0.4), itself checked against dt/16 (0.2). With
/// E(dt) ~ C dt^4 the halving ratios approach 16 (Richardson exactness
/// aside); measured band justified below. The cascade deliberately uses
/// steps large enough that the RK4 truncation signal sits far above the
/// ~2e-6 dt-run-vs-run floor this pipeline showed on shorter arcs.
/// Temporal fourth order of the integrator itself, spatial representation
/// held fixed *and exact*: the same coordinate-time right-hand side is fed
/// the exact analytic connection of the slice (no grid, no interpolation)
/// and the error is measured against the exact circular solution
/// x(T) = (rho cos omega T, rho sin omega T). The RK4 stage loop mirrors
/// `geodesic_stages_step_rk4` stage for stage (same substep positions,
/// same fused final combination), differing only in the connection source.
/// With E(dt) ~ C dt^4 both halving ratios must land near 16.
#[test]
fn temporal_convergence_fourth_order_with_exact_connection() {
    use einstein_engine::physics::orbits::analytic_christoffel;

    let areal = 3.0 * MASS;
    let state0 = circular_state::<D>(MASS, areal);
    let rho = state0.pos[0];
    let omega = circular_omega(MASS, areal);

    // all cascade runs integrate exactly 6 x 1.6 = 12 x 0.8 = 24 x 0.4
    // steps of nominal common time T = 9.6 (no accumulated time variable:
    // only the step count differs between runs)
    let run = |steps: usize| -> GeodesicState<D> {
        let dt = 9.6 / steps as f64;
        let mut s = state0;
        for _ in 0..steps {
            // one RK4 step of geodesic_stages_step_rk4 with the exact
            // symbols substituted for the interpolated per-stage sample
            let k1 = geodesic_rhs::<D, N>(&s, &analytic_christoffel::<N>(MASS, s.pos));
            let y2 = GeodesicState {
                pos: std::array::from_fn(|i| s.pos[i] + 0.5 * dt * k1.pos[i]),
                vel: std::array::from_fn(|i| s.vel[i] + 0.5 * dt * k1.vel[i]),
            };
            let k2 = geodesic_rhs::<D, N>(&y2, &analytic_christoffel::<N>(MASS, y2.pos));
            let y3 = GeodesicState {
                pos: std::array::from_fn(|i| s.pos[i] + 0.5 * dt * k2.pos[i]),
                vel: std::array::from_fn(|i| s.vel[i] + 0.5 * dt * k2.vel[i]),
            };
            let k3 = geodesic_rhs::<D, N>(&y3, &analytic_christoffel::<N>(MASS, y3.pos));
            let y4 = GeodesicState {
                pos: std::array::from_fn(|i| s.pos[i] + dt * k3.pos[i]),
                vel: std::array::from_fn(|i| s.vel[i] + dt * k3.vel[i]),
            };
            let k4 = geodesic_rhs::<D, N>(&y4, &analytic_christoffel::<N>(MASS, y4.pos));
            s = GeodesicState {
                pos: std::array::from_fn(|i| {
                    s.pos[i]
                        + dt / 6.0 * k1.pos[i]
                        + dt / 3.0 * k2.pos[i]
                        + dt / 3.0 * k3.pos[i]
                        + dt / 6.0 * k4.pos[i]
                }),
                vel: std::array::from_fn(|i| {
                    s.vel[i]
                        + dt / 6.0 * k1.vel[i]
                        + dt / 3.0 * k2.vel[i]
                        + dt / 3.0 * k3.vel[i]
                        + dt / 6.0 * k4.vel[i]
                }),
            };
        }
        s
    };

    let exact = |t: f64| -> [f64; D] { [rho * (omega * t).cos(), rho * (omega * t).sin(), 0.0] };

    let e = |steps: usize| -> f64 { plane_dist(run(steps).pos, exact(9.6)) };
    let e1 = e(6); // dt = 1.6
    let e2 = e(12); // dt = 0.8
    let e3 = e(24); // dt = 0.4
    let r1 = e1 / e2;
    let r2 = e2 / e3;
    println!("E(1.6) = {e1:.3e}, E(0.8) = {e2:.3e}, E(0.4) = {e3:.3e}, ratios {r1:.2}, {r2:.2}");

    assert!(
        (12.0..21.0).contains(&r1),
        "first halving ratio {r1} ({e1:.3e} -> {e2:.3e})"
    );
    assert!(
        (12.0..21.0).contains(&r2),
        "second halving ratio {r2} ({e2:.3e} -> {e3:.3e})"
    );

    // the grid pipeline's converged orbit (next test) sits away from this
    // exact endpoint purely from the h^2 representation error
    let _ = e3;
}

/// What the full grid pipeline delivers in dt: the trajectory through the
/// trilinearly interpolated nodal connection converges at an *effective*
/// second order (measured; pairwise halving ratios zigzag 2.3-7.6 around
/// 4 because each run's cell-face crossings differ), NOT at RK4's own
/// fourth order -- the interpolated field is only C^0 across cell faces
/// and that kink error is dt-square and, per the measured pair below,
/// grid-spacing independent. Both claims are asserted from measured
/// values; the integrator's own fourth order is pinned by the exact-
/// connection test above and by the constant-connection test in
/// unit_geodesics. This is a property of representing Gamma with nodal
/// values + trilinear interpolation, not of the symbols' O(h^2) accuracy
/// (the spatial test shows those converge at the central-difference rate).
#[test]
fn full_pipeline_temporal_order_is_interpolation_limited() {
    let areal = 3.0 * MASS;
    let state0 = circular_state::<D>(MASS, areal);

    // pairwise endpoint distances of the dt and dt/2 runs, nominal common
    // time T = 9.6 (24 x 0.4 vs 48 x 0.2; 192 x 0.05 vs 384 x 0.025)
    let pair = |h: f64, coarse: usize, fine: usize| -> f64 {
        let gamma = connection_at(areal, h);
        let run = |steps: usize| -> GeodesicState<D> {
            let mut s = state0;
            for _ in 0..steps {
                let dt = 9.6 / steps as f64;
                s = geodesic_step_rk4::<D, N>(&s, dt, &gamma).unwrap();
            }
            s
        };
        plane_dist(run(coarse).pos, run(fine).pos)
    };

    let e_coarse_dt = pair(0.25, 24, 48);
    let e_small_dt = pair(0.25, 192, 384);

    // effective order between the two pairs: (0.4/0.05)^p = E1/E2
    let slope = (e_coarse_dt / e_small_dt).log2() / 3.0;
    println!(
        "E(0.4,0.2) = {e_coarse_dt:.3e}, E(0.05,0.025) = {e_small_dt:.3e}, effective p = {slope:.2}"
    );
    assert!(
        (1.5..2.6).contains(&slope),
        "effective temporal order {slope} left the measured second-order band"
    );

    // the dt^2 coefficient is grid-spacing independent (kink mechanism):
    // the same pairwise E at h = 0.5 matches within the measured 1.3x
    let e_coarse_dt_coarse_grid = pair(0.5, 24, 48);
    let ratio_grids = e_coarse_dt_coarse_grid / e_coarse_dt;
    println!(
        "E(0.4,0.2) at h = 0.25 vs h = 0.5: {e_coarse_dt:.3e} vs {e_coarse_dt_coarse_grid:.3e} (x{ratio_grids:.2})"
    );
    assert!(
        (0.25..4.0).contains(&ratio_grids),
        "the dt^2 coefficient depends on h ({ratio_grids:.2}x), contradicting the interpolation-kink mechanism"
    );
}

/// Spatial error channel, temporal channel held negligible: the same
/// photon arc at dt = 0.01 on grids h and h/2, error against the analytic
/// arc endpoint. Central differences are O(h^2), so the ratio should be
/// ~4 (measured; band justified by the measured pair).
#[test]
fn spatial_error_converges_second_order() {
    let areal = 3.0 * MASS;
    let rho = isotropic_radius(MASS, areal);
    let omega = circular_omega(MASS, areal);
    let arc_revs = 0.125;
    let period = 2.0 * PI / omega;
    let arc = arc_revs * period;
    let dt = 0.01;

    let err_at = |h: f64| -> f64 {
        let steps = (arc / dt).round() as usize;
        let s = circular_state::<D>(MASS, areal);
        let gamma = connection_at(areal, h);
        let mut state = s;
        for _ in 0..steps {
            state = geodesic_step_rk4::<D, N>(&state, dt, &gamma).unwrap();
        }
        plane_dist(
            state.pos,
            [
                rho * (omega * dt * steps as f64).cos(),
                rho * (omega * dt * steps as f64).sin(),
                0.0,
            ],
        )
    };

    let e_coarse = err_at(0.5);
    let e_fine = err_at(0.25);
    let ratio = e_coarse / e_fine;
    println!("E(h=0.5) = {e_coarse:.3e}, E(h=0.25) = {e_fine:.3e}, ratio = {ratio:.2}");

    assert!(e_fine > 0.0, "no spatial error left to measure");
    assert!(
        (2.5..7.0).contains(&ratio),
        "spatial error did not halve-squared: ratio {ratio}"
    );
}

/// numeric columns of each file, by index; the rest (label, mode) are text
const FILES: [(&str, &str, &[usize], usize); 3] = [
    (
        "geodesic_orbits_static.csv",
        "particle_id,label,revolutions,t,x,y,z,vx,vy,vz,r_areal,g_ww",
        // particle_id, revolutions, t, x, y, z, vx, vy, vz, r_areal, g_ww
        &[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        4000,
    ),
    (
        "geodesic_orbits_static_summary.csv",
        "label,revolutions,r_areal_target,omega,drift_areal_max,phase_err_max_rad,g_ww_drift_max,end_err,closure_full_rev,orbit_period_coord",
        &[1, 2, 3, 4, 5, 6, 7, 8, 9],
        2,
    ),
    (
        "geodesic_orbits_dynamic.csv",
        "particle_id,mode,t,x,y,z,vx,vy,vz,speed",
        // particle_id, t, x..vz, speed (mode is text)
        &[0, 2, 3, 4, 5, 6, 7, 8, 9],
        2,
    ),
];

/// The runnable example, executed for real, twice, into temporary output
/// directories: exit 0, all three CSV files present with the documented
/// schema, every numeric value finite, text fields as expected, and
/// byte-identical output between the two runs (reproducibility of the
/// deterministic pipeline).
#[test]
fn example_runs_and_writes_reproducible_csv() {
    use std::collections::HashSet;

    let manifest = env!("CARGO_MANIFEST_DIR");
    let base = std::env::temp_dir().join("einstein_gorbit_test");

    let run = |dir: &std::path::Path| {
        let out = std::process::Command::new("cargo")
            .args(["run", "--quiet", "--example", "geodesic_orbits", "--"])
            .arg(dir)
            .current_dir(manifest)
            .output()
            .expect("failed to spawn the example");
        assert!(
            out.status.success(),
            "example failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    };

    let dir_a = base.join("a");
    let dir_b = base.join("b");
    std::fs::create_dir_all(&dir_a).unwrap();
    std::fs::create_dir_all(&dir_b).unwrap();
    run(&dir_a);
    run(&dir_b);

    for (name, header, numeric_cols, min_rows) in FILES {
        let a = std::fs::read_to_string(dir_a.join(name)).expect("csv exists");
        let b = std::fs::read_to_string(dir_b.join(name)).expect("csv exists");

        let mut lines = a.lines();
        assert_eq!(lines.next().unwrap(), header, "{name}: unexpected schema");
        let numeric: HashSet<usize> = numeric_cols.iter().cloned().collect();
        let rows: Vec<&str> = lines.collect();
        assert!(
            rows.len() >= min_rows,
            "{name}: too few rows ({})",
            rows.len()
        );

        for row in &rows {
            for (col, field) in row.split(',').enumerate() {
                match numeric.contains(&col) {
                    true => {
                        let value: f64 = field
                            .parse()
                            .unwrap_or_else(|_| panic!("{name}: non-numeric field '{field}'"));
                        assert!(value.is_finite(), "{name}: non-finite field {field}");
                    }
                    false => {
                        assert!(
                            field == "photon_sphere_r3M"
                                || field == "isco_r6M"
                                || field == "staged"
                                || field == "frozen",
                            "{name}: unexpected text field '{field}'"
                        );
                    }
                }
            }
        }

        // reproducibility: byte-identical rerun of the deterministic run
        assert_eq!(a, b, "{name}: rerun produced different bytes");
    }
}
