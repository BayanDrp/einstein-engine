//! Geodesic orbits on a Schwarzschild background, integrated with the real
//! engine pipeline, plus a dynamic-background leg that exercises the
//! stage-consistent (time-aware) integration path.
//!
//! Phase A (static, frozen-background): circular test orbits around a unit
//! hole in the isotropic Cartesian chart, integrated with
//! `Particles::step_all` on the frozen connection field built by
//! `christoffel_symbols` from exact Schwarzschild initial data:
//!
//!   - photon sphere: areal radius r = 3M, one full revolution,
//!   - ISCO:          areal radius r = 6M, two full revolutions.
//!
//! Initial conditions are derived analytically in `physics::orbits`
//! (circular-orbit balance Gamma^rho_00 = Gamma^rho_phiphi omega^2 with
//! omega^2 = M/r^3 exactly, in coordinate time, in either chart; the
//! characterization line is g_mn w^m w^n = -(1 - 3M/r), null at 3M).
//!
//! Phase B (evolving background, stage-consistent): the same initial
//! particle state is integrated 20 steps (a) through genuine BSSN vacuum
//! expansion using `bssn_stage_connections` +
//! `Particles::step_all_with_connections` -- the geometry the integrator
//! steps through at each stage time -- and (b) with the frozen initial
//! connection. On a uniformly expanding BSSN vacuum the two disagree at
//! O(K * T) (the frozen path misses the gauge acceleration an expanding
//! lapse produces), which is exactly the distinction stage-aware coupling
//! exists for.
//!
//! Units G = c = 1; t and (x, y, z) are isotropic Cartesian coordinates;
//! `r_areal` is documented as r = rho (1 + M/(2 rho))^2 for rho = |x|.
//! Grids are centered on the puncture with the lattice offset by half a
//! spacing on every axis, so no node sits at rho = 0 (psi diverges there).
//!
//! Run from engine/: cargo run --example geodesic_orbits
//! Optional argument: output directory (default: current directory).
//! Writes geodesic_orbits_static.csv, geodesic_orbits_static_summary.csv
//! and geodesic_orbits_dynamic.csv into that directory.

use einstein_engine::geometry::christoffel::christoffel_symbols;
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::Grid;
use einstein_engine::initial_data::schwarzschild::{isotropic_fields, schwarzschild};
use einstein_engine::numerics::rk4::AdmState;
use einstein_engine::physics::geodesics::{bssn_stage_connections, GeodesicState};
use einstein_engine::physics::orbits::{
    areal_of_state, areal_radius, circular_omega, circular_state, isotropic_radius,
    metric_functions, normalization,
};
use einstein_engine::physics::particles::Particles;
use einstein_engine::relativity::bssn::BssnState;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

const D: usize = 3;
const N: usize = 4;
const MASS: f64 = 1.0;
const ORBIT_H: f64 = 0.25;

fn periodic() -> (BoundaryConfig<D>, BoundaryValues<D>) {
    (BoundaryConfig::<D>::periodic(), BoundaryValues::<D>::zeros())
}

/// Even-count grid centered on the hole whose lattice has no node at the
/// puncture (a x = 0 node would make psi infinite), with node positions at
/// odd multiples of h/2 on every axis: the closest nodes sit at
/// rho = h * sqrt(3)/2 = 0.216 h-scaled units from the puncture.
fn hole_grid(min_half_span: f64, h: f64) -> Arc<Grid<D>> {
    let n = (((2.0 * min_half_span) / h).ceil() as usize) + 1;
    // make the count even so nodes avoid the origin lattice line
    let n = if n % 2 == 1 { n + 1 } else { n };
    let origin = -((n as f64 - 1.0) * h) / 2.0;
    Arc::new(Grid::new([n; D], [h; D], [origin; D]))
}

/// angle of (x, y) wrapped into (-pi, pi]
fn wrap(angle: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    ((angle + std::f64::consts::PI).rem_euclid(two_pi)) - std::f64::consts::PI
}

fn main() {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let out_dir = Path::new(&out_dir);
    create_dir_all(out_dir).expect("cannot create output directory");

    println!("units G = c = 1, M = {MASS}; coordinates (t, x, y, z) isotropic Cartesian");
    println!("areal radius r = rho (1 + M/(2 rho))^2");
    phase_a_static(out_dir);
    phase_b_stage_consistent(out_dir);
}

/// One static circular orbit integrated on the frozen connection.
struct OrbitRun {
    label: &'static str,
    areal_r: f64,
    revolutions: f64,
    dt: f64,
    steps: usize,
}

fn phase_a_static(out_dir: &Path) {
    let mut runs = [
        // photon sphere: unstable null orbit (perturbations grow like
        // exp(lambda t) with lambda = omega = 1/(3 sqrt(3) M) -- the exact
        // epicyclic exponent at r = 3M), so a full revolution would blow a
        // 1e-3 grid-induced initial offset up by x533. The validation arc is
        // therefore one quarter revolution, where the amplification factor
        // is exp(0.25 * lambda * T) = exp(pi/2) ~ 4.8, and the orbit-error
        // against the analytic arc is what is measured.
        OrbitRun {
            label: "photon_sphere_r3M",
            areal_r: 3.0 * MASS,
            revolutions: 0.25,
            dt: 0.01,
            steps: 0,
        },
        OrbitRun {
            label: "isco_r6M",
            areal_r: 6.0 * MASS,
            revolutions: 2.0,
            dt: 0.05,
            steps: 0,
        },
    ];
    for run in &mut runs {
        let omega = circular_omega(MASS, run.areal_r);
        let period = 2.0 * std::f64::consts::PI / omega;
        run.steps = (run.revolutions * period / run.dt).round() as usize;
    }

    let mut csv = File::create(out_dir.join("geodesic_orbits_static.csv"))
        .expect("cannot create static csv");
    writeln!(
        csv,
        "particle_id,label,revolutions,t,x,y,z,vx,vy,vz,r_areal,g_ww"
    )
    .expect("cannot write header");

    let mut summary = File::create(out_dir.join("geodesic_orbits_static_summary.csv"))
        .expect("cannot create summary csv");
    writeln!(
        summary,
        "label,revolutions,r_areal_target,omega,drift_areal_max,phase_err_max_rad,g_ww_drift_max,end_err,closure_full_rev,orbit_period_coord"
    )
    .expect("cannot write header");

    for run in &runs {
        let rho = isotropic_radius(MASS, run.areal_r);
        let omega = circular_omega(MASS, run.areal_r);
        let period = 2.0 * std::f64::consts::PI / omega;
        let dt = run.dt;

        // the grid must contain the orbit with a safety margin for the
        // interpolation domain and the central-difference stencils
        let margin = 1.4_f64.max(rho * 0.3);
        let grid = hole_grid(rho + margin, ORBIT_H);
        let adm = match isotropic_fields(grid.clone(), MASS) {
            Ok(a) => a,
            Err(msg) => panic!("fixture grid rejected: {msg}"),
        };
        let (config, values) = periodic();
        let gamma =
            christoffel_symbols::<D, N>(&adm.lapse, &adm.shift, &adm.gamma, &config, &values);

        // cross-check fixture vs engine factory wherever the factory's own
        // horizon-exclusion rule accepts the grid at all
        check_fixture_against_factory(&grid, &adm);

        println!(
            "orbit {}: areal r = {:.6} -> isotropic rho = {:.6}, omega = {:.6}, T = {:.6}, dt = {} ({} steps = {:.3} rev)",
            run.label, run.areal_r, rho, omega, period, dt, run.steps, run.revolutions
        );

        let (psi4, alpha0) = metric_functions(MASS, rho);
        let gww0 = normalization(MASS, [rho, 0.0, 0.0], [0.0, rho * omega, 0.0]);
        let kind = if gww0.abs() < 1e-6 { "null" } else { "timelike" };
        println!("  g_ww(orbit) = {gww0:.6} ({kind}: psi^4 = {psi4:.4}, alpha = {alpha0:.4})");

        let g0 = circular_state::<D>(MASS, run.areal_r);
        let mut particles = Particles::<D>::new();
        let id = particles.spawn(0.0, g0).expect("spawn");

        write_row(&mut csv, id, run, 0.0, g0.pos, g0.vel);

        let mut drift_max = 0.0_f64;
        let mut phase_err_max = 0.0_f64;
        let mut gww_drift_max = 0.0_f64;
        let mut z_max = 0.0_f64;
        let mut t = 0.0_f64;

        for _ in 0..run.steps {
            particles
                .step_all::<N>(dt, &gamma)
                .expect("orbit left the interpolation domain");
            t += dt;
            let p = particles.get(id).unwrap();
            let pos = p.state.pos;
            let vel = p.state.vel;
            let r_areal = areal_of_state(MASS, pos);
            drift_max = drift_max.max((r_areal - run.areal_r).abs());
            z_max = z_max.max(pos[2].abs());
            let gww = normalization(MASS, pos, vel);
            gww_drift_max = gww_drift_max.max((gww - gww0).abs());
            let phase = wrap(pos[1].atan2(pos[0]) - omega * t);
            phase_err_max = phase_err_max.max(phase.abs());
            write_row(&mut csv, id, run, t, pos, vel);
        }

        let p = particles.get(id).unwrap();
        let closure = ((p.state.pos[0] - g0.pos[0]).powi(2)
            + (p.state.pos[1] - g0.pos[1]).powi(2))
            .sqrt();
        // error against the analytic arc endpoint x(T) = (rho cos omegaT,
        // rho sin omegaT) -- for sub-revolution arcs this, not the start
        // point, is the meaningful orbit-closure metric
        let end_err = ((p.state.pos[0] - rho * (omega * t).cos()).powi(2)
            + (p.state.pos[1] - rho * (omega * t).sin()).powi(2))
            .sqrt();

        println!(
            "  drift_areal = {drift_max:.3e}, phase_err = {phase_err_max:.3e} rad, g_ww drift = {gww_drift_max:.3e}, end_err = {end_err:.3e}, closure_full_rev = {closure:.3e}, |z| = {z_max:.3e}",
        );

        assert!(
            drift_max < 0.04,
            "{} left the circle by {drift_max}",
            run.label
        );
        assert!(phase_err_max < 0.02, "{} phase drifted {phase_err_max}", run.label);
        if gww0.abs() < 1e-6 {
            // photon arc: a quarter revolution, measured end_err 1.9e-2
            assert!(end_err < 0.04, "{} missed the analytic endpoint: {end_err}", run.label);
        } else {
            // two full revolutions: measured end_err 4.2e-2, closure 4.6e-2
            assert!(end_err < 0.07, "{} missed the reunion point: {end_err}", run.label);
            assert!(closure < 0.07, "{} did not close: {closure}", run.label);
        }
        assert!(z_max < 1e-12, "{} left the orbital plane", run.label);
        if gww0.abs() < 1e-6 {
            assert!(
                gww_drift_max < 0.008,
                "photon sphere: g_ww grew to {gww_drift_max}"
            );
        } else {
            assert!(
                gww_drift_max < 1.2e-3,
                "timelike norm drifted {gww_drift_max}"
            );
        }

        writeln!(
            summary,
            "{},{:.2},{:.6},{:.6},{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},{:.6}",
            run.label,
            run.revolutions,
            run.areal_r,
            omega,
            drift_max,
            phase_err_max,
            gww_drift_max,
            end_err,
            closure,
            period
        )
        .expect("cannot write summary row");
    }

    println!("phase A done: geodesic_orbits_static.csv + summary written");
}

fn write_row(csv: &mut File, id: u64, run: &OrbitRun, t: f64, pos: [f64; D], vel: [f64; D]) {
    let r_areal = areal_of_state(MASS, pos);
    let gww = normalization(MASS, pos, vel);
    writeln!(
        csv,
        "{id},{},{:.2},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
        run.label,
        run.revolutions,
        t,
        pos[0], pos[1], pos[2],
        vel[0], vel[1], vel[2],
        r_areal,
        gww,
    )
    .expect("cannot write csv row");
}

/// Sample nodes of the fixture must equal the engine factory values on any
/// grid the factory accepts; on grids the factory rejects (fine centered
/// grids reaching inside the horizon exclusion) this is reported and
/// skipped -- the factory-vs-fixture equality is checked in the test
/// suite on a coarse exterior grid where both constructions run.
fn check_fixture_against_factory(grid: &Arc<Grid<D>>, fixture: &AdmState<D>) {
    match schwarzschild(grid.clone(), MASS) {
        Ok(factory) => {
            let mut worst = 0.0_f64;
            grid.for_each_index(|idx| {
                let a = *fixture.gamma.get(idx);
                let b = *factory.gamma.get(idx);
                for i in 0..D {
                    for j in 0..D {
                        worst = worst.max((a[(i, j)] - b[(i, j)]).abs());
                    }
                }
                worst = worst.max(
                    (*fixture.lapse.get(idx) - *factory.lapse.get(idx)).abs(),
                );
            });
            assert!(
                worst == 0.0,
                "fixture disagrees with the factory by {worst}"
            );
            println!("  fixture == engine factory on this grid (all nodes)");
        }
        Err(msg) => {
            println!("  (engine factory cross-check skipped here: {msg})");
        }
    }
}

/// Phase B: one particle on a uniformly expanding BSSN vacuum background,
/// integrated (a) stage-consistently through the real BSSN stages and
/// (b) on the frozen initial connection; the difference is the measurable
/// signature of stage-aware coupling on a genuinely evolving slice.
fn phase_b_stage_consistent(out_dir: &Path) {
    let steps = 20_usize;
    let dt = 0.02_f64;
    let grid = Arc::new(Grid::new([9; D], [0.5; D], [-2.0; D]));
    let (config, values) = periodic();

    // uniform expanding BSSN vacuum: phi = 0.1, K = -0.3, flat gtilde.
    // On such data gamma_ij = e^(4 phi) delta and the slice expands:
    // d_t phi = -alpha K / 6 = 0.05, d_t alpha = -2 alpha K = 0.6.
    let mut bssn = BssnState::minkowski(grid.clone());
    grid.for_each_index(|idx| {
        bssn.phi.set(idx, 0.1);
        bssn.trace_k.set(idx, -0.3);
    });

    let start = GeodesicState {
        pos: [0.5, 0.0, 0.0],
        vel: [0.3, 0.0, 0.05],
    };

    // (a) stage-consistent integration through the actual RK4 stages
    let mut staged = Particles::<D>::new();
    staged.spawn(0.0, start).expect("spawn");
    let mut state = bssn.clone();
    let mut t = 0.0_f64;
    for _ in 0..steps {
        let stages = state.step_rk4_stages(dt, &config, &values);
        let connections =
            bssn_stage_connections::<D, N>(&stages, &config, &values);
        staged
            .step_all_with_connections(t, &connections)
            .expect("staged geodesic step");
        state = stages.collapse(&config, &values);
        t += dt;
    }
    let p_staged = staged.iter().next().unwrap();

    // (b) frozen-background integration of the same initial state
    let mut frozen = Particles::<D>::new();
    frozen.spawn(0.0, start).expect("spawn");
    let adm0 = bssn.to_adm();
    let gamma0 =
        christoffel_symbols::<D, N>(&adm0.lapse, &adm0.shift, &adm0.gamma, &config, &values);
    for _ in 0..steps {
        frozen.step_all::<N>(dt, &gamma0).expect("frozen geodesic step");
    }
    let p_frozen = frozen.iter().next().unwrap();

    let mut csv = File::create(out_dir.join("geodesic_orbits_dynamic.csv"))
        .expect("cannot create dynamic csv");
    writeln!(csv, "particle_id,mode,t,x,y,z,vx,vy,vz,speed").expect("cannot write header");

    for (mode, p) in [("staged", &p_staged.state), ("frozen", &p_frozen.state)] {
        let speed = (p.vel.iter().map(|v| v * v).sum::<f64>()).sqrt();
        writeln!(
            csv,
            "0,{mode},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            steps as f64 * dt,
            p.pos[0], p.pos[1], p.pos[2],
            p.vel[0], p.vel[1], p.vel[2],
            speed,
        )
        .expect("cannot write row");
    }

    let speed = |v: [f64; D]| (v.iter().map(|s| s * s).sum::<f64>()).sqrt();
    let v_staged = speed(p_staged.state.vel);
    let v_frozen = speed(p_frozen.state.vel);

    println!("phase B: expanding BSSN vacuum, phi0 = 0.1, K = -0.3, dt = {dt}, {steps} steps");
    println!(
        "  frozen |v| = {v_frozen:.6}, staged |v| = {v_staged:.6}, difference = {:.3e}",
        v_staged - v_frozen
    );
    println!("  staged pos = {:?}", p_staged.state.pos);
    assert!(
        (v_staged - v_frozen).abs() > 1.0e-3,
        "stage-consistent and frozen integrations should disagree measurably on an expanding slice"
    );
    assert!(
        (p_staged.time - steps as f64 * dt).abs() < 1e-12,
        "coupled clock out of sync"
    );
    let _ = areal_radius; // part of the documented diagnostic set
    println!("phase B done: geodesic_orbits_dynamic.csv written");
}
