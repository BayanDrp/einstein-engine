//! The stage-consistent matter coupling from both formulations' sides.
//!
//! M5 made the coupled dust steppers expose their RK4 stages:
//! `AdmState::step_rk4_with_dust_stages` (new -- ADM previously only had
//! the frozen-source `step_rk4_with_matter`) and
//! `BssnState::step_rk4_with_dust_stages` (the classic
//! `step_rk4_with_dust` refactored into a literal collapse of these
//! stages). The suite proves, per category:
//!
//! 1. per-stage source rebuild: every returned (y_i, m_i, k_gi, k_mi)
//!    pair is exactly the evaluation of the pure kernels at that pair;
//! 2. bit-exact refactor: the classic BSSN coupled stepper returns the
//!    bitwise collapse of the exposed stages;
//! 3. structural identity: the ADM stage bundles carry the true metric
//!    rate per stage (`Gamma^x_0x = alpha K_xx / gamma_xx` on data with
//!    a spatially uniform metric), the wired-in discipline the geodesic
//!    couplings consume;
//! 4. freshness is measurable: staged != frozen-source on evolving
//!    geometry, with a static control where they must agree exactly;
//! 5. vacuum limit: empty dust reduces both coupled steppers to the
//!    vacuum steppers exactly;
//! 6. particle clock sync on the dust-coupled BSSN stage bundles;
//! 7. boundary-stop and exemption policy on the same bundles;
//! 8. FLRW manufactured references: mass losslessly conserved, slice
//!    bitwise uniform, constraints held, and a ten-step stage-driven run
//!    bitwise-equal to the classic path;
//! 9. the frozen-geometry dust stepper the stage steppers reuse for
//!    transport-only work is unaffected by the refactor (M3 pin).

use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::numerics::rk4::AdmState;
use einstein_engine::physics::dust::{dust_sources, DustState};
use einstein_engine::physics::dust_evolution::{dust_rhs, dust_step_rk4};
use einstein_engine::physics::geodesics::{
    bssn_stage_connections, geodesic_stages_step_rk4, stage_connections, GeodesicState,
    StageConnections,
};
use einstein_engine::physics::particles::{ParticleError, ParticleStatus, Particles};
use einstein_engine::relativity::bssn::BssnState;
use einstein_engine::validation::constraints::residuals;
use std::sync::Arc;

const D: usize = 3;
const N: usize = 4;

fn periodic() -> (BoundaryConfig<D>, BoundaryValues<D>) {
    (
        BoundaryConfig::<D>::periodic(),
        BoundaryValues::<D>::zeros(),
    )
}

/// FLRW-like homogeneous patch: flat uniform metric with curvature K
/// (gamma evolves as -2 alpha K on the whole field), uniform dust at rest.
/// The density is constraint-consistent with the initial trace, so |H|
/// starts at roundoff. 8x8x8 nodes, spacing 0.125, origin 0.5.
const C_TRACE: f64 = 0.1;

fn flrw() -> (Arc<Grid<D>>, AdmState<D>, DustState<D>) {
    let g = Arc::new(Grid::new([8; D], [0.125; D], [0.5; D]));
    let rho0 = 6.0 * C_TRACE * C_TRACE / (16.0 * std::f64::consts::PI);

    let mut identity = Tensor2::<D>::zero();
    let mut k0 = Tensor2::<D>::zero();
    for i in 0..D {
        identity[(i, i)] = 1.0;
        k0[(i, i)] = C_TRACE;
    }
    let gamma = Field::new(g.clone(), identity);
    let adm = AdmState {
        grid: g.clone(),
        gamma: gamma.clone(),
        k: Field::new(g.clone(), k0),
        lapse: Field::new(g.clone(), 1.0),
        shift: Field::new(g.clone(), Vector::<D>::zero()),
    };
    let dust = DustState::from_primitives(
        &Field::new(g.clone(), rho0),
        &Field::new(g.clone(), Vector::<D>::zero()),
        &gamma,
    );
    (g, adm, dust)
}

fn flrw_bssn() -> (BssnState<D>, DustState<D>) {
    let (config, values) = periodic();
    let (_, adm, dust) = flrw();
    let bssn = BssnState::<D>::from_adm(&adm, &config, &values);
    (bssn, dust)
}

/// FLRW metric plus a periodic mass wave:
/// rho(x) = rho0 (1 + 0.25 sin 2pi x), whose unit cell is exactly the
/// 8-cell domain length, so the periodic seam samples consistent values
/// and every stage branch of the coupled steppers is driven.
fn flrw_wave_dust() -> DustState<D> {
    let g = Arc::new(Grid::new([8; D], [0.125; D], [0.5; D]));
    let rho0 = 6.0 * C_TRACE * C_TRACE / (16.0 * std::f64::consts::PI);
    let mut identity = Tensor2::<D>::zero();
    for i in 0..D {
        identity[(i, i)] = 1.0;
    }
    let gamma = Field::new(g.clone(), identity);
    let mut rho = Field::new(g.clone(), 0.0);
    g.for_each_index(|idx| {
        let x = idx[0] as f64 * g.spacing[0] + g.origin[0];
        rho.set(
            idx,
            rho0 * (1.0 + 0.25 * (2.0 * std::f64::consts::PI * x).sin()),
        );
    });
    DustState::from_primitives(&rho, &Field::new(g.clone(), Vector::<D>::zero()), &gamma)
}

/// Flat metric with a spatially varying contraction rate
/// K_xx(x) = 0.05 (1 + 0.3 sin 2pi x): the metric becomes inhomogeneous
/// *within* the first step, so the matter source term
/// (alpha sqrt gamma / 2) T^{mu nu} d_j g_{mu nu} turns on at stage 2 and
/// the per-stage rebuild produces measurably different dust rates.
fn strain_adm_with_dust() -> (AdmState<D>, DustState<D>) {
    let g = Arc::new(Grid::new([8; D], [0.125; D], [0.5; D]));
    let mut identity = Tensor2::<D>::zero();
    for i in 0..D {
        identity[(i, i)] = 1.0;
    }
    let gamma = Field::new(g.clone(), identity);
    let mut k = Field::new(g.clone(), Tensor2::<D>::zero());
    g.for_each_index(|idx| {
        let x = idx[0] as f64 * g.spacing[0] + g.origin[0];
        let mut kk = Tensor2::<D>::zero();
        kk[(0, 0)] = 0.05 * (1.0 + 0.3 * (2.0 * std::f64::consts::PI * x).sin());
        k.set(idx, kk);
    });
    let adm = AdmState {
        grid: g.clone(),
        gamma: gamma.clone(),
        k,
        lapse: Field::new(g.clone(), 1.0),
        shift: Field::new(g.clone(), Vector::<D>::zero()),
    };
    let dust = DustState::from_primitives(
        &Field::new(g.clone(), 1e-3),
        &Field::new(g.clone(), Vector::<D>::zero()),
        &gamma,
    );
    (adm, dust)
}

/// exactly zero-density, zero-momentum dust on the given grid (the
/// `d_tilde <= 0` recovery floor turns it into rho = 0, v = 0 exactly)
fn zero_dust(grid: &Arc<Grid<D>>) -> DustState<D> {
    let mut identity = Tensor2::<D>::zero();
    for i in 0..D {
        identity[(i, i)] = 1.0;
    }
    let gamma = Field::new(grid.clone(), identity);
    DustState::from_primitives(
        &Field::new(grid.clone(), 0.0),
        &Field::new(grid.clone(), Vector::<D>::zero()),
        &gamma,
    )
}

fn scalar_max_diff(a: &Field<f64, D>, b: &Field<f64, D>) -> f64 {
    let mut worst = 0.0f64;
    a.grid.for_each_index(|idx| {
        worst = worst.max((a.get(idx) - b.get(idx)).abs());
    });
    worst
}

fn tensor_max_diff(a: &Field<Tensor2<D>, D>, b: &Field<Tensor2<D>, D>) -> f64 {
    let mut worst = 0.0f64;
    a.grid.for_each_index(|idx| {
        for i in 0..D {
            for j in 0..D {
                worst = worst.max((a.get(idx)[(i, j)] - b.get(idx)[(i, j)]).abs());
            }
        }
    });
    worst
}

fn vector_max_diff(a: &Field<Vector<D>, D>, b: &Field<Vector<D>, D>) -> f64 {
    let mut worst = 0.0f64;
    a.grid.for_each_index(|idx| {
        for i in 0..D {
            worst = worst.max((a.get(idx)[i] - b.get(idx)[i]).abs());
        }
    });
    worst
}

/// value distance between two dust states (f64 == semantics, so a
/// difference in the sign of zero does not count; the steppers are judged
/// on values, not on sign-of-zero bits)
fn dust_max_diff(a: &DustState<D>, b: &DustState<D>) -> f64 {
    scalar_max_diff(&a.densitized_rest_mass, &b.densitized_rest_mass).max(vector_max_diff(
        &a.densitized_momentum,
        &b.densitized_momentum,
    ))
}

fn dust_total(dust: &DustState<D>) -> f64 {
    let h = dust.grid.spacing[0];
    let mut total = 0.0f64;
    dust.grid.for_each_index(|idx| {
        total += dust.densitized_rest_mass.get(idx);
    });
    total * h.powi(3)
}

/// value distance over the full BSSN state (all seven carried fields)
fn bssn_max_diff(a: &BssnState<D>, b: &BssnState<D>) -> f64 {
    scalar_max_diff(&a.phi, &b.phi)
        .max(tensor_max_diff(&a.conformal_metric, &b.conformal_metric))
        .max(scalar_max_diff(&a.trace_k, &b.trace_k))
        .max(tensor_max_diff(&a.conformal_a, &b.conformal_a))
        .max(vector_max_diff(&a.conformal_gamma, &b.conformal_gamma))
        .max(scalar_max_diff(&a.lapse, &b.lapse))
        .max(vector_max_diff(&a.shift, &b.shift))
}

/// value distance over the full ADM state (all four carried fields)
fn adm_max_diff(a: &AdmState<D>, b: &AdmState<D>) -> f64 {
    tensor_max_diff(&a.gamma, &b.gamma)
        .max(tensor_max_diff(&a.k, &b.k))
        .max(scalar_max_diff(&a.lapse, &b.lapse))
        .max(vector_max_diff(&a.shift, &b.shift))
}

/// |H| and |M| of an ADM state with dust-consistent sources
fn adm_constraints(
    adm: &AdmState<D>,
    dust: &DustState<D>,
    config: &BoundaryConfig<D>,
    values: &BoundaryValues<D>,
) -> (f64, f64) {
    let src = dust_sources::<N, D>(dust, &adm.lapse, &adm.shift, &adm.gamma).unwrap();
    let r = residuals(&adm.gamma, &adm.k, &src.rho, &src.momentum, config, values);
    (r.hamiltonian_max, r.momentum_max)
}

// 1. Per-stage source rebuild proof (ADM): every returned dust rate k_mi
// is bitwise the frozen-geometry kernel evaluated at that stage's own
// (m_i, y_i) pair, and every returned geometry rate is bitwise
// `rhs_with_matter` of the sources rebuilt at that same pair. A stepper
// that sourced any stage from a different pair (stale sources, cross-
// wired pairing) fails. On the uniform FLRW patch the dust rates are all
// bitwise zero -- dust feels geometry through metric *gradients* and the
// flux (alpha v - beta), both identically zero here -- while the
// geometry rates genuinely change across stages because the rebuilt
// sources scale with the beta-independent W = 1/alpha of each stage;
// both facts are asserted so the fidelity cannot pass vacuously.
#[test]
fn adm_dust_stage_pairs_are_self_consistent() {
    let (config, values) = periodic();
    let (_, adm, dust) = flrw();

    let (stages, d_stages) = adm
        .step_rk4_with_dust_stages(&dust, 0.03125, &config, &values)
        .unwrap();

    let geoms = [&stages.y0, &stages.y2, &stages.y3, &stages.y4];
    let dust_states = [&d_stages.m0, &d_stages.m2, &d_stages.m3, &d_stages.m4];
    let rates_m = [&d_stages.k1, &d_stages.k2, &d_stages.k3, &d_stages.k4];
    let rates_g = [&stages.k1, &stages.k2, &stages.k3, &stages.k4];

    for i in 0..4 {
        let rebuilt_m = dust_rhs(
            dust_states[i],
            &geoms[i].lapse,
            &geoms[i].shift,
            &geoms[i].gamma,
            &config,
            &values,
        );
        assert_eq!(
            dust_max_diff(&rebuilt_m, rates_m[i]),
            0.0,
            "k_m stage {i} is not the dust rate of its own (m, y) pair"
        );

        let src = dust_sources::<N, D>(
            dust_states[i],
            &geoms[i].lapse,
            &geoms[i].shift,
            &geoms[i].gamma,
        )
        .unwrap();
        let rebuilt_g = geoms[i].rhs_with_matter(&src, &config, &values);
        assert_eq!(
            adm_max_diff(&rebuilt_g, rates_g[i]),
            0.0,
            "k_g stage {i} is not the geometry rate of its own (m, y) pair"
        );

        // rest dust on a uniform, evolving metric: no gradient force, no
        // flux, so the rate must be the zero state exactly
        assert_eq!(dust_max_diff(&d_stages.k1, &d_stages.k2), 0.0);
    }

    // non-vacuity for the geometry side: the stage states move and the
    // geometry rates differ, so the pairwise fidelity means something
    assert!(
        adm_max_diff(&stages.y2, &stages.y0) > 0.0,
        "the stage states never moved"
    );
    assert!(
        adm_max_diff(&stages.k1, &stages.k2) > 0.0,
        "geometry rates identical across stages: source rebuild not exercised"
    );
}

// 2. Bit-exact refactor proof (BSSN): the classic `step_rk4_with_dust`
// returns exactly what collapsing the exposed stages produces, on an
// inhomogeneous fixture (mass wave) that drives every stage branch.
#[test]
fn bssn_dust_stepper_is_the_literal_collapse_of_its_stages() {
    let (config, values) = periodic();
    let (bssn, _) = flrw_bssn();
    let dust = flrw_wave_dust();

    let dt = 0.5 * 0.125;
    let (next, next_m) = bssn
        .step_rk4_with_dust(&dust, dt, &config, &values)
        .unwrap();

    let (stages, d_stages) = bssn
        .step_rk4_with_dust_stages(&dust, dt, &config, &values)
        .unwrap();
    let collapsed = stages.collapse(&config, &values);
    let collapsed_m = d_stages.collapse();

    assert_eq!(
        bssn_max_diff(&next, &collapsed),
        0.0,
        "BSSN state: the classic stepper and the stage collapse disagree"
    );
    assert_eq!(
        dust_max_diff(&next_m, &collapsed_m),
        0.0,
        "dust state: the classic stepper and the stage collapse disagree"
    );
}

// 3. Structural identity per stage (ADM): with a spatially uniform metric
// the advection and shift-strain terms vanish exactly, so the metric
// component of each bundle must satisfy the wiring-only identity
// Gamma^x_0x = -alpha K_xx / gamma_xx (the engine's K is positive-
// contracting, so a growing universe with K < 0 gives Gamma^x_0x > 0,
// the Hubble term), with every factor taken from that stage's own y_i.
// This is the ADM counterpart of the BSSN per-stage identity
// Gamma^x_0x = 2 d_t phi already pinned by unit_bssn_geodesics.
#[test]
fn adm_stage_bundles_carry_the_true_metric_rate() {
    let (config, values) = periodic();
    let (_, adm, dust) = flrw();

    let (stages, _) = adm
        .step_rk4_with_dust_stages(&dust, 0.03125, &config, &values)
        .unwrap();

    let conns: StageConnections<N, D> = stage_connections(&stages, &config, &values);

    let geoms = [&stages.y0, &stages.y2, &stages.y3, &stages.y4];
    for (i, y) in geoms.iter().enumerate() {
        y.grid.for_each_index(|idx| {
            let gamma = y.gamma.get(idx);
            let kxx = y.k.get(idx)[(0, 0)];
            let alpha = *y.lapse.get(idx);
            let identity = -alpha * kxx / gamma[(0, 0)];

            let wired = conns.connections[i].get(idx).data[1][0][1];
            assert!(
                (wired - identity).abs() <= 1e-12 * identity.abs().max(1e-6),
                "stage {i}: Gamma^x_0x = {wired} vs alpha K_xx/gamma_xx = {identity} at {:?}",
                idx
            );
        });
    }
}

// 4. Freshness is measurable (ADM): on the strain patch (inhomogeneous
// metric within the step) one staged step must differ from the frozen-
// source stepper in both systems, and the stage dust rates must differ
// from each other (the momentum source turns on at stage 2, when the
// metric first has gradients); and the staged path is the bitwise
// collapse of its stages on both systems (the ADM twin of test 2).
#[test]
fn fresh_sources_leave_a_measurable_stamp() {
    let (config, values) = periodic();
    let (adm, dust) = strain_adm_with_dust();

    let dt = 0.0625;
    let (stages, d_stages) = adm
        .step_rk4_with_dust_stages(&dust, dt, &config, &values)
        .unwrap();
    let staged = stages.collapse(&config, &values);
    let staged_dust = d_stages.collapse();

    let frozen = dust_sources::<N, D>(&dust, &adm.lapse, &adm.shift, &adm.gamma).unwrap();
    let frozeng = adm.step_rk4_with_matter(dt, &frozen, &config, &values);
    let frozend = dust_step_rk4(
        &dust, dt, &adm.lapse, &adm.shift, &adm.gamma, &config, &values,
    );

    let dg = adm_max_diff(&staged, &frozeng);
    let dm = dust_max_diff(&staged_dust, &frozend);
    println!(
        "strain staged-vs-frozen: geometry {dg:.3e}, dust {dm:.3e}; dust rates across stages:"
    );
    for (a, b) in [
        (&d_stages.k1, &d_stages.k2),
        (&d_stages.k2, &d_stages.k3),
        (&d_stages.k1, &d_stages.k3),
    ] {
        println!("  {}", dust_max_diff(a, b));
    }
    assert!(
        dg > 1e-14,
        "staged and frozen geometry agree although sources should have been rebuilt ({dg})"
    );
    assert!(
        dm > 1e-14,
        "staged and frozen dust agree although the metric gains gradients ({dm})"
    );
    assert!(
        dust_max_diff(&d_stages.k1, &d_stages.k2) > 0.0,
        "the per-stage dust rates do not change: gradient coupling not exercised"
    );

    // control (ADM twin of the BSSN bit-exact refactor proof): the
    // collapsing stepper returns exactly the collapse of the stages it
    // exposes, so the whole difference above is the rebuild, not a
    // divergent path
    let collapsed = stages.collapse(&config, &values);
    let collapsed_dust = d_stages.collapse();
    assert_eq!(adm_max_diff(&staged, &collapsed), 0.0);
    assert_eq!(dust_max_diff(&staged_dust, &collapsed_dust), 0.0);

    // and the classic facade must produce the same pair bitwise
    let (classic, classic_dust) = adm.step_rk4_with_dust(&dust, dt, &config, &values).unwrap();
    assert_eq!(adm_max_diff(&staged, &classic), 0.0);
    assert_eq!(dust_max_diff(&staged_dust, &classic_dust), 0.0);
}

// 5. Vacuum limit: zero-density dust is exactly the zero sources; both
// coupled steppers must then reduce to their vacuum steppers, and the
// returned dust must stay identically zero.
#[test]
fn vacuum_dust_reduces_to_the_vacuum_step() {
    let (config, values) = periodic();

    // ADM
    let (g, adm, _) = flrw();
    let (adm_next, dust_next) = adm
        .step_rk4_with_dust(&zero_dust(&g), 0.0625, &config, &values)
        .unwrap();
    let vacuum_next = adm.step_rk4(0.0625, &config, &values);

    assert_eq!(
        adm_max_diff(&adm_next, &vacuum_next),
        0.0,
        "empty-dust ADM step is not the vacuum step"
    );
    assert_eq!(
        dust_max_diff(&dust_next, &zero_dust(&g)),
        0.0,
        "dust grew from nothing"
    );

    // BSSN
    let (bssn, _) = flrw_bssn();
    let (b_next, b_dust_next) = bssn
        .step_rk4_with_dust(&zero_dust(&bssn.grid), 0.03125, &config, &values)
        .unwrap();
    let b_vacuum = bssn.step_rk4(0.03125, &config, &values);

    assert_eq!(
        bssn_max_diff(&b_next, &b_vacuum),
        0.0,
        "empty-dust BSSN step is not the vacuum step"
    );
    assert_eq!(
        dust_max_diff(&b_dust_next, &zero_dust(&bssn.grid)),
        0.0,
        "BSSN dust grew from nothing"
    );
}

// 6. Clock sync (BSSN side, dust present): the staged bundles from
// step_rk4_with_dust_stages drive the shared particles pipeline, the
// particle timestamp is committed to t + dt exactly, and a re-step at the
// old clock is refused before anything moves.
#[test]
fn particles_synchronize_with_the_bssn_dust_stage_bundles() {
    let (config, values) = periodic();
    let (bssn, _) = flrw_bssn();
    let dust = flrw_wave_dust();

    let dt = 0.03125;
    let (stages, _) = bssn
        .step_rk4_with_dust_stages(&dust, dt, &config, &values)
        .unwrap();
    let conns: StageConnections<N, D> = bssn_stage_connections::<D, N>(&stages, &config, &values);

    let start = GeodesicState::<D> {
        pos: [0.7, 0.8, 0.9],
        vel: [0.05, -0.02, 0.03],
    };
    let mut particles = Particles::<D>::new();
    let id = particles.spawn(0.0, start).unwrap();
    particles
        .step_all_with_connections(0.0, &conns)
        .expect("synchronized step at the coupling time");
    let after = particles.get(id).unwrap();
    assert_eq!(after.time, dt, "the particle clock must land on t + dt");
    assert_eq!(after.status, ParticleStatus::Active);
    let after_state = after.state;

    // identical endpoint to the shared stage stepper on the same bundles
    let manual = geodesic_stages_step_rk4(
        &start,
        conns.dt,
        [
            &conns.connections[0],
            &conns.connections[1],
            &conns.connections[2],
            &conns.connections[3],
        ],
    )
    .unwrap();
    assert_eq!(after.state, manual, "pipeline and stepper disagree");

    // the same bundles again from the old clock: refused atomically
    let error = particles
        .step_all_with_connections(0.0, &conns)
        .unwrap_err();
    assert!(
        matches!(error, ParticleError::TimeMismatch { .. }),
        "expected TimeMismatch, got {error:?}"
    );
    let unchanged = particles.get(id).unwrap();
    assert_eq!(unchanged.time, dt, "a refused step must not move the clock");
    assert_eq!(
        unchanged.state, after_state,
        "a refused step must not move the state"
    );
}

// 7. Boundary policy on the dust-coupled bundles: a particle whose step
// reaches outside the interpolation domain stops at its last valid state
// while its neighbor advances; a stopped particle carries no committed
// sample and is exempt from later clock checks.
#[test]
fn boundary_stop_policy_holds_on_dust_coupled_data() {
    let (config, values) = periodic();
    let (bssn, _) = flrw_bssn();
    let dust = flrw_wave_dust();

    let dt = 0.03125;
    let (stages, _) = bssn
        .step_rk4_with_dust_stages(&dust, dt, &config, &values)
        .unwrap();
    let conns = bssn_stage_connections::<D, N>(&stages, &config, &values);

    let mut particles = Particles::<D>::new();
    // stays inside the [0.5, 1.375] box and stamps t + dt
    let _calm = particles
        .spawn(
            0.0,
            GeodesicState::<D> {
                pos: [0.7, 0.8, 0.9],
                vel: [0.05, 0.0, 0.0],
            },
        )
        .unwrap();
    // v = 40: the last stage sample leaves the box
    let rocket_id = particles
        .spawn(
            0.0,
            GeodesicState::<D> {
                pos: [0.7, 0.8, 0.9],
                vel: [40.0, 0.0, 0.0],
            },
        )
        .unwrap();

    particles.step_all_with_connections(0.0, &conns).unwrap();

    let calm = particles.get(0).unwrap();
    let rocket = particles.get(rocket_id).unwrap();
    assert_eq!(calm.status, ParticleStatus::Active);
    assert_eq!(calm.time, dt);
    assert_eq!(calm.history.len(), 2);
    assert_eq!(rocket.status, ParticleStatus::StoppedAtBoundary);
    assert_eq!(
        rocket.time, 0.0,
        "a stopped particle keeps its last timestamp"
    );
    assert_eq!(rocket.state.pos, [0.7, 0.8, 0.9]);
    assert_eq!(rocket.state.vel, [40.0, 0.0, 0.0]);
    assert_eq!(
        rocket.history.len(),
        1,
        "no sample is committed for the aborted step"
    );

    // the stopped particle is now exempt from the clock: the batch can
    // step again and the rocket stays put
    let (stages2, _) = bssn
        .step_rk4_with_dust_stages(&dust, dt, &config, &values)
        .unwrap();
    let conns2: StageConnections<N, D> = bssn_stage_connections::<D, N>(&stages2, &config, &values);
    particles.step_all_with_connections(dt, &conns2).unwrap();
    let rocket = particles.get(rocket_id).unwrap();
    assert_eq!(rocket.time, 0.0, "stopped particles must not multiply-step");
    assert_eq!(particles.get(0).unwrap().time, 2.0 * dt);
}

// 8. FLRW manufactured references: (a) ADM -- ten coupled steps conserve
// the dust mass bitwise, keep the slice value-uniform, and hold the
// constraints; (b) BSSN -- ten steps driven through the exposed stages
// collapse bitwise-equal to the classic stepper (the stage path is
// production-grade, not just single-step compatible).
#[test]
fn flrw_coupled_runs_hold_conservation_and_constraints() {
    let (config, values) = periodic();
    let dt = 0.5 * 0.125;

    // (a) ADM
    let (_, mut adm, mut dust) = flrw();
    let total0 = dust_total(&dust);
    for _ in 0..10 {
        let (n, nm) = adm.step_rk4_with_dust(&dust, dt, &config, &values).unwrap();
        adm = n;
        dust = nm;
    }
    assert_eq!(dust_total(&dust), total0, "ADM coupled run lost dust mass");

    // value-uniformity of every evolved field
    let mut kmin = f64::INFINITY;
    let mut kmax = f64::NEG_INFINITY;
    let mut gmin = f64::INFINITY;
    let mut gmax = f64::NEG_INFINITY;
    let mut lmin = f64::INFINITY;
    let mut lmax = f64::NEG_INFINITY;
    adm.grid.for_each_index(|idx| {
        let gxx = adm.gamma.get(idx)[(0, 0)];
        let kxx = adm.k.get(idx)[(0, 0)];
        let alpha = *adm.lapse.get(idx);
        gmin = gmin.min(gxx);
        gmax = gmax.max(gxx);
        kmin = kmin.min(kxx);
        kmax = kmax.max(kxx);
        lmin = lmin.min(alpha);
        lmax = lmax.max(alpha);
    });
    assert_eq!(gmax - gmin, 0.0, "metric left the uniform slice");
    assert_eq!(kmax - kmin, 0.0, "trace left the uniform slice");
    assert_eq!(lmax - lmin, 0.0, "lapse left the uniform slice");
    assert!(gmin.is_finite() && kmin.is_finite() && lmin.is_finite());

    let (hf, mf) = adm_constraints(&adm, &dust, &config, &values);
    println!("ADM FLRW 10 steps: |H| = {hf:.3e}, |M| = {mf:.3e}");
    assert!(hf < 1e-9, "ADM coupled run drifted |H| to {hf}");
    assert!(mf < 1e-12, "ADM coupled run drifted |M| to {mf}");

    // (b) BSSN through the stage path, ten steps, exact class equality
    let mut classic = flrw_bssn();
    let mut via_stages = flrw_bssn();
    for _ in 0..10 {
        let (cn, cm) = classic
            .0
            .step_rk4_with_dust(&classic.1, dt, &config, &values)
            .unwrap();
        classic = (cn, cm);

        let (st, ds) = via_stages
            .0
            .step_rk4_with_dust_stages(&via_stages.1, dt, &config, &values)
            .unwrap();
        via_stages = (st.collapse(&config, &values), ds.collapse());
    }
    assert_eq!(
        bssn_max_diff(&classic.0, &via_stages.0) + dust_max_diff(&classic.1, &via_stages.1),
        0.0,
        "ten-step stage-driven BSSN run differs from the classic run"
    );
}

// 9. M3 regression pin: the frozen-geometry dust stepper, which the
// coupled steppers reuse underneath, still behaves on the FLRW patch
// (mass conserved bitwise, momentum stays exactly zero, no finiteness
// surprises over several steps).
#[test]
fn frozen_geometry_dust_transport_is_unchanged() {
    let (config, values) = periodic();
    let (_, adm, dust) = flrw();
    let total0 = dust_total(&dust);

    let mut state = dust;
    for _ in 0..4 {
        state = dust_step_rk4(
            &state, 0.03125, &adm.lapse, &adm.shift, &adm.gamma, &config, &values,
        );
    }

    assert_eq!(dust_total(&state), total0, "frozen transport leaked mass");
    state.grid.for_each_index(|idx| {
        let p = *state.densitized_momentum.get(idx);
        for i in 0..D {
            assert_eq!(p[i], 0.0, "rest dust gained momentum [{i}]");
        }
        let d = *state.densitized_rest_mass.get(idx);
        assert!(
            d.is_finite() && d > 0.0,
            "densitized mass left the positive reals"
        );
    });
}
