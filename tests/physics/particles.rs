use einstein_engine::geometry::christoffel::{christoffel_symbols, Christoffel, ChristoffelField};
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::initial_data::schwarzschild::schwarzschild;
use einstein_engine::physics::geodesics::GeodesicState;
use einstein_engine::physics::particles::{ParticleError, ParticleStatus, Particles};
use std::sync::Arc;

const D: usize = 3;
const N: usize = 4;

// n nodes per axis, spacing 0.25, origin at 0: domain [0, (n-1)/4].
fn minkowski(n: usize) -> ChristoffelField<N, D> {
    let grid = Arc::new(Grid::new([n; D], [0.25; D], [0.0; D]));
    let mut identity = Tensor2::<D>::zero();
    for i in 0..D {
        identity[(i, i)] = 1.0;
    }
    let gamma = Field::new(grid.clone(), identity);
    let lapse = Field::new(grid.clone(), 1.0);
    let shift = Field::new(grid.clone(), Vector::<D>::zero());
    let (config, values) = (
        BoundaryConfig::<D>::periodic(),
        BoundaryValues::<D>::zeros(),
    );
    christoffel_symbols::<D, N>(&lapse, &shift, &gamma, &config, &values)
}

// Static isotropic Schwarzschild slice, hole of `mass` at the origin.
// 24 nodes per axis at spacing 1.0: domain [-11.5, 11.5]^3.
fn schwarzschild_connection(mass: f64) -> ChristoffelField<N, D> {
    let n = 24;
    let h = 1.0;
    let half = (n as f64 - 1.0) * h / 2.0;
    let grid = Arc::new(Grid::new([n; D], [h; D], [-half; D]));
    let adm = schwarzschild::<D>(grid, mass).expect("valid data");
    let (config, values) = (
        BoundaryConfig::<D>::periodic(),
        BoundaryValues::<D>::zeros(),
    );
    christoffel_symbols::<D, N>(&adm.lapse, &adm.shift, &adm.gamma, &config, &values)
}

// Conserved energy E = -u_t = alpha^2 W from the analytic isotropic metric,
// W = 1 / sqrt(alpha^2 - psi^4 |v|^2) with alpha = (1 - M/2r)/(1 + M/2r)
// and psi^4 = (1 + M/2r)^4, r the isotropic radius.
fn energy(pos: [f64; D], vel: [f64; D], mass: f64) -> f64 {
    let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
    let p = 1.0 + mass / (2.0 * r);
    let a = (1.0 - mass / (2.0 * r)) / p;
    let psi4 = p * p * p * p;
    let v2 = vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2];
    let w = 1.0 / (a * a - psi4 * v2).sqrt();
    a * a * w
}

fn radius(pos: [f64; D]) -> f64 {
    (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt()
}

fn state(pos: [f64; D], vel: [f64; D]) -> GeodesicState<D> {
    GeodesicState { pos, vel }
}

#[test]
fn spawn_assigns_unique_ids_and_seeds_history() {
    let mut particles = Particles::<D>::new();
    let first = particles
        .spawn(0.0, state([1.0; D], [0.1, 0.0, 0.0]))
        .unwrap();
    let second = particles
        .spawn(0.0, state([1.0; D], [0.0, 0.1, 0.0]))
        .unwrap();

    assert_ne!(first, second);
    assert_eq!(particles.len(), 2);
    assert!(!particles.is_empty());
    assert_eq!(particles.iter().count(), 2);

    let particle = particles.get(first).unwrap();
    assert_eq!(particle.id, first);
    assert_eq!(particle.time, 0.0);
    assert_eq!(particle.status, ParticleStatus::Active);
    assert_eq!(particle.history.len(), 1);
    assert_eq!(particle.history[0].t, 0.0);
    assert_eq!(particle.history[0].pos, [1.0; D]);
    assert_eq!(particle.history[0].vel, [0.1, 0.0, 0.0]);

    assert!(particles.get(99).is_none());
}

#[test]
fn step_all_advances_every_active_particle() {
    let gamma = minkowski(8);
    let mut particles = Particles::<D>::new();
    let a = particles
        .spawn(0.0, state([1.0; D], [0.2, 0.0, 0.0]))
        .unwrap();
    let b = particles
        .spawn(0.0, state([1.0; D], [0.0, 0.1, 0.0]))
        .unwrap();

    let dt = 0.05;
    for _ in 0..5 {
        particles.step_all(dt, &gamma).unwrap();
    }

    // Minkowski: straight lines are reproduced exactly
    let a = particles.get(a).unwrap();
    assert!((a.time - 0.25).abs() < 1e-15);
    assert_eq!(a.history.len(), 6);
    assert!((a.history.last().unwrap().t - 0.25).abs() < 1e-15);
    let expected = [1.0 + 0.2 * 0.25, 1.0, 1.0];
    for i in 0..D {
        assert!((a.state.pos[i] - expected[i]).abs() < 1e-12);
    }
    assert!((a.state.vel[0] - 0.2).abs() < 1e-12);
    assert!(a.state.vel[1].abs() < 1e-12);
    assert!(a.state.vel[2].abs() < 1e-12);

    let b = particles.get(b).unwrap();
    assert!((b.time - 0.25).abs() < 1e-15);
    assert!((b.state.pos[0] - 1.0).abs() < 1e-12);
    assert!((b.state.pos[1] - 1.025).abs() < 1e-12);
    assert!((b.state.vel[1] - 0.1).abs() < 1e-12);
}

#[test]
fn step_all_rejects_invalid_time_steps_without_mutating() {
    let gamma = minkowski(8);
    let mut particles = Particles::<D>::new();
    let id = particles
        .spawn(0.0, state([1.0; D], [0.1, 0.0, 0.0]))
        .unwrap();

    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            particles.step_all(bad, &gamma),
            Err(ParticleError::InvalidTimeStep)
        ));
    }

    let particle = particles.get(id).unwrap();
    assert_eq!(particle.time, 0.0);
    assert_eq!(particle.history.len(), 1);
}

#[test]
fn spawn_rejects_non_finite_time_or_state() {
    let mut particles = Particles::<D>::new();

    assert!(matches!(
        particles.spawn(f64::NAN, state([1.0; D], [0.0; D])),
        Err(ParticleError::NonFiniteTime)
    ));
    assert!(matches!(
        particles.spawn(0.0, state([f64::NAN, 1.0, 1.0], [0.0; D])),
        Err(ParticleError::NonFiniteState)
    ));
    assert!(matches!(
        particles.spawn(0.0, state([1.0; D], [f64::INFINITY, 0.0, 0.0])),
        Err(ParticleError::NonFiniteState)
    ));

    assert!(particles.is_empty());
}

// The fast particle leaves the domain on its second step; it stops at its
// last valid state while the slow particle keeps stepping. Both calls
// return Ok -- an exit is not a batch failure.
#[test]
fn particle_leaving_the_domain_stops_and_others_keep_stepping() {
    let gamma = minkowski(8);
    let mut particles = Particles::<D>::new();
    let fast = particles
        .spawn(0.0, state([1.5, 1.0, 1.0], [2.0, 0.0, 0.0]))
        .unwrap();
    let slow = particles
        .spawn(0.0, state([0.5, 0.5, 0.5], [0.1, 0.0, 0.0]))
        .unwrap();

    for _ in 0..6 {
        particles.step_all(0.1, &gamma).unwrap();
    }

    // step 1 commits (1.5 -> 1.7, inside [0, 1.75]); step 2 would substage
    // at 1.8, so it stops at 1.7 with the 0.1 timestamp
    let fast_particle = particles.get(fast).unwrap();
    assert_eq!(fast_particle.status, ParticleStatus::StoppedAtBoundary);
    assert!((fast_particle.time - 0.1).abs() < 1e-12);
    assert!((fast_particle.state.pos[0] - 1.7).abs() < 1e-12);
    assert_eq!(fast_particle.history.len(), 2);

    let slow_particle = particles.get(slow).unwrap();
    assert_eq!(slow_particle.status, ParticleStatus::Active);
    assert!((slow_particle.time - 0.6).abs() < 1e-12);
    assert_eq!(slow_particle.history.len(), 7);

    let frozen_time = fast_particle.time;
    let frozen_pos = fast_particle.state.pos;
    particles.step_all(0.1, &gamma).unwrap();

    let fast_particle = particles.get(fast).unwrap();
    assert_eq!(fast_particle.status, ParticleStatus::StoppedAtBoundary);
    assert_eq!(fast_particle.time, frozen_time);
    assert_eq!(fast_particle.state.pos, frozen_pos);
    assert_eq!(fast_particle.history.len(), 2);

    let slow_particle = particles.get(slow).unwrap();
    assert!((slow_particle.time - 0.7).abs() < 1e-12);
    assert_eq!(slow_particle.history.len(), 8);
}

// A non-finite timestamp candidate aborts the whole call: the healthy
// particle computed before the bad one is not committed either.
#[test]
fn non_finite_time_aborts_without_committing() {
    let gamma = minkowski(8);
    let mut particles = Particles::<D>::new();
    let rest = particles
        .spawn(0.0, state([1.0; D], [0.0; D]))
        .unwrap();
    let missile = particles
        .spawn(f64::MAX, state([1.0; D], [0.0; D]))
        .unwrap();

    assert!(matches!(
        particles.step_all(f64::MAX, &gamma),
        Err(ParticleError::NonFiniteTime)
    ));

    assert_eq!(particles.get(rest).unwrap().time, 0.0);
    assert_eq!(particles.get(rest).unwrap().history.len(), 1);
    assert_eq!(particles.get(missile).unwrap().time, f64::MAX);
    assert_eq!(particles.get(missile).unwrap().history.len(), 1);
}

// A NaN connection node poisons only the final RK4 stage: the particle
// starts in a clean cell and its y4 substage lands in the poisoned cell, so
// the candidate state comes out non-finite while every sampled position
// stayed in-domain. The far particle's candidate is valid but must not
// commit because the batch aborts first.
//
// (If the NaN is sampled at an earlier stage, the trial *positions*
// themselves go non-finite and the interpolator's OutsideDomain turns the
// step into a boundary stop instead -- conservative either way, nothing
// non-finite is ever committed.)
#[test]
fn non_finite_state_aborts_without_committing() {
    let mut gamma = minkowski(16);
    let mut bad = Christoffel::<N>::zero();
    bad.data[1][0][0] = f64::NAN;
    gamma.set([11, 12, 12], bad);

    let mut particles = Particles::<D>::new();
    let far = particles
        .spawn(0.0, state([0.5, 0.5, 0.5], [0.05, 0.0, 0.0]))
        .unwrap();
    // x: 2.2 -> y2/y3 at 2.4 (cell 9), y4 at 2.6 (cell 10, holds the NaN)
    let doomed = particles
        .spawn(0.0, state([2.2, 3.0, 3.0], [2.0, 0.0, 0.0]))
        .unwrap();

    assert!(matches!(
        particles.step_all(0.2, &gamma),
        Err(ParticleError::NonFiniteState)
    ));

    for id in [far, doomed] {
        let particle = particles.get(id).unwrap();
        assert_eq!(particle.time, 0.0);
        assert_eq!(particle.history.len(), 1);
        assert_eq!(particle.status, ParticleStatus::Active);
    }
}

// Two test particles half an orbit apart on the same circular orbit
// (isotropic r = 10M, Omega = sqrt(M/R_areal^3)) evolve through the
// collection API for an eighth of a period. Both stay circular, keep the
// conserved energy E = -u_t, and remain diametrically opposite by symmetry.
#[test]
fn schwarzschild_circular_orbit_pair_through_particles() {
    let mass = 1.0;
    let gamma = schwarzschild_connection(mass);

    let r0 = 10.0;
    let psi = 1.0 + mass / (2.0 * r0);
    let r_areal = r0 * psi * psi;
    let omega = (mass / (r_areal * r_areal * r_areal)).sqrt();

    let mut particles = Particles::<D>::new();
    let a = particles
        .spawn(0.0, state([r0, 0.0, 0.0], [0.0, r0 * omega, 0.0]))
        .unwrap();
    let b = particles
        .spawn(0.0, state([-r0, 0.0, 0.0], [0.0, -r0 * omega, 0.0]))
        .unwrap();

    let e0 = energy([r0, 0.0, 0.0], [0.0, r0 * omega, 0.0], mass);

    let period = 2.0 * std::f64::consts::PI / omega;
    let dt = 0.5;
    let steps = (period / 8.0 / dt).round() as usize;
    for _ in 0..steps {
        particles.step_all(dt, &gamma).unwrap();
    }

    let pa = particles.get(a).unwrap();
    let pb = particles.get(b).unwrap();
    assert_eq!(pa.status, ParticleStatus::Active);
    assert_eq!(pa.history.len(), steps + 1);
    assert!((pa.time - steps as f64 * dt).abs() < 1e-12);

    let ra = radius(pa.state.pos);
    let rb = radius(pb.state.pos);
    let ra_err = (ra - r0).abs();
    let rb_err = (rb - r0).abs();
    assert!(ra_err < 0.05, "orbit A left the circle: {ra_err}");
    assert!(rb_err < 0.05, "orbit B left the circle: {rb_err}");
    assert!(pa.state.pos[2].abs() < 1e-9);
    assert!(pb.state.pos[2].abs() < 1e-9);

    // exact by symmetry: mirrored initial data mirrors every RK4 stage
    let mut mirror_err = 0.0f64;
    for i in 0..D {
        mirror_err = mirror_err.max((pa.state.pos[i] + pb.state.pos[i]).abs());
    }
    assert!(mirror_err < 1e-12, "pair lost mirror symmetry: {mirror_err}");

    let ea = energy(pa.state.pos, pa.state.vel, mass);
    let eb = energy(pb.state.pos, pb.state.vel, mass);
    assert!((ea - e0).abs() < 1e-3, "dE(A) = {}", (ea - e0).abs());
    assert!((eb - e0).abs() < 1e-3, "dE(B) = {}", (eb - e0).abs());
    println!("circular pair: radius drift ({ra_err:.3e}, {rb_err:.3e}), mirror {mirror_err:.3e}, dE ({:.3e}, {:.3e})", (ea - e0).abs(), (eb - e0).abs());
}

// A particle released from rest at isotropic r = 10M falls monotonically
// inward, speeds up, and keeps its conserved energy. This exercises the
// whole pipeline (Schwarzschild data -> Christoffels -> interpolated
// sampling -> particle stepping) on a genuinely curved, accelerating
// trajectory.
#[test]
fn schwarzschild_radial_infall_from_rest() {
    let mass = 1.0;
    let gamma = schwarzschild_connection(mass);

    let r0 = 10.0;
    let mut particles = Particles::<D>::new();
    let id = particles
        .spawn(0.0, state([r0, 0.0, 0.0], [0.0; D]))
        .unwrap();

    let e0 = energy([r0, 0.0, 0.0], [0.0; D], mass);
    assert!((e0 - (1.0 - mass / (2.0 * r0)) / (1.0 + mass / (2.0 * r0))).abs() < 1e-15);

    let dt = 0.5;
    let mut previous_r = r0;
    let mut previous_speed = 0.0;
    let mut max_e_err = 0.0f64;
    for _ in 0..20 {
        particles.step_all(dt, &gamma).unwrap();
        let p = particles.get(id).unwrap();
        let r = radius(p.state.pos);
        let speed = (p.state.vel[0] * p.state.vel[0]
            + p.state.vel[1] * p.state.vel[1]
            + p.state.vel[2] * p.state.vel[2])
        .sqrt();
        assert!(r < previous_r, "infall stalled: {r} >= {previous_r}");
        assert!(speed > previous_speed, "did not speed up: {speed} <= {previous_speed}");
        max_e_err = max_e_err.max((energy(p.state.pos, p.state.vel, mass) - e0).abs());
        previous_r = r;
        previous_speed = speed;
    }

    assert!(max_e_err < 1e-4, "energy drift {max_e_err}");
    println!(
        "radial infall: r {r0} -> {previous_r:.6}, speed {previous_speed:.6}, dE {max_e_err:.3e}"
    );
    assert!(previous_r < r0 - 0.1, "fell only to {previous_r}");
}
