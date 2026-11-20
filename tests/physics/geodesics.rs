use einstein_engine::geometry::christoffel::{christoffel_symbols, Christoffel};
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::initial_data::schwarzschild::schwarzschild;
use einstein_engine::numerics::interpolation::interpolate_christoffel;
use einstein_engine::physics::geodesics::{geodesic_rhs, geodesic_step_rk4, GeodesicState};
use std::sync::Arc;

const D: usize = 3;
const N: usize = 4;

fn periodic() -> (BoundaryConfig<D>, BoundaryValues<D>) {
    (
        BoundaryConfig::<D>::periodic(),
        BoundaryValues::<D>::zeros(),
    )
}

fn minkowski_gamma(
    grid: &Arc<Grid<D>>,
) -> einstein_engine::geometry::christoffel::ChristoffelField<N, D> {
    let mut identity = Tensor2::<D>::zero();
    for i in 0..D {
        identity[(i, i)] = 1.0;
    }
    let gamma = Field::new(grid.clone(), identity);
    let lapse = Field::new(grid.clone(), 1.0);
    let shift = Field::new(grid.clone(), Vector::<D>::zero());
    let (config, values) = periodic();
    christoffel_symbols::<D, N>(&lapse, &shift, &gamma, &config, &values)
}

// Minkowski: all Christoffel symbols vanish, so v stays constant and x(t)
// is exactly linear. Tolerance is pure floating-point accumulation.
#[test]
fn minkowski_straight_line() {
    let g = Arc::new(Grid::new([8; D], [0.25; D], [0.0; D]));
    let gamma = minkowski_gamma(&g);

    let mut state = GeodesicState {
        pos: [0.9, 1.0, 1.1],
        vel: [0.2, -0.1, 0.05],
    };
    let dt = 0.05;
    for _ in 0..20 {
        state = geodesic_step_rk4::<D, N>(&state, dt, &gamma).unwrap();
    }

    let t = 20.0 * dt;
    assert!((state.pos[0] - (0.9 + 0.2 * t)).abs() < 1e-12);
    assert!((state.pos[1] - (1.0 - 0.1 * t)).abs() < 1e-12);
    assert!((state.pos[2] - (1.1 + 0.05 * t)).abs() < 1e-12);
    assert!((state.vel[0] - 0.2).abs() < 1e-12);
    assert!((state.vel[1] + 0.1).abs() < 1e-12);
    assert!((state.vel[2] - 0.05).abs() < 1e-12);
}

// The coordinate-time RHS, both terms: with only Gamma^0_00 = c and
// Gamma^x_00 = d nonzero, dv^x/dt must equal -d + v^x * c. The Gamma^0
// term is exercised (c != 0); setting c = 0 would hide a sign error there.
#[test]
fn rhs_uses_both_gamma_terms() {
    let mut gamma = Christoffel::<N>::zero();
    let (c, d) = (0.5, 1.5);
    gamma.data[0][0][0] = c;
    gamma.data[1][0][0] = d;

    let state = GeodesicState {
        pos: [0.0; D],
        vel: [0.4, 0.0, 0.0],
    };
    let r = geodesic_rhs::<D, N>(&state, &gamma);
    // dx/dt = v
    assert_eq!(r.pos, state.vel);
    // dv^x/dt = -d + v^x * c; transverse components stay zero
    assert!((r.vel[0] - (-d + 0.4 * c)).abs() < 1e-15);
    assert_eq!(r.vel[1], 0.0);
    assert_eq!(r.vel[2], 0.0);
}

// Pure-spatial connection term: with only Gamma^x_xx = e, dv^x/dt = -e v^2.
#[test]
fn rhs_spatial_term_quadratic_in_velocity() {
    let mut gamma = Christoffel::<N>::zero();
    gamma.data[1][1][1] = 2.0;

    let state = GeodesicState {
        pos: [0.0; D],
        vel: [0.3, 0.0, 0.0],
    };
    let r = geodesic_rhs::<D, N>(&state, &gamma);
    assert!((r.vel[0] + 2.0 * 0.3 * 0.3).abs() < 1e-15);
}

// Manufactured constant-connection ODE: Gamma^0_00 = c, all else zero.
// Exact solution v(t) = v0 e^{ct}, x(t) = x0 + v0 (e^{ct} - 1)/c.
// Constant data interpolates exactly, so this isolates integrator error.
#[test]
fn rk4_converges_fourth_order() {
    let g = Arc::new(Grid::new([8; D], [0.25; D], [0.0; D]));
    let mut conn = Christoffel::<N>::zero();
    conn.data[0][0][0] = 0.5;
    let gamma = Field::new(g, conn);

    fn final_x(
        dt: f64,
        steps: usize,
        gamma: &einstein_engine::geometry::christoffel::ChristoffelField<N, D>,
    ) -> f64 {
        let mut s = GeodesicState {
            pos: [1.0, 1.0, 1.0],
            vel: [0.1, 0.0, 0.0],
        };
        for _ in 0..steps {
            s = geodesic_step_rk4::<D, N>(&s, dt, gamma).unwrap();
        }
        s.pos[0]
    }

    // exact: x(1) = 1 + 0.1 (e^0.5 - 1)/0.5
    let exact = 1.0 + 0.1 * (0.5f64.exp() - 1.0) / 0.5;
    let e1 = (final_x(0.1, 10, &gamma) - exact).abs();
    let e2 = (final_x(0.05, 20, &gamma) - exact).abs();
    let e3 = (final_x(0.025, 40, &gamma) - exact).abs();
    let r1 = e1 / e2;
    let r2 = e2 / e3;
    assert!(
        (8.0..32.0).contains(&r1),
        "expected ~16x first halving, got {r1} ({e1:.3e} -> {e2:.3e})"
    );
    assert!(
        (8.0..32.0).contains(&r2),
        "expected ~16x second halving, got {r2} ({e2:.3e} -> {e3:.3e})"
    );
}

// Timelike normalization g(u,u) = -1 is preserved along a Minkowski
// trajectory without any renormalization step in the integrator.
#[test]
fn minkowski_norm_preserved() {
    let g = Arc::new(Grid::new([16; D], [0.25; D], [0.0; D]));
    let gamma = minkowski_gamma(&g);

    let v0 = [0.6, 0.0, 0.0];
    let w = 1.0 / (1.0 - 0.36f64).sqrt();
    let mut state = GeodesicState {
        pos: [0.8, 1.0, 1.0],
        vel: v0,
    };
    for _ in 0..40 {
        state = geodesic_step_rk4::<D, N>(&state, 0.05, &gamma).unwrap();
        // u^mu = W (1, v^i) on unit lapse, zero shift, flat space
        let norm = -w * w + w * w * (state.vel[0] * state.vel[0]);
        assert!((norm + 1.0).abs() < 1e-12, "norm drifted to {norm}");
    }
}

// Out-of-domain start, mid-box NaN, and infinite coordinates all return
// Err instead of panicking.
#[test]
fn outside_domain_is_an_error() {
    let g = Arc::new(Grid::new([8; D], [0.25; D], [0.0; D]));
    let gamma = minkowski_gamma(&g);

    let outside = GeodesicState {
        pos: [5.0, 1.0, 1.0],
        vel: [0.1, 0.0, 0.0],
    };
    assert!(geodesic_step_rk4::<D, N>(&outside, 0.01, &gamma).is_err());

    let nan = GeodesicState {
        pos: [f64::NAN, 1.0, 1.0],
        vel: [0.1, 0.0, 0.0],
    };
    assert!(geodesic_step_rk4::<D, N>(&nan, 0.01, &gamma).is_err());
}

// Schwarzschild circular orbit in the repo's isotropic Cartesian coordinates.
// Hole of mass M = 1 at the grid center; the orbit radius r = 10 is given
// in isotropic coordinates, so the areal radius is R = r (1 + M/2r)^2 and
// the Kepler rate is Omega = sqrt(M / R^3). The test checks:
// (a) the circular-orbit condition (zero radial acceleration),
// (b) the orbit stays circular over an arc,
// (c) the conserved energy E = -u_t stays constant,
// (d) the timelike norm stays -1.
#[test]
fn schwarzschild_circular_orbit() {
    let mass = 1.0;
    let n = 24;
    let h = 1.0;
    let half = (n as f64 - 1.0) * h / 2.0;
    let g = Arc::new(Grid::new([n; D], [h; D], [-half, -half, -half]));
    let adm = schwarzschild::<D>(g.clone(), mass).expect("valid data");
    let (config, values) = periodic();
    let gamma = christoffel_symbols::<D, N>(&adm.lapse, &adm.shift, &adm.gamma, &config, &values);

    // isotropic r = 10 -> areal R = r (1 + M/2r)^2
    let r0 = 10.0;
    let psi = 1.0 + mass / (2.0 * r0);
    let r_areal = r0 * psi * psi;
    let omega = (mass / (r_areal * r_areal * r_areal)).sqrt();

    let mut state = GeodesicState {
        pos: [r0, 0.0, 0.0],
        vel: [0.0, r0 * omega, 0.0],
    };

    // (a) circular-orbit condition: radial acceleration equals the
    // centripetal value -v^2/r (not zero -- circular motion accelerates
    // inward). This validates the connection + RHS against Kepler's law.
    let rhs = geodesic_rhs::<D, N>(&state, &interpolate_christoffel(&gamma, state.pos).unwrap());
    let v2 = omega * omega * r0 * r0;
    let want_ar = -v2 / r0;
    println!(
        "circular-orbit radial accel a_r = {:.3e}, want {:.3e}",
        rhs.vel[0], want_ar
    );
    assert!(
        (rhs.vel[0] - want_ar).abs() < 1.0e-3,
        "radial accel should be centripetal: got {}, want {want_ar}",
        rhs.vel[0]
    );

    // helper: conserved energy E = -u_t and norm from the analytic metric
    let energy_norm = |pos: [f64; D], vel: [f64; D]| -> (f64, f64) {
        let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        let p = 1.0 + mass / (2.0 * r);
        let a = (1.0 - mass / (2.0 * r)) / (1.0 + mass / (2.0 * r));
        let psi4 = p * p * p * p;
        let v2 = vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2];
        let s = 1.0 / (a * a - psi4 * v2).sqrt();
        let ut = -a * a * s;
        let norm = -a * a * s * s + psi4 * v2 * s * s;
        (-ut, norm)
    };
    let (e0, n0) = energy_norm(state.pos, state.vel);
    println!("E(0) = {e0:.6}, norm(0) = {n0:.6e}");

    // (b-d) integrate an eighth of an orbit, tracking radius, energy, norm
    let period = 2.0 * std::f64::consts::PI / omega;
    let dt = 0.5;
    let steps = (period / 8.0 / dt).round() as usize;
    let mut max_r_err = 0.0f64;
    let mut max_e_err = 0.0f64;
    let mut max_z = 0.0f64;
    let mut max_n = 0.0f64;
    for _ in 0..steps {
        state = geodesic_step_rk4::<D, N>(&state, dt, &gamma).unwrap();
        let r = (state.pos[0] * state.pos[0]
            + state.pos[1] * state.pos[1]
            + state.pos[2] * state.pos[2])
            .sqrt();
        max_r_err = max_r_err.max((r - r0).abs());
        max_z = max_z.max(state.pos[2].abs());
        let (e, nrm) = energy_norm(state.pos, state.vel);
        max_e_err = max_e_err.max((e - e0).abs());
        max_n = max_n.max((nrm + 1.0).abs());
    }
    println!("radius drift = {max_r_err:.3e}, |z| = {max_z:.3e}, dE = {max_e_err:.3e}, |norm+1| = {max_n:.3e}");
    assert!(max_r_err < 0.05, "orbit left the circle: {max_r_err}");
    assert!(max_e_err < 1.0e-3, "energy drifted: {max_e_err}");
    assert!(max_z < 1.0e-9, "left the plane: {max_z}");
    assert!(max_n < 1.0e-9, "norm drifted: {max_n}");
}
