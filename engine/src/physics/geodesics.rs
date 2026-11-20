//! Geodesic test particles integrated in coordinate time.
//!
//! State: spatial position `pos[i] = x^i` and coordinate velocity
//! `vel[i] = dx^i/dt`, evolved by
//!
//! ```text
//! dx^i/dt = v^i,
//! dv^i/dt = -Gamma^i_{mu nu} w^mu w^nu + v^i Gamma^0_{mu nu} w^mu w^nu,
//! ```
//!
//! with `w^0 = 1`, `w^i = v^i` and spacetime indices `mu, nu = 0..N` where
//! index 0 is time (`Christoffel.data[a][b][c]` = Gamma^a_bc). This is the
//! coordinate-time form of the geodesic equation, not the affine-parameter
//! form: the `v^i Gamma^0` term accounts for `dt/dlambda` varying along the
//! curve. Uses the project's -+++ signature throughout.
//!
//! Time-dependent backgrounds are supported through `StageConnections`
//! and `geodesic_stages_step_rk4`: an integrator hands over its RK4 stage
//! data -- `AdmRk4Stages` from the ADM formulation, or `BssnRk4Stages`
//! from the BSSN one (physical metric and time derivatives re-derived
//! from the conformal variables by [`bssn_stage_connections`]) -- and
//! each connection bundle (y_i, k_i) pair carries the true d_t g_ab of
//! that stage (gauge contributions included), while a particle substage
//! at stage time tau samples the geometry the integrator really stepped
//! through at tau instead of a frozen slice. The static shortcut
//! `geodesic_step_rk4` keeps resampling one frozen field -- it is the
//! all-four-stages-equal case of the stage stepper. Both formulations
//! feed the same bundle builder, the same stage stepper and the same
//! particle batch-commit; only the geometry source differs.
//!
//! The state holds plain arrays, never grid fields: particles live between
//! nodes and the ODE layer stays independent of storage, grids, and the
//! ADM/BSSN integrators (no generic integrator exists to reuse; each
//! formulation carries its own RK4 loop).

use crate::geometry::christoffel::{
    christoffel_symbols_with_time_derivatives, Christoffel, ChristoffelField,
};
use crate::grid::boundary::{BoundaryConfig, BoundaryValues};
use crate::numerics::interpolation::{interpolate_christoffel, InterpolationError};
use crate::numerics::rk4::AdmRk4Stages;
use crate::relativity::bssn::{stage_physical_adm, BssnRk4Stages};

/// Test-particle state: position x^i and coordinate velocity v^i = dx^i/dt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeodesicState<const D: usize> {
    pub pos: [f64; D],
    pub vel: [f64; D],
}

/// Right-hand side of the coordinate-time geodesic equation for one
/// connection table: returns (v^i, a^i) with `a` the `dv^i/dt` above.
pub fn geodesic_rhs<const D: usize, const N: usize>(
    state: &GeodesicState<D>,
    gamma: &Christoffel<N>,
) -> GeodesicState<D> {
    assert!(
        N == D + 1,
        "spacetime dim N must be D + 1, got N = {N}, D = {D}"
    );

    // w^mu = dx^mu/dt = (1, v^i) it is the velocity in spacetime coordinates, the first component is 1 because we are using coordinate time as the parameter.
    let mut w = [0.0f64; N];
    w[0] = 1.0;
    for i in 0..D {
        w[1 + i] = state.vel[i];
    }
    // the acceleration vector
    let mut acc = [0.0f64; D];
    for i in 0..D {
        let mut a = 0.0;
        let mut g0 = 0.0;
        for mu in 0..N {
            for nu in 0..N {
                a += gamma.data[1 + i][mu][nu] * w[mu] * w[nu];
                g0 += gamma.data[0][mu][nu] * w[mu] * w[nu];
            }
        }
        acc[i] = -a + state.vel[i] * g0;
    }

    GeodesicState {
        pos: state.vel,
        vel: acc,
    }
}

fn geodesic_axpy<const D: usize>(
    state: &GeodesicState<D>,
    a: f64,
    rhs: &GeodesicState<D>,
) -> GeodesicState<D> {
    let mut pos = [0.0f64; D];
    let mut vel = [0.0f64; D];
    for i in 0..D {
        pos[i] = state.pos[i] + a * rhs.pos[i];
        vel[i] = state.vel[i] + a * rhs.vel[i];
    }
    GeodesicState { pos, vel }
}

/// One classical RK4 step in coordinate time.
///
/// The connection is re-interpolated at each of the four substep positions;
/// an out-of-domain substage aborts the step with the interpolation error
/// instead of extrapolating. This is the static-background shortcut:
/// every stage samples the same frozen field. For a genuinely evolving
/// slice use [`geodesic_stages_step_rk4`] with `StageConnections`.
pub fn geodesic_step_rk4<const D: usize, const N: usize>(
    state: &GeodesicState<D>,
    dt: f64,
    gamma: &ChristoffelField<N, D>,
) -> Result<GeodesicState<D>, InterpolationError> {
    geodesic_stages_step_rk4(state, dt, [gamma, gamma, gamma, gamma])
}

/// One classical RK4 step in coordinate time where stage i samples its
/// own connection bundle. The bundles are ordered by ADM stage time:
/// index 0 belongs to stage time 0, indices 1 and 2 to dt/2 (the two
/// RK4 half stages), index 3 to dt. With all four entries pointing at
/// the same static field this reduces exactly to
/// [`geodesic_step_rk4`]; with per-stage bundles from
/// [`stage_connections`] the particle sees the geometry as it evolves
/// within one ADM step. An out-of-domain substage aborts the step with
/// the interpolation error instead of extrapolating.
pub fn geodesic_stages_step_rk4<const D: usize, const N: usize>(
    state: &GeodesicState<D>,
    dt: f64,
    connections: [&ChristoffelField<N, D>; 4],
) -> Result<GeodesicState<D>, InterpolationError> {
    assert!(
        N == D + 1,
        "spacetime dim N must be D + 1, got N = {N}, D = {D}"
    );
    assert!(dt.is_finite() && dt > 0.0);

    let k1 = geodesic_rhs(state, &interpolate_christoffel(connections[0], state.pos)?);
    let y2 = geodesic_axpy(state, 0.5 * dt, &k1);
    let k2 = geodesic_rhs(&y2, &interpolate_christoffel(connections[1], y2.pos)?);
    let y3 = geodesic_axpy(state, 0.5 * dt, &k2);
    let k3 = geodesic_rhs(&y3, &interpolate_christoffel(connections[2], y3.pos)?);
    let y4 = geodesic_axpy(state, dt, &k3);
    let k4 = geodesic_rhs(&y4, &interpolate_christoffel(connections[3], y4.pos)?);

    let mut next = *state;
    for i in 0..D {
        next.pos[i] += dt / 6.0 * k1.pos[i]
            + dt / 3.0 * k2.pos[i]
            + dt / 3.0 * k3.pos[i]
            + dt / 6.0 * k4.pos[i];
        next.vel[i] += dt / 6.0 * k1.vel[i]
            + dt / 3.0 * k2.vel[i]
            + dt / 3.0 * k3.vel[i]
            + dt / 6.0 * k4.vel[i];
    }
    Ok(next)
}

/// The four connection bundles that couple one ADM RK4 step to the
/// geodesic integrator: connection i is built from the ADM stage pair
/// (y0, k1), (y2, k2), (y3, k3), (y4, k4) with the full
/// `christoffel_symbols_with_time_derivatives` treatment, so every
/// bundle carries that stage's d_t g_ab. Coupled through
/// `geodesic_stages_step_rk4` (and `Particles::step_all_with_connections`),
/// the geodesic phase of each substage is measured against the ADM
/// stage state that RK4 stepped through at the same stage time, which
/// is what makes the coupling consistent to the order of the scheme.
pub struct StageConnections<const N: usize, const D: usize> {
    /// the ADM step size; every particle steps by this same dt
    pub dt: f64,
    /// one bundle per ADM stage, in stage order: 0, dt/2, dt/2, dt
    pub connections: [ChristoffelField<N, D>; 4],
}

/// Build the four time derivative-aware connection bundles from one
/// ADM RK4 step. RK4 computes the rate ki as the right-hand side of the
/// state it is applied to: k1 == rhs(y0), k2 == rhs(y2), k3 == rhs(y3)
/// and k4 == rhs(y4), and `AdmRk4Stages` stores exactly those pairs, so
/// the (yi, ki) pairs here are consistent with what RK4 evaluates.
///
/// All grids are shared with the ADM state by construction. The stage dt
/// is validated here (`assert`) and again by every consumer helper
/// (`geodesic_stages_step_rk4`, `Particles::step_all_with_connections`).
pub fn stage_connections<const D: usize, const N: usize>(
    stages: &AdmRk4Stages<D>,
    boundary: &BoundaryConfig<D>,
    boundary_values: &BoundaryValues<D>,
) -> StageConnections<N, D> {
    assert!(
        N == D + 1,
        "spacetime dim N must be D + 1, got N = {N}, D = {D}"
    );
    assert!(
        stages.dt.is_finite() && stages.dt > 0.0,
        "stage dt must be finite and positive"
    );

    let first = christoffel_symbols_with_time_derivatives::<D, N>(
        &stages.y0.lapse,
        &stages.y0.shift,
        &stages.y0.gamma,
        &stages.k1.gamma,
        &stages.k1.shift,
        &stages.k1.lapse,
        boundary,
        boundary_values,
    );
    let second = christoffel_symbols_with_time_derivatives::<D, N>(
        &stages.y2.lapse,
        &stages.y2.shift,
        &stages.y2.gamma,
        &stages.k2.gamma,
        &stages.k2.shift,
        &stages.k2.lapse,
        boundary,
        boundary_values,
    );
    let third = christoffel_symbols_with_time_derivatives::<D, N>(
        &stages.y3.lapse,
        &stages.y3.shift,
        &stages.y3.gamma,
        &stages.k3.gamma,
        &stages.k3.shift,
        &stages.k3.lapse,
        boundary,
        boundary_values,
    );
    let fourth = christoffel_symbols_with_time_derivatives::<D, N>(
        &stages.y4.lapse,
        &stages.y4.shift,
        &stages.y4.gamma,
        &stages.k4.gamma,
        &stages.k4.shift,
        &stages.k4.lapse,
        boundary,
        boundary_values,
    );

    StageConnections {
        dt: stages.dt,
        connections: [first, second, third, fourth],
    }
}

/// The BSSN counterpart of [`stage_connections`]: the same four connection
/// bundles, built from the same time-aware constructor, but with the
/// per-stage geometry re-derived from the BSSN stage pairs through
/// `relativity::bssn::stage_physical_adm` -- the physical metric
/// gamma_ij = exp(4 phi) gtilde_ij of each stage state, its product-rule
/// time derivative e^{4 phi}(4 d_t phi gtilde_ij + d_t gtilde_ij) with
/// `d_t phi` and `d_t gtilde_ij` taken from the BSSN right-hand side of
/// that stage, and the BSSN gauge rates. Nothing below the bundle level
/// differs between the two formulations: the returned bundles feed the
/// same `geodesic_stages_step_rk4` and the same
/// `Particles::step_all_with_connections`. Both orderings here preserve
/// [`stage_connections`]'s assertion discipline (`N == D + 1`, dt finite
/// and positive).
pub fn bssn_stage_connections<const D: usize, const N: usize>(
    stages: &BssnRk4Stages<D>,
    boundary: &BoundaryConfig<D>,
    boundary_values: &BoundaryValues<D>,
) -> StageConnections<N, D> {
    assert!(
        N == D + 1,
        "spacetime dim N must be D + 1, got N = {N}, D = {D}"
    );
    assert!(
        stages.dt.is_finite() && stages.dt > 0.0,
        "stage dt must be finite and positive"
    );

    let pairs = [
        (&stages.y0, &stages.k1),
        (&stages.y2, &stages.k2),
        (&stages.y3, &stages.k3),
        (&stages.y4, &stages.k4),
    ];

    let bundles = pairs.map(|(y, k)| {
        let (adm, dgamma_dt, dshift_dt, dlapse_dt) = stage_physical_adm(y, k);
        christoffel_symbols_with_time_derivatives::<D, N>(
            &adm.lapse,
            &adm.shift,
            &adm.gamma,
            &dgamma_dt,
            &dshift_dt,
            &dlapse_dt,
            boundary,
            boundary_values,
        )
    });

    StageConnections {
        dt: stages.dt,
        connections: bundles,
    }
}
