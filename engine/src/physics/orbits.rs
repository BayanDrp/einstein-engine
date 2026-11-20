//! Schwarzschild circular-orbit fixtures: the analytic side of the orbit
//! validation (initial conditions, conversions, diagnostics).
//!
//! Units `G = c = 1`, hole mass `M`, coordinates `(t, x, y, z)` in the
//! isotropic Cartesian chart of the time-symmetric Schwarzschild slice
//! (`initial_data::schwarzschild`, and the unrestricted variant
//! `isotropic_fields` understood as the same functions):
//!
//! ```text
//! g_00 = -alpha^2,  gamma_ij = psi^4 delta_ij,
//! alpha = (1 - M/(2 rho)) / (1 + M/(2 rho)),  psi = 1 + M/(2 rho),
//! with rho = sqrt(x^2 + y^2) in the orbital plane,
//! and the areal radius r = psi^2 rho.
//! ```
//!
//! # Circular-geodesic initial conditions
//!
//! Work in the orbital plane on the chart `(t, rho, phi)` derived from the
//! Cartesian one. With zero shift, `\dot rho = 0` and `\dot phi = omega`,
//! the geodesic equation in the form the engine integrates,
//! `d^2 x^mu/dt^2 + Gamma^mu_{ab} w^a w^b = 0` with `w = (1, v)` and
//! `d^2 rho/dt^2 = 0` on the orbit, needs the radial balance
//!
//! ```text
//! Gamma^rho_00 + Gamma^rho_phi phi omega^2 = 0.
//! ```
//!
//! The two symbols of the static slice:
//!
//! ```text
//! Gamma^rho_00   = (1/2) g^{rho rho} (-d_rho g_00)
//!                = (1/2) psi^-4 (2 alpha d_rho alpha)
//!                = alpha d_rho alpha / psi^4,
//! with d_rho alpha = M/(rho^2 psi^2)  =>  Gamma^rho_00 = M alpha/(rho^2 psi^6),
//!
//! Gamma^rho_phi phi = (1/2) g^{rho rho} (-d_rho (psi^4 rho^2))
//!                   = -(psi rho - M/rho)/psi = -rho + M/(psi rho).
//! ```
//!
//! The balance gives
//!
//! ```text
//! omega^2 = M alpha/(rho psi^5 (psi rho^2 - M)) = M chi/(rho psi^6 (psi rho^2 - M)),
//! ```
//!
//! with `chi = 1 - M/(2 rho)` and `alpha = chi/psi`. The areal radius is
//! `r = psi^2 rho`, and since `rho^2 (psi - chi) = rho^2 * M/rho = M`,
//!
//! ```text
//! psi rho^2 - M = chi rho^2  =>  omega^2 = M chi/(rho psi^6 chi rho^2)
//!                                        = M/(psi^6 rho^3) = M/r^3.
//! ```
//!
//! So the coordinate angular velocity of *every* circular geodesic --
//! massive or null, in either chart (phi and t are shared coordinates, and
//! a circular areal-radius orbit is a circular isotropic-radius orbit) --
//! is exactly Kepler's third law in coordinate time:
//!
//! ```text
//! omega = sqrt(M / r^3).
//! ```
//!
//! The curvature scalar along the orbit decides its character:
//!
//! ```text
//! g_{mu nu} w^mu w^nu = -alpha^2 + psi^4 rho^2 omega^2
//!                     = -(1 - 2M/r) + M/r = -(1 - 3M/r).
//! ```
//!
//! This vanishes exactly at the photon sphere `r = 3M` (null circular
//! orbit, unstable), is negative for `r > 3M` (timelike), and stability
//! switches at the ISCO `r = 6M`. The proper-time rate along the orbit is
//! `d tau/dt = sqrt(1 - 3M/r)`.
//!
//! # Chart conversion
//!
//! `r = psi^2 rho` with `psi = 1 + M/(2 rho)`. Solving the resulting
//! quadratic in `1/rho` for the exterior branch:
//!
//! ```text
//! rho(r) = (r - M + sqrt(r (r - 2M))) / 2   (monotone increasing, rho -> r as r -> infinity).
//! ```
//!
//! Check: `rho(3M) = M (2 + sqrt(3))/2`, `rho(6M) = M (5 + 2 sqrt(6))/2`.
//! Never substitute an areal radius into an isotropic grid directly.

use crate::geometry::christoffel::Christoffel;
use crate::physics::geodesics::GeodesicState;

/// `(psi^4, alpha)` of the isotropic slice at isotropic radius `rho`.
///
/// Valid wherever `rho > 0`; for `rho <= M/2` these are the smooth
/// continuation values of the interior chart (`alpha` changes sign there)
/// and are only ever used as grid data, never as orbit geometry.
pub fn metric_functions(mass: f64, rho: f64) -> (f64, f64) {
    assert!(rho > 0.0, "isotropic radius must be positive");
    assert!(
        mass.is_finite() && mass > 0.0,
        "mass must be positive and finite"
    );

    let psi = 1.0 + mass / (2.0 * rho);
    let psi4 = psi * psi * psi * psi;
    let alpha = (1.0 - mass / (2.0 * rho)) / (1.0 + mass / (2.0 * rho));
    (psi4, alpha)
}

/// Exterior-branch inverse of the areal-radius map `r = psi^2 rho`.
///
/// Solves `rho (1 + M/(2 rho))^2 = r` for `rho`, taking the branch that
/// tends to `r` as `r -> infinity`, i.e. `rho = (r - M + sqrt(r (r - 2M)))/2`.
/// Requires `r > 2M` (outside the horizon), which every circular orbit
/// satisfies.
pub fn isotropic_radius(mass: f64, areal: f64) -> f64 {
    assert!(
        mass.is_finite() && mass > 0.0,
        "mass must be positive and finite"
    );
    assert!(
        areal.is_finite() && areal > 2.0 * mass,
        "areal radius must exceed 2M"
    );

    (areal - mass + (areal * (areal - 2.0 * mass)).sqrt()) / 2.0
}

/// Areal radius `r = psi^2 rho` of an isotropic radius `rho` (rho > 0).
pub fn areal_radius(mass: f64, iso: f64) -> f64 {
    assert!(iso > 0.0, "isotropic radius must be positive");
    assert!(
        mass.is_finite() && mass > 0.0,
        "mass must be positive and finite"
    );

    let psi = 1.0 + mass / (2.0 * iso);
    iso * psi * psi
}

/// Coordinate angular velocity `omega = sqrt(M / r^3)` of a circular
/// geodesic at areal radius `r` (the derivation above), for any `r > 3M`
/// timelike, any `r >= 3M` null.
pub fn circular_omega(mass: f64, areal: f64) -> f64 {
    assert!(
        mass.is_finite() && mass > 0.0,
        "mass must be positive and finite"
    );
    assert!(
        areal.is_finite() && areal >= 3.0 * mass,
        "circular geodesics exist for r >= 3M (null at exactly 3M, timelike beyond); got r = {areal} M={mass}"
    );

    (mass / (areal * areal * areal)).sqrt()
}

/// Initial state of a circular geodesic at areal radius `r`, in the x-y
/// plane through the hole: position `(rho(r), 0, 0)` and coordinate
/// velocity `(0, rho(r) omega, 0)`, so that `x(t) = rho cos(omega t)`,
/// `y(t) = rho sin(omega t)` is the exact circular solution.
///
/// The Cartesian speed is `|v| = rho omega = r omega / psi^2` (the areal
/// radius, not rho, enters Kepler's law through the radial balance above).
pub fn circular_state<const D: usize>(mass: f64, areal: f64) -> GeodesicState<D> {
    assert!(D >= 2, "need at least two spatial dimensions for an orbit");
    let rho = isotropic_radius(mass, areal);
    let omega = circular_omega(mass, areal);

    let mut pos = [0.0; D];
    pos[0] = rho;
    let mut vel = [0.0; D];
    vel[1] = rho * omega;

    GeodesicState { pos, vel }
}

/// The invariant `g_{mu nu} w^mu w^nu` of a coordinate-time state, computed
/// pointwise from the analytic isotropic metric functions (not from grid
/// interpolation), so it folds the chart, the lapse and the speed together:
/// negative timelike, zero null, positive spacelike.
pub fn normalization<const D: usize>(mass: f64, pos: [f64; D], vel: [f64; D]) -> f64 {
    let iso = (pos.iter().map(|x| x * x).sum::<f64>()).sqrt();
    let (psi4, alpha) = metric_functions(mass, iso);
    let v2: f64 = vel.iter().map(|v| v * v).sum();

    -alpha * alpha + psi4 * v2
}

/// Areal radius of a Cartesian position's isotropic radius.
pub fn areal_of_state<const D: usize>(mass: f64, pos: [f64; D]) -> f64 {
    let iso = (pos.iter().map(|x| x * x).sum::<f64>()).sqrt();
    areal_radius(mass, iso)
}

/// Exact Christoffel symbols `Gamma^a_bc` of the isotropic Schwarzschild
/// slice at one point, with all spacetime rows (`a = 0` included). The
/// slice is static with zero shift, so
///
/// ```text
/// Gamma^0_{0i} = alpha_i / alpha,          (alpha_i = d_i alpha)
/// Gamma^i_{00} = alpha psi^-4 alpha_i,
/// Gamma^i_{jk} = 2/psi (delta^i_j psi_k + delta^i_k psi_j - delta_jk psi_i),
/// Gamma^0_{00} = Gamma^0_{ij} = Gamma^i_{0j} = 0,
/// ```
///
/// with `psi_i = -M x_i/(2 rho^3)` and `alpha_i = M x_i/(rho^3 psi^2)`.
/// Along the derived circular state this reproduces the exact radial
/// balance `Gamma^rho_00 + Gamma^rho_phi phi omega^2 = 0` and the vanishing
/// transverse rows, so it can drive the geodesic right-hand side directly
/// wherever the connection must be exact rather than interpolated
/// (temporal-order studies: spatial representation fixed and exact).
pub fn analytic_christoffel<const N: usize>(mass: f64, pos: [f64; 3]) -> Christoffel<N> {
    assert!(
        N == 4,
        "the 3D isotropic slice has spacetime dim 4, got N = {N}"
    );
    let rho2 = pos.iter().map(|x| x * x).sum::<f64>();
    assert!(rho2 > 0.0, "no finite symbols at the puncture itself");
    let rho = rho2.sqrt();
    let psi = 1.0 + mass / (2.0 * rho);
    let alpha = (1.0 - mass / (2.0 * rho)) / (1.0 + mass / (2.0 * rho));
    let psi4 = psi * psi * psi * psi;

    // directional derivatives of the two scalar fields
    let psi_axis: [f64; 3] = core::array::from_fn(|m| -mass * pos[m] / (2.0 * rho * rho * rho));
    let alpha_axis: [f64; 3] =
        core::array::from_fn(|m| mass * pos[m] / (rho * rho * rho * psi * psi));

    let mut g = Christoffel::<N>::zero();

    for k in 0..3 {
        for i in 0..3 {
            for j in 0..3 {
                let dkj = f64::from(k == j);
                let dki = f64::from(k == i);
                let dij = f64::from(i == j);
                g.data[1 + k][1 + i][1 + j] =
                    2.0 / psi * (dkj * psi_axis[i] + dki * psi_axis[j] - dij * psi_axis[k]);
            }
        }
    }
    for i in 0..3 {
        g.data[1 + i][0][0] = alpha * alpha_axis[i] / psi4;
        g.data[0][0][1 + i] = alpha_axis[i] / alpha;
        g.data[0][1 + i][0] = alpha_axis[i] / alpha;
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    // round trip of the exterior conversion at the three canonical radii
    #[test]
    fn radius_conversion_round_trip() {
        let mass = 1.7;
        for &r in &[3.5 * mass, 3.0 * mass, 6.0 * mass, 12.0 * mass, 50.0 * mass] {
            let rho = isotropic_radius(mass, r);
            assert!(
                (areal_radius(mass, rho) - r).abs() < 1e-12,
                "round trip failed at r = {r}"
            );
        }
        // documented closed bodies
        assert!((isotropic_radius(1.0, 3.0) - (2.0 + 3f64.sqrt()) / 2.0).abs() < 1e-12);
        assert!((isotropic_radius(1.0, 6.0) - (5.0 + 2.0 * 6f64.sqrt()) / 2.0).abs() < 1e-12);
    }

    // the character line g_ww = -(1 - 3M/r): null at 3M, -1/2 at 6M
    #[test]
    fn circular_state_norm_is_null_or_timelike() {
        let mass = 1.0;

        let photon = circular_state::<3>(mass, 3.0 * mass);
        let g_photon = normalization(mass, photon.pos, photon.vel);
        assert!(
            g_photon.abs() < 1e-12,
            "photon-sphere circular state must be exactly null (within fp), got {g_photon}"
        );

        let isco = circular_state::<3>(mass, 6.0 * mass);
        let g_isco = normalization(mass, isco.pos, isco.vel);
        assert!(
            (g_isco + 0.5).abs() < 1e-12,
            "ISCO circular state must have g_ww = -1/2, got {g_isco}"
        );

        // speed consistency: psi^4 rho^2 = r^2 exactly, so psi^4 |v|^2 = M/r
        let rho = isotropic_radius(mass, 6.0 * mass);
        let speed_sq = isco.vel[1] * isco.vel[1];
        let (psi4, _) = metric_functions(mass, rho);
        assert!((psi4 * speed_sq - mass / (6.0 * mass)).abs() < 1e-12);
    }

    // the exact symbols reproduce the Kepler balance at machine precision:
    // a_x == -rho omega^2 and a_y == 0 at both canonical radii
    #[test]
    fn analytic_symbols_satisfy_circular_balance() {
        for &areal in &[3.0, 6.0] {
            let mass = 1.0;
            let state = circular_state::<3>(mass, areal);
            let rho = state.pos[0];
            let omega = circular_omega(mass, areal);

            let g = analytic_christoffel::<4>(mass, state.pos);
            let rhs = crate::physics::geodesics::geodesic_rhs::<3, 4>(&state, &g);

            let ax_exact = -rho * omega * omega;
            assert!(
                (rhs.vel[0] - ax_exact).abs() < 1e-12 * ax_exact.abs(),
                "exact-symbol Kepler balance off: {ax_exact} vs {} at r = {areal}",
                rhs.vel[0]
            );
            assert!(
                rhs.vel[1].abs() < 1e-12 * ax_exact.abs(),
                "exact-symbol transverse row must vanish"
            );
        }
    }
}
