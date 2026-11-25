//! V4 ISCO and photon-sphere validation.
//!

use einstein_engine::geometry::christoffel::christoffel_symbols;
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::Grid;
use einstein_engine::initial_data::schwarzschild::isotropic_fields;
use einstein_engine::physics::orbits::{
    areal_of_state, circular_omega, circular_state, isotropic_radius, normalization,
};
use einstein_engine::physics::particles::{ParticleStatus, Particles};

use std::f64::consts::PI;
use std::sync::Arc;

const D: usize = 3;
const N: usize = 4;
const MASS: f64 = 1.0;
const H: f64 = 0.25;
const DT: f64 = 0.05;
const PHOTON_DT: f64 = 0.01;

pub fn run() -> Result<(), String> {
    let areal = 6.0 * MASS;
    let gamma = connection(areal)?;
    let omega = circular_omega(MASS, areal);
    let period = 2.0 * PI / omega;
    let steps = (2.0 * period / DT).round() as usize;

    let initial = circular_state::<D>(MASS, areal);
    let initial_g_ww = normalization(MASS, initial.pos, initial.vel);
    if (initial_g_ww + 0.5).abs() >= 1e-12 {
        return Err(format!(
            "ISCO initial normalization {initial_g_ww:.3e} is not -1/2"
        ));
    }

    let mut particles = Particles::<D>::new();
    let id = particles
        .spawn(0.0, initial)
        .map_err(|error| format!("could not spawn ISCO particle: {error:?}"))?;

    let mut t = 0.0;
    let mut radius_drift_max = 0.0_f64;
    let mut phase_error_max = 0.0_f64;
    let mut normalization_drift_max = 0.0_f64;
    let mut z_max = 0.0_f64;
    let mut radius_errors = Vec::with_capacity(steps);

    for step in 0..steps {
        particles
            .step_all::<N>(DT, &gamma)
            .map_err(|error| format!("ISCO particle step failed: {error:?}"))?;
        t += DT;

        let particle = particles
            .get(id)
            .ok_or_else(|| "ISCO particle disappeared after stepping".to_string())?;
        if particle.status != ParticleStatus::Active {
            return Err(format!("ISCO particle left the grid at step {}", step + 1));
        }

        let pos = particle.state.pos;
        let vel = particle.state.vel;
        let radius_error = (areal_of_state(MASS, pos) - areal).abs();
        radius_errors.push(radius_error);
        radius_drift_max = radius_drift_max.max(radius_error);
        phase_error_max = phase_error_max.max(wrap(pos[1].atan2(pos[0]) - omega * t).abs());
        normalization_drift_max =
            normalization_drift_max.max((normalization(MASS, pos, vel) - initial_g_ww).abs());
        z_max = z_max.max(pos[2].abs());
    }

    let quarter_steps = steps / 4;
    let early_quarter_max = radius_errors[..quarter_steps]
        .iter()
        .copied()
        .fold(0.0, f64::max);
    let late_quarter_max = radius_errors[steps - quarter_steps..]
        .iter()
        .copied()
        .fold(0.0, f64::max);
    if late_quarter_max >= early_quarter_max * 8.0 + 1e-12 {
        return Err(format!(
            "ISCO radius error grew too much: early {early_quarter_max:.3e}, late {late_quarter_max:.3e}"
        ));
    }

    let final_particle = particles
        .get(id)
        .ok_or_else(|| "ISCO particle disappeared at the end of the run".to_string())?;
    let closure = ((final_particle.state.pos[0] - initial.pos[0]).powi(2)
        + (final_particle.state.pos[1] - initial.pos[1]).powi(2))
    .sqrt();

    if radius_drift_max >= 0.012 {
        return Err(format!(
            "ISCO areal-radius drift {radius_drift_max:.3e} exceeds 0.012 M"
        ));
    }
    if phase_error_max >= 0.02 {
        return Err(format!(
            "ISCO phase error {phase_error_max:.3e} exceeds 0.02 rad"
        ));
    }
    if normalization_drift_max >= 1.2e-3 {
        return Err(format!(
            "ISCO normalization drift {normalization_drift_max:.3e} exceeds 1.2e-3"
        ));
    }
    if closure >= 0.07 {
        return Err(format!("ISCO closure error {closure:.3e} exceeds 0.07"));
    }
    if z_max >= 1e-12 {
        return Err(format!(
            "ISCO left the orbital plane: max |z| = {z_max:.3e}"
        ));
    }

    println!(
        "V4 ISCO: radius drift {:.3e}, phase error {:.3e}, normalization drift {:.3e}, closure {:.3e}, early/late quarter drift {:.3e}/{:.3e}",
        radius_drift_max,
        phase_error_max,
        normalization_drift_max,
        closure,
        early_quarter_max,
        late_quarter_max,
    );

    run_photon_sphere()?;

    Ok(())
}

fn run_photon_sphere() -> Result<(), String> {
    let areal = 3.0 * MASS;
    let gamma = connection(areal)?;
    let rho = isotropic_radius(MASS, areal);
    let omega = circular_omega(MASS, areal);
    let period = 2.0 * PI / omega;
    let steps = (0.25 * period / PHOTON_DT).round() as usize;

    let initial = circular_state::<D>(MASS, areal);
    let initial_g_ww = normalization(MASS, initial.pos, initial.vel);
    if initial_g_ww.abs() >= 1e-12 {
        return Err(format!(
            "photon-sphere initial normalization {initial_g_ww:.3e} is not zero"
        ));
    }

    let mut particles = Particles::<D>::new();
    let id = particles
        .spawn(0.0, initial)
        .map_err(|error| format!("could not spawn photon-sphere particle: {error:?}"))?;

    let mut t = 0.0;
    let mut radius_drift_max = 0.0_f64;
    let mut phase_error_max = 0.0_f64;
    let mut normalization_max = 0.0_f64;
    let mut z_max = 0.0_f64;

    for step in 0..steps {
        particles
            .step_all::<N>(PHOTON_DT, &gamma)
            .map_err(|error| format!("photon-sphere particle step failed: {error:?}"))?;
        t += PHOTON_DT;

        let particle = particles
            .get(id)
            .ok_or_else(|| "photon-sphere particle disappeared after stepping".to_string())?;
        if particle.status != ParticleStatus::Active {
            return Err(format!(
                "photon-sphere particle left the grid at step {}",
                step + 1
            ));
        }

        let pos = particle.state.pos;
        let vel = particle.state.vel;
        radius_drift_max = radius_drift_max.max((areal_of_state(MASS, pos) - areal).abs());
        phase_error_max = phase_error_max.max(wrap(pos[1].atan2(pos[0]) - omega * t).abs());
        normalization_max = normalization_max.max(normalization(MASS, pos, vel).abs());
        z_max = z_max.max(pos[2].abs());
    }

    let final_particle = particles
        .get(id)
        .ok_or_else(|| "photon-sphere particle disappeared at the end of the run".to_string())?;
    let final_pos = final_particle.state.pos;
    let endpoint = [rho * (omega * t).cos(), rho * (omega * t).sin(), 0.0];
    let endpoint_error = ((final_pos[0] - endpoint[0]).powi(2)
        + (final_pos[1] - endpoint[1]).powi(2)
        + (final_pos[2] - endpoint[2]).powi(2))
    .sqrt();

    if radius_drift_max >= 0.04 {
        return Err(format!(
            "photon-sphere radius drift {radius_drift_max:.3e} exceeds 0.04 M"
        ));
    }
    if phase_error_max >= 0.02 {
        return Err(format!(
            "photon-sphere phase error {phase_error_max:.3e} exceeds 0.02 rad"
        ));
    }
    if normalization_max >= 0.008 {
        return Err(format!(
            "photon-sphere |g_ww| {normalization_max:.3e} exceeds 0.008"
        ));
    }
    if endpoint_error >= 0.04 {
        return Err(format!(
            "photon-sphere endpoint error {endpoint_error:.3e} exceeds 0.04"
        ));
    }
    if z_max >= 1e-12 {
        return Err(format!(
            "photon-sphere particle left the orbital plane: max |z| = {z_max:.3e}"
        ));
    }

    println!(
        "V4 photon sphere: radius drift {:.3e}, phase error {:.3e}, |g_ww| {:.3e}, endpoint error {:.3e}",
        radius_drift_max, phase_error_max, normalization_max, endpoint_error,
    );

    Ok(())
}

fn connection(
    areal: f64,
) -> Result<einstein_engine::geometry::christoffel::ChristoffelField<N, D>, String> {
    let isotropic = isotropic_radius(MASS, areal);
    let min_half_span = isotropic + 1.4;
    let mut points = (((2.0 * min_half_span) / H).ceil() as usize) + 1;
    if points % 2 == 1 {
        points += 1;
    }
    let origin = -((points as f64 - 1.0) * H) / 2.0;
    let grid = Arc::new(Grid::new([points; D], [H; D], [origin; D]));
    let adm = isotropic_fields(grid, MASS).map_err(str::to_string)?;
    let boundary = BoundaryConfig::<D>::periodic();
    let boundary_values = BoundaryValues::<D>::zeros();

    Ok(christoffel_symbols::<D, N>(
        &adm.lapse,
        &adm.shift,
        &adm.gamma,
        &boundary,
        &boundary_values,
    ))
}

fn wrap(angle: f64) -> f64 {
    ((angle + PI).rem_euclid(2.0 * PI)) - PI
}
