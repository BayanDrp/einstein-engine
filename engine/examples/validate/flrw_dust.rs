//! V5 FLRW dust: genuinely coupled dust and geometry on a homogeneous patch.
//!
//! The patch is spatially flat and expanding:
//!
//!   gamma_ij = a0^2 delta_ij
//!   K_ij     = -H0 gamma_ij
//!   rho_0    = 3 H0^2 / (8 pi)
//!
//! with a0 = 1, H0 = 0.1, uniform comoving dust, and periodic boundaries.
//! Every stage rebuilds the matter sources from the intermediate dust and
//! geometry, so this is a coupled evolution, not a fixed source response.
//!
//! Pass criteria and tolerances follow VALIDATION.md, V5.

use einstein_engine::geometry::metric::{ExtrinsicCurvature, LapseField, ShiftField, SliceMetric};
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::numerics::rk4::AdmState;
use einstein_engine::physics::dust::{dust_sources, DustState};
use einstein_engine::relativity::bssn::BssnState;
use einstein_engine::validation::constraints::{residuals, ConstraintResidual};
use std::sync::Arc;

const D: usize = 3;
const H: f64 = 0.5;
const A0: f64 = 1.0;
const H0: f64 = 0.1;
const RHO0: f64 = 3.0 * H0 * H0 / (8.0 * std::f64::consts::PI);

/// VALIDATION.md V5: at least ten coupled steps.
const STEPS: usize = 10;
const DT: f64 = 0.0625;

const INITIAL_H_TOL: f64 = 1.0e-12;
const INITIAL_M_TOL: f64 = 1.0e-14;
const FINAL_H_TOL: f64 = 1.0e-9;
const FINAL_M_TOL: f64 = 1.0e-12;

pub fn run() -> Result<(), String> {
    let grid = Arc::new(Grid::new([4, 4, 4], [H; D], [0.0; D]));
    let config = BoundaryConfig::<D>::periodic();
    let values = BoundaryValues::<D>::zeros();

    // Expanding flat FLRW slice: gamma_ij = a0^2 delta_ij and
    // K_ij = -H0 gamma_ij. Positive H0 means the scale factor grows.
    let mut gamma0 = Tensor2::<D>::zero();
    let mut k0 = Tensor2::<D>::zero();
    for i in 0..D {
        gamma0[(i, i)] = A0 * A0;
        k0[(i, i)] = -H0 * A0 * A0;
    }

    let spatial_metric: SliceMetric<D> = SliceMetric::new(grid.clone(), gamma0);
    let extrinsic: ExtrinsicCurvature<D> = ExtrinsicCurvature::new(grid.clone(), k0);
    let lapse: LapseField<D> = LapseField::new(grid.clone(), 1.0);
    let shift: ShiftField<D> = ShiftField::new(grid.clone(), Vector::<D>::zero());

    let adm = AdmState {
        grid: grid.clone(),
        gamma: spatial_metric,
        k: extrinsic,
        lapse,
        shift,
    };

    // Homogeneous comoving dust: v^i = 0, W = 1. The Hamiltonian
    // constraint fixes rho_0 = 3 H0^2 / (8 pi).
    let rho = Field::new(grid.clone(), RHO0);
    let velocity = Field::new(grid.clone(), Vector::<D>::zero());
    let dust = DustState::from_primitives(&rho, &velocity, &adm.gamma);

    let bssn = BssnState::from_adm(&adm, &config, &values);

    // Initial data must start on shell in both representations.
    let initial_adm = constraint_residual(&adm, &dust, &config, &values)?;
    check_residual("initial ADM", &initial_adm, INITIAL_H_TOL, INITIAL_M_TOL)?;

    let initial_bssn_adm = bssn.to_adm();
    let initial_bssn = constraint_residual(&initial_bssn_adm, &dust, &config, &values)?;
    check_residual("initial BSSN", &initial_bssn, INITIAL_H_TOL, INITIAL_M_TOL)?;

    check_adm_uniform("initial ADM", &adm)?;
    check_bssn_uniform("initial BSSN", &bssn)?;
    check_dust_uniform("initial dust", &dust)?;

    let initial_mass = dust_total(&dust);

    // Run the ADM path as an independent cross-check, and two BSSN paths:
    // the classic coupled stepper and the explicit stage collapse.
    let mut adm_evolved = adm.clone();
    let mut adm_dust = dust.clone();

    let mut bssn_classic = bssn.clone();
    let mut dust_classic = dust.clone();

    let mut bssn_staged = bssn;
    let mut dust_staged = dust;

    for step in 1..=STEPS {
        let (next_adm, next_dust) = adm_evolved
            .step_rk4_with_dust(&adm_dust, DT, &config, &values)
            .map_err(|error| format!("ADM coupled step {step} failed: {error:?}"))?;
        adm_evolved = next_adm;
        adm_dust = next_dust;

        let (next_bssn, next_dust) = bssn_classic
            .step_rk4_with_dust(&dust_classic, DT, &config, &values)
            .map_err(|error| format!("BSSN classic step {step} failed: {error:?}"))?;
        bssn_classic = next_bssn;
        dust_classic = next_dust;

        let (bssn_stages, dust_stages) = bssn_staged
            .step_rk4_with_dust_stages(&dust_staged, DT, &config, &values)
            .map_err(|error| format!("BSSN stage step {step} failed: {error:?}"))?;
        bssn_staged = bssn_stages.collapse(&config, &values);
        dust_staged = dust_stages.collapse();
    }

    // Final constraints on the coupled ADM and BSSN paths.
    let final_adm = constraint_residual(&adm_evolved, &adm_dust, &config, &values)?;
    check_residual("final ADM", &final_adm, FINAL_H_TOL, FINAL_M_TOL)?;

    let final_bssn_adm = bssn_classic.to_adm();
    let final_bssn = constraint_residual(&final_bssn_adm, &dust_classic, &config, &values)?;
    check_residual("final BSSN", &final_bssn, FINAL_H_TOL, FINAL_M_TOL)?;

    // Homogeneity must survive exactly.
    check_adm_uniform("final ADM", &adm_evolved)?;
    check_bssn_uniform("final BSSN classic", &bssn_classic)?;
    check_bssn_uniform("final BSSN staged", &bssn_staged)?;
    check_dust_uniform("final ADM dust", &adm_dust)?;
    check_dust_uniform("final BSSN classic dust", &dust_classic)?;
    check_dust_uniform("final BSSN staged dust", &dust_staged)?;

    // The conserved rest mass must not move.
    let mass_adm = dust_total(&adm_dust);
    let mass_classic = dust_total(&dust_classic);
    let mass_staged = dust_total(&dust_staged);
    for (label, mass) in [
        ("ADM", mass_adm),
        ("BSSN classic", mass_classic),
        ("BSSN staged", mass_staged),
    ] {
        if mass != initial_mass {
            return Err(format!(
                "{label}: conserved dust mass changed from {initial_mass:.17e} to {mass:.17e}"
            ));
        }
    }

    // The stage-driven and classic BSSN paths must be bitwise identical.
    let bssn_difference = max_bssn_diff(&bssn_classic, &bssn_staged);
    let dust_difference = max_dust_diff(&dust_classic, &dust_staged);
    if bssn_difference != 0.0 || dust_difference != 0.0 {
        return Err(format!(
            "stage-driven and classic BSSN paths differ: geometry {bssn_difference:.3e}, dust {dust_difference:.3e}"
        ));
    }

    println!(
        "V5 FLRW dust: H {:.3e}/{:.3e}, M {:.3e}/{:.3e}, mass drift 0, stage diff 0",
        initial_adm.hamiltonian_max,
        final_bssn.hamiltonian_max,
        initial_adm.momentum_max,
        final_bssn.momentum_max,
    );

    Ok(())
}

fn constraint_residual(
    adm: &AdmState<D>,
    dust: &DustState<D>,
    config: &BoundaryConfig<D>,
    values: &BoundaryValues<D>,
) -> Result<ConstraintResidual, String> {
    let sources = dust_sources::<4, D>(dust, &adm.lapse, &adm.shift, &adm.gamma)
        .map_err(|error| format!("dust source construction failed: {error:?}"))?;

    Ok(residuals(
        &adm.gamma,
        &adm.k,
        &sources.rho,
        &sources.momentum,
        config,
        values,
    ))
}

fn check_residual(
    label: &str,
    residual: &ConstraintResidual,
    hamiltonian_tol: f64,
    momentum_tol: f64,
) -> Result<(), String> {
    if !residual.hamiltonian_max.is_finite() || !residual.momentum_max.is_finite() {
        return Err(format!("{label}: non finite constraint residual"));
    }

    if residual.hamiltonian_max > hamiltonian_tol || residual.momentum_max > momentum_tol {
        return Err(format!(
            "{label}: H = {:.3e} (tol {:.1e}), M = {:.3e} (tol {:.1e})",
            residual.hamiltonian_max, hamiltonian_tol, residual.momentum_max, momentum_tol,
        ));
    }

    Ok(())
}

fn dust_total(dust: &DustState<D>) -> f64 {
    let cell_volume: f64 = dust.grid.spacing.iter().product();
    let mut total = 0.0f64;

    dust.grid.for_each_index(|idx| {
        total += *dust.densitized_rest_mass.get(idx);
    });

    total * cell_volume
}

fn require_uniform(label: &str, name: &str, range: f64) -> Result<(), String> {
    if range != 0.0 {
        return Err(format!(
            "{label}: {name} is not uniform (max-min = {range:.3e})"
        ));
    }

    Ok(())
}

fn check_adm_uniform(label: &str, adm: &AdmState<D>) -> Result<(), String> {
    require_uniform(label, "gamma", tensor_range(&adm.gamma))?;
    require_uniform(label, "K", tensor_range(&adm.k))?;
    require_uniform(label, "lapse", scalar_range(&adm.lapse))?;
    require_uniform(label, "shift", vector_range(&adm.shift))
}

fn check_bssn_uniform(label: &str, bssn: &BssnState<D>) -> Result<(), String> {
    require_uniform(label, "phi", scalar_range(&bssn.phi))?;
    require_uniform(
        label,
        "conformal_metric",
        tensor_range(&bssn.conformal_metric),
    )?;
    require_uniform(label, "trace_k", scalar_range(&bssn.trace_k))?;
    require_uniform(label, "conformal_a", tensor_range(&bssn.conformal_a))?;
    require_uniform(
        label,
        "conformal_gamma",
        vector_range(&bssn.conformal_gamma),
    )?;
    require_uniform(label, "lapse", scalar_range(&bssn.lapse))?;
    require_uniform(label, "shift", vector_range(&bssn.shift))
}

fn check_dust_uniform(label: &str, dust: &DustState<D>) -> Result<(), String> {
    require_uniform(
        label,
        "densitized_rest_mass",
        scalar_range(&dust.densitized_rest_mass),
    )?;
    require_uniform(
        label,
        "densitized_momentum",
        vector_range(&dust.densitized_momentum),
    )
}

fn scalar_range(field: &Field<f64, D>) -> f64 {
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;

    for &value in &field.data {
        if !value.is_finite() {
            return f64::INFINITY;
        }
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }

    maximum - minimum
}

fn vector_range(field: &Field<Vector<D>, D>) -> f64 {
    let mut worst = 0.0f64;

    for i in 0..D {
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;

        for value in &field.data {
            let component = value[i];
            if !component.is_finite() {
                return f64::INFINITY;
            }
            minimum = minimum.min(component);
            maximum = maximum.max(component);
        }

        worst = worst.max(maximum - minimum);
    }

    worst
}

fn tensor_range(field: &Field<Tensor2<D>, D>) -> f64 {
    let mut worst = 0.0f64;

    for i in 0..D {
        for j in 0..D {
            let mut minimum = f64::INFINITY;
            let mut maximum = f64::NEG_INFINITY;

            for value in &field.data {
                let component = value[(i, j)];
                if !component.is_finite() {
                    return f64::INFINITY;
                }
                minimum = minimum.min(component);
                maximum = maximum.max(component);
            }

            worst = worst.max(maximum - minimum);
        }
    }

    worst
}

fn max_bssn_diff(a: &BssnState<D>, b: &BssnState<D>) -> f64 {
    scalar_diff(&a.phi, &b.phi)
        .max(tensor_diff(&a.conformal_metric, &b.conformal_metric))
        .max(scalar_diff(&a.trace_k, &b.trace_k))
        .max(tensor_diff(&a.conformal_a, &b.conformal_a))
        .max(vector_diff(&a.conformal_gamma, &b.conformal_gamma))
        .max(scalar_diff(&a.lapse, &b.lapse))
        .max(vector_diff(&a.shift, &b.shift))
}

fn max_dust_diff(a: &DustState<D>, b: &DustState<D>) -> f64 {
    scalar_diff(&a.densitized_rest_mass, &b.densitized_rest_mass)
        .max(vector_diff(&a.densitized_momentum, &b.densitized_momentum))
}

fn scalar_diff(a: &Field<f64, D>, b: &Field<f64, D>) -> f64 {
    a.data
        .iter()
        .zip(&b.data)
        .fold(0.0f64, |worst, (x, y)| worst.max((x - y).abs()))
}

fn vector_diff(a: &Field<Vector<D>, D>, b: &Field<Vector<D>, D>) -> f64 {
    let mut worst = 0.0f64;

    for (x, y) in a.data.iter().zip(&b.data) {
        for i in 0..D {
            worst = worst.max((x[i] - y[i]).abs());
        }
    }

    worst
}

fn tensor_diff(a: &Field<Tensor2<D>, D>, b: &Field<Tensor2<D>, D>) -> f64 {
    let mut worst = 0.0f64;

    for (x, y) in a.data.iter().zip(&b.data) {
        for i in 0..D {
            for j in 0..D {
                worst = worst.max((x[(i, j)] - y[(i, j)]).abs());
            }
        }
    }

    worst
}
