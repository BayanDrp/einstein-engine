//! V1 Minkowski: flat spacetime is an exact fixed point of both the ADM and
//! the BSSN evolution.
//!
//! Pass criteria and tolerances follow VALIDATION.md, V1.

use einstein_engine::geometry::metric::{ExtrinsicCurvature, LapseField, ShiftField, SliceMetric};
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::numerics::rk4::AdmState;
use einstein_engine::relativity::bssn::BssnState;
use einstein_engine::validation::constraints::{residuals, ConstraintResidual};
use std::sync::Arc;

const D: usize = 3;
const H: f64 = 0.5;

/// VALIDATION.md V1: at least 200 steps at dt = 0.25 h.
const STEPS: usize = 200;
const DT: f64 = 0.25 * H;

/// Flat data is exact, so only roundoff is allowed.
const TOL: f64 = 1e-12;

pub fn run() -> Result<(), String> {
    let grid = Arc::new(Grid::new([4, 4, 4], [H; D], [0.0; D]));
    let config = BoundaryConfig::<D>::periodic();
    let values = BoundaryValues::<D>::zeros();

    // Minkowski ADM slice: unit lapse, zero shift, flat spatial metric,
    // and zero extrinsic curvature.
    let mut identity = Tensor2::<D>::zero();
    for i in 0..D {
        identity[(i, i)] = 1.0;
    }

    let lapse: LapseField<D> = LapseField::new(grid.clone(), 1.0);
    let shift: ShiftField<D> = ShiftField::new(grid.clone(), Vector::zero());
    let spatial: SliceMetric<D> = SliceMetric::new(grid.clone(), identity);
    let extrinsic: ExtrinsicCurvature<D> = ExtrinsicCurvature::new(grid.clone(), Tensor2::zero());

    // Create an ADM state from the metric elements.
    let adm_state = AdmState {
        grid: grid.clone(),
        lapse,
        shift,
        gamma: spatial,
        k: extrinsic,
    };

    // Convert the ADM slice to BSSN instead of writing the fields by hand.
    let bssn_state = BssnState::from_adm(&adm_state, &config, &values);

    // Vacuum sources: rho = 0 and S^i = 0.
    let rho = Field::new(grid.clone(), 0.0);
    let momentum = Field::new(grid.clone(), Vector::zero());

    // The constraints must already hold on the initial slice, in both
    // representations of the same geometry.
    let initial_adm_res = residuals(
        &adm_state.gamma,
        &adm_state.k,
        &rho,
        &momentum,
        &config,
        &values,
    );
    check_constraints("initial ADM", &initial_adm_res)?;

    let bssn_as_adm = bssn_state.to_adm();
    let initial_bssn_res = residuals(
        &bssn_as_adm.gamma,
        &bssn_as_adm.k,
        &rho,
        &momentum,
        &config,
        &values,
    );
    check_constraints("initial BSSN", &initial_bssn_res)?;

    // Snapshot the initial data, then evolve both formulations side by side.
    let initial_adm = adm_state.clone();
    let initial_bssn = bssn_state.clone();

    let mut adm = adm_state;
    let mut bssn = bssn_state;

    for step in 1..=STEPS {
        adm = adm.step_rk4(DT, &config, &values);
        bssn = bssn.step_rk4(DT, &config, &values);

        if !adm_is_finite(&adm) {
            return Err(format!("ADM field became non finite at step {step}"));
        }
        if !bssn_is_finite(&bssn) {
            return Err(format!("BSSN field became non finite at step {step}"));
        }
    }

    // Flat data must not move at all beyond roundoff.
    let adm_drift = max_adm_drift(&adm, &initial_adm);
    if adm_drift > TOL {
        return Err(format!("ADM drift {adm_drift:.3e} exceeds {TOL:.1e}"));
    }

    let bssn_drift = max_bssn_drift(&bssn, &initial_bssn);
    if bssn_drift > TOL {
        return Err(format!("BSSN drift {bssn_drift:.3e} exceeds {TOL:.1e}"));
    }

    // The constraints must still hold at the end of the run.
    let final_adm_res = residuals(&adm.gamma, &adm.k, &rho, &momentum, &config, &values);
    check_constraints("final ADM", &final_adm_res)?;

    let bssn_as_adm = bssn.to_adm();
    let final_bssn_res = residuals(
        &bssn_as_adm.gamma,
        &bssn_as_adm.k,
        &rho,
        &momentum,
        &config,
        &values,
    );
    check_constraints("final BSSN", &final_bssn_res)?;

    println!(
        "V1 Minkowski: H {:.3e}/{:.3e}, M {:.3e}/{:.3e}, ADM drift {:.3e}, BSSN drift {:.3e}",
        initial_adm_res.hamiltonian_max,
        final_adm_res.hamiltonian_max,
        initial_adm_res.momentum_max,
        final_adm_res.momentum_max,
        adm_drift,
        bssn_drift,
    );

    Ok(())
}

fn check_constraints(label: &str, residual: &ConstraintResidual) -> Result<(), String> {
    if !residual.hamiltonian_max.is_finite() || !residual.momentum_max.is_finite() {
        return Err(format!("{label}: non finite constraint residual"));
    }

    if residual.hamiltonian_max > TOL || residual.momentum_max > TOL {
        return Err(format!(
            "{label}: H = {:.3e}, M = {:.3e}, tolerance {TOL:.1e}",
            residual.hamiltonian_max, residual.momentum_max,
        ));
    }

    Ok(())
}

fn adm_is_finite(state: &AdmState<D>) -> bool {
    scalar_finite(&state.lapse)
        && vector_finite(&state.shift)
        && tensor_finite(&state.gamma)
        && tensor_finite(&state.k)
}

fn bssn_is_finite(state: &BssnState<D>) -> bool {
    scalar_finite(&state.phi)
        && tensor_finite(&state.conformal_metric)
        && scalar_finite(&state.trace_k)
        && tensor_finite(&state.conformal_a)
        && vector_finite(&state.conformal_gamma)
        && scalar_finite(&state.lapse)
        && vector_finite(&state.shift)
}

fn max_adm_drift(a: &AdmState<D>, b: &AdmState<D>) -> f64 {
    tensor_diff(&a.gamma, &b.gamma)
        .max(tensor_diff(&a.k, &b.k))
        .max(scalar_diff(&a.lapse, &b.lapse))
        .max(vector_diff(&a.shift, &b.shift))
}

fn max_bssn_drift(a: &BssnState<D>, b: &BssnState<D>) -> f64 {
    scalar_diff(&a.phi, &b.phi)
        .max(tensor_diff(&a.conformal_metric, &b.conformal_metric))
        .max(scalar_diff(&a.trace_k, &b.trace_k))
        .max(tensor_diff(&a.conformal_a, &b.conformal_a))
        .max(vector_diff(&a.conformal_gamma, &b.conformal_gamma))
        .max(scalar_diff(&a.lapse, &b.lapse))
        .max(vector_diff(&a.shift, &b.shift))
}

fn scalar_finite(field: &Field<f64, D>) -> bool {
    field.data.iter().all(|value| value.is_finite())
}

fn vector_finite(field: &Field<Vector<D>, D>) -> bool {
    field
        .data
        .iter()
        .all(|value| (0..D).all(|i| value[i].is_finite()))
}

fn tensor_finite(field: &Field<Tensor2<D>, D>) -> bool {
    field
        .data
        .iter()
        .all(|value| (0..D).all(|i| (0..D).all(|j| value[(i, j)].is_finite())))
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
