// V3 is not wired into `suite::run` yet; remove this once it is.
#![allow(dead_code)]

use einstein_engine::geometry::metric::{ExtrinsicCurvature, LapseField, ShiftField, SliceMetric};
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::numerics::rk4::AdmState;
use einstein_engine::relativity::bssn::BssnState;
use einstein_engine::validation::constraints::residuals;

use std::sync::Arc;

// V3 Schwarzschild validation

const D: usize = 3;
const H: f64 = 0.5;
const STEPS: usize = 200;
const DT: f64 = 0.25 * H;
const M: f64 = 1.0;

// Cartesian domain origin.
// All grid points must satisfy rho > M/2.
const ORIGIN: [f64; D] = [2.0; D];

pub fn run() -> Result<(), String> {
    let grid = Arc::new(Grid::new([4, 4, 4], [H; D], ORIGIN));

    let config = BoundaryConfig::<D>::periodic();
    let values = BoundaryValues::<D>::zeros();

    // Identity spatial tensor.
    let mut identity = Tensor2::<D>::zero();

    for i in 0..D {
        identity[(i, i)] = 1.0;
    }

    // Initial ADM fields.
    let mut lapse: LapseField<D> = LapseField::new(grid.clone(), 0.0);

    let shift: ShiftField<D> = ShiftField::new(grid.clone(), Vector::zero());

    let mut spatial: SliceMetric<D> = SliceMetric::new(grid.clone(), identity);

    let extrinsic: ExtrinsicCurvature<D> =
        ExtrinsicCurvature::new(grid.clone(), Tensor2::<D>::zero());

    // Schwarzschild isotropic initial data.
    grid.for_each_index(|idx| {
        let coords = grid.coords(idx);

        let x = coords[0];
        let y = coords[1];
        let z = coords[2];

        let r_iso = (x * x + y * y + z * z).sqrt();

        let psi = 1.0 + M / (2.0 * r_iso);

        let alpha = (1.0 - M / (2.0 * r_iso)) / psi;

        let psi4 = psi.powi(4);

        let mut gamma = identity;

        for i in 0..D {
            gamma[(i, i)] *= psi4;
        }

        lapse.set(idx, alpha);
        spatial.set(idx, gamma);
    });

    let initialized_adm_state = AdmState {
        grid: grid.clone(),
        lapse,
        shift,
        gamma: spatial,
        k: extrinsic,
    };

    let initialized_bssn_state = BssnState::from_adm(&initialized_adm_state, &config, &values);

    // Initial finite checks.
    if !adm_is_finite(&initialized_adm_state) {
        return Err("Initial ADM fields contain non-finite values".into());
    }

    if !bssn_is_finite(&initialized_bssn_state) {
        return Err("Initial BSSN fields contain non-finite values".into());
    }

    // Vacuum matter sources.
    let rho = Field::new(grid.clone(), 0.0);
    let momentum = Field::new(grid.clone(), Vector::zero());

    // Initial ADM constraint residuals.
    let initial_adm_res = residuals(
        &initialized_adm_state.gamma,
        &initialized_adm_state.k,
        &rho,
        &momentum,
        &config,
        &values,
    );

    // Evolution.
    let initial_adm_snapshot = initialized_adm_state.clone();
    let initial_bssn_snapshot = initialized_bssn_state.clone();
    let mut adm_state = initialized_adm_state;
    let mut bssn_state = initialized_bssn_state;

    for _ in 0..STEPS {
        adm_state = adm_state.step_rk4(DT, &config, &values);

        bssn_state = bssn_state.step_rk4(DT, &config, &values);

        if !adm_is_finite(&adm_state) {
            return Err("ADM evolution produced non-finite fields".into());
        }

        if !bssn_is_finite(&bssn_state) {
            return Err("BSSN evolution produced non-finite fields".into());
        }
    }

    // Final ADM constraint residuals.
    let final_adm_res = residuals(
        &adm_state.gamma,
        &adm_state.k,
        &rho,
        &momentum,
        &config,
        &values,
    );

    check_constraint_residual("initial ADM", &initial_adm_res)?;
    check_constraint_residual("final ADM", &final_adm_res)?;

    if initial_adm_res.momentum_max != 0.0 {
        return Err(format!(
            "initial momentum constraint is not exactly zero: {:.3e}",
            initial_adm_res.momentum_max
        ));
    }
    if final_adm_res.momentum_max != 0.0 {
        return Err(format!(
            "final momentum constraint is not exactly zero: {:.3e}",
            final_adm_res.momentum_max
        ));
    }

    let adm_drift = max_adm_drift(&initial_adm_snapshot, &adm_state);
    let bssn_drift = max_bssn_drift(&initial_bssn_snapshot, &bssn_state);

    println!(
        "V3 residuals: initial H/M {:.3e}/{:.3e}, final H/M {:.3e}/{:.3e}",
        initial_adm_res.hamiltonian_max,
        initial_adm_res.momentum_max,
        final_adm_res.hamiltonian_max,
        final_adm_res.momentum_max,
    );
    println!("V3 drift: ADM {:.3e}, BSSN {:.3e}", adm_drift, bssn_drift);

    Ok(())
}

fn check_constraint_residual(
    label: &str,
    residual: &einstein_engine::validation::constraints::ConstraintResidual,
) -> Result<(), String> {
    if !residual.hamiltonian_max.is_finite() || !residual.momentum_max.is_finite() {
        return Err(format!("{label}: non-finite constraint residual"));
    }
    Ok(())
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
