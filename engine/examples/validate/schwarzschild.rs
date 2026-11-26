use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::initial_data::schwarzschild::schwarzschild;
use einstein_engine::numerics::rk4::AdmState;
use einstein_engine::relativity::bssn::BssnState;
use einstein_engine::relativity::constraints::{hamiltonian_constraint, momentum_constraint};

use std::sync::Arc;

// V3 Schwarzschild static slice validation (VALIDATION.md, V3).

const D: usize = 3;
const STATIC_M: f64 = 0.5;
const STATIC_ORIGIN: [f64; D] = [0.5; D];
const WINDOW: (f64, f64) = (0.75, 1.25);

// Window-focused grid for the BSSN RHS check: covers [0.6875, 1.3125]^3
// (a 6-cell halo around [0.75, 1.25]^3) at h = 1/96 so second-order
// truncation in d_t Atilde sits well below 1e-6 while staying fast.
const BSSN_SHAPE: usize = 61;
const BSSN_H: f64 = 1.0 / 96.0;
const BSSN_ORIGIN: [f64; D] = [0.6875; D];

pub fn run() -> Result<(), String> {
    let coarse_h = validate_adm_slice(16, 1.0 / 16.0, STATIC_M, "coarse")?;
    let fine_h = validate_adm_slice(32, 1.0 / 32.0, STATIC_M, "fine")?;
    let ratio = coarse_h / fine_h;

    if coarse_h <= 0.0 || fine_h <= 0.0 || !ratio.is_finite() {
        return Err(format!(
            "Hamiltonian residuals must be finite and nonzero: coarse {coarse_h:.3e}, fine {fine_h:.3e}"
        ));
    }
    if !(3.0..5.0).contains(&ratio) {
        return Err(format!(
            "Hamiltonian residual convergence ratio was {ratio:.3}, expected 3 to 5"
        ));
    }

    validate_bssn_fixed_point(STATIC_M)?;

    println!("V3 Schwarzschild static slice: H {coarse_h:.3e} -> {fine_h:.3e}, ratio {ratio:.3}");
    Ok(())
}

fn validate_adm_slice(shape: usize, h: f64, mass: f64, label: &str) -> Result<f64, String> {
    let grid = Arc::new(Grid::new([shape; D], [h; D], STATIC_ORIGIN));
    let config = BoundaryConfig::<D>::periodic();
    let values = BoundaryValues::<D>::zeros();
    let adm = schwarzschild(grid.clone(), mass).map_err(|error| error.to_owned())?;

    if !adm_is_finite(&adm) {
        return Err(format!("{label} Schwarzschild ADM fields are non-finite"));
    }

    let rho = Field::new(grid.clone(), 0.0);
    let momentum_source = Field::new(grid.clone(), Vector::zero());
    let hamiltonian = hamiltonian_constraint(&adm.gamma, &adm.k, &rho, &config, &values);
    let momentum = momentum_constraint(&adm.gamma, &adm.k, &momentum_source, &config, &values);

    let momentum_max = max_abs_vector(&momentum, label, "momentum constraint")?;
    if momentum_max != 0.0 {
        return Err(format!(
            "{label} momentum constraint is not exactly zero: {momentum_max:.3e}"
        ));
    }

    if !hamiltonian.data.iter().all(|value| value.is_finite()) {
        return Err(format!("{label}: non-finite Hamiltonian constraint"));
    }
    let hamiltonian_max = window_max_scalar(&hamiltonian, label)?;

    check_analytic_window(&adm, mass, label)?;

    Ok(hamiltonian_max)
}

fn validate_bssn_fixed_point(mass: f64) -> Result<(), String> {
    let label = "BSSN fine";
    let grid = Arc::new(Grid::new([BSSN_SHAPE; D], [BSSN_H; D], BSSN_ORIGIN));
    let config = BoundaryConfig::<D>::periodic();
    let values = BoundaryValues::<D>::zeros();
    let adm = schwarzschild(grid, mass).map_err(|error| error.to_owned())?;

    if !adm_is_finite(&adm) {
        return Err(format!("{label} Schwarzschild ADM fields are non-finite"));
    }

    let (metric_error, lapse_error) = check_analytic_window(&adm, mass, label)?;

    let bssn = BssnState::from_adm(&adm, &config, &values);
    let rhs = bssn.rhs(&config, &values);
    if !bssn_is_finite(&rhs) {
        return Err(format!("{label} BSSN RHS contains non-finite fields"));
    }
    let phi_rhs = window_max_scalar(&rhs.phi, label)?;
    let trace_rhs = window_max_scalar(&rhs.trace_k, label)?;
    let a_rhs = window_max_tensor(&rhs.conformal_a, label)?;
    let gamma_rhs = window_max_vector(&rhs.conformal_gamma, label)?;

    for (name, value, limit) in [
        ("d_t phi", phi_rhs, 1.0e-6),
        ("d_t K", trace_rhs, 1.0e-3),
        ("d_t Atilde", a_rhs, 1.0e-6),
        ("d_t Gamma", gamma_rhs, 1.0e-6),
    ] {
        if value >= limit {
            return Err(format!(
                "{label} {name} is {value:.3e}, expected < {limit:.1e}"
            ));
        }
    }

    println!(
        "V3 {label}: BSSN RHS phi/K/A/Gamma {phi_rhs:.3e}/{trace_rhs:.3e}/{a_rhs:.3e}/{gamma_rhs:.3e}, analytic metric/lapse {metric_error:.1e}/{lapse_error:.1e}"
    );
    Ok(())
}

fn check_analytic_window(adm: &AdmState<D>, mass: f64, label: &str) -> Result<(f64, f64), String> {
    let mut metric_error = 0.0f64;
    let mut lapse_error = 0.0f64;
    let mut analytic_fields_finite = true;

    adm.grid.for_each_index(|idx| {
        let coords = adm.grid.coords(idx);
        if !in_window(coords) {
            return;
        }

        let radius = coords
            .iter()
            .map(|coordinate| coordinate * coordinate)
            .sum::<f64>()
            .sqrt();
        let psi = 1.0 + mass / (2.0 * radius);
        let expected_metric = psi.powi(4);
        let expected_lapse = (1.0 - mass / (2.0 * radius)) / psi;
        let numerical_metric = adm.gamma.get(idx);
        let numerical_lapse = *adm.lapse.get(idx);

        if !expected_metric.is_finite()
            || !expected_lapse.is_finite()
            || !numerical_lapse.is_finite()
        {
            analytic_fields_finite = false;
            return;
        }
        lapse_error = lapse_error.max((numerical_lapse - expected_lapse).abs());
        for i in 0..D {
            for j in 0..D {
                let expected = if i == j { expected_metric } else { 0.0 };
                let error = (numerical_metric[(i, j)] - expected).abs();
                if !error.is_finite() {
                    analytic_fields_finite = false;
                } else {
                    metric_error = metric_error.max(error);
                }
            }
        }
    });

    if !analytic_fields_finite {
        return Err(format!(
            "{label} analytic metric/lapse comparison is non-finite"
        ));
    }
    if metric_error > 1.0e-12 || lapse_error > 1.0e-12 {
        return Err(format!(
            "{label} analytic errors exceed 1e-12: metric {metric_error:.3e}, lapse {lapse_error:.3e}"
        ));
    }

    Ok((metric_error, lapse_error))
}

fn in_window(coords: [f64; D]) -> bool {
    coords
        .iter()
        .all(|coordinate| *coordinate >= WINDOW.0 && *coordinate <= WINDOW.1)
}

fn max_abs_vector(field: &Field<Vector<D>, D>, label: &str, what: &str) -> Result<f64, String> {
    let mut worst = 0.0f64;
    for value in &field.data {
        for i in 0..D {
            let component = value[i];
            if !component.is_finite() {
                return Err(format!("{label}: non-finite {what}"));
            }
            worst = worst.max(component.abs());
        }
    }
    Ok(worst)
}

fn window_max_scalar(field: &Field<f64, D>, label: &str) -> Result<f64, String> {
    let mut worst = 0.0f64;
    let mut finite = true;
    field.grid.for_each_index(|idx| {
        if !in_window(field.grid.coords(idx)) {
            return;
        }
        let value = *field.get(idx);
        if value.is_finite() {
            worst = worst.max(value.abs());
        } else {
            finite = false;
        }
    });
    if finite {
        Ok(worst)
    } else {
        Err(format!("{label} window contains a non-finite scalar"))
    }
}

fn window_max_vector(field: &Field<Vector<D>, D>, label: &str) -> Result<f64, String> {
    let mut worst = 0.0f64;
    let mut finite = true;
    field.grid.for_each_index(|idx| {
        if !in_window(field.grid.coords(idx)) {
            return;
        }
        for i in 0..D {
            let component = field.get(idx)[i];
            if component.is_finite() {
                worst = worst.max(component.abs());
            } else {
                finite = false;
            }
        }
    });
    if finite {
        Ok(worst)
    } else {
        Err(format!("{label} window contains a non-finite vector"))
    }
}

fn window_max_tensor(field: &Field<Tensor2<D>, D>, label: &str) -> Result<f64, String> {
    let mut worst = 0.0f64;
    let mut finite = true;
    field.grid.for_each_index(|idx| {
        if !in_window(field.grid.coords(idx)) {
            return;
        }
        let tensor = field.get(idx);
        for i in 0..D {
            for j in 0..D {
                let value = tensor[(i, j)];
                if value.is_finite() {
                    worst = worst.max(value.abs());
                } else {
                    finite = false;
                }
            }
        }
    });
    if finite {
        Ok(worst)
    } else {
        Err(format!("{label} window contains a non-finite tensor"))
    }
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
