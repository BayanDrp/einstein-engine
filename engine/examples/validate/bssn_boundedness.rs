//! V7 BSSN boundedness: the static Schwarzschild slice stays bounded over a
//! long BSSN run.
//!
//! Mass `M = 1` on a `16^3` grid with `h = 1/16`, periodic boundaries, and
//! 40 steps at `dt = 0.5 h`. The puncture sits at the coordinate origin,
//! outside the grid, and the periodic faces wrap non-periodic data, so the
//! diagnostics are taken in the interior window `[0.8, 1.2]^3`.
//!
//! Pass criteria and tolerances follow VALIDATION.md, V7.

use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::initial_data::schwarzschild::schwarzschild;
use einstein_engine::relativity::bssn::BssnState;
use einstein_engine::relativity::constraints::hamiltonian_constraint;
use std::sync::Arc;

const D: usize = 3;
const M: f64 = 1.0;
const H: f64 = 1.0 / 16.0;
const ORIGIN: [f64; D] = [0.5, 0.5, 0.5];
const STEPS: usize = 40;
const DT: f64 = 0.5 * H;

const WINDOW_LOW: f64 = 0.8;
const WINDOW_HIGH: f64 = 1.2;

const K_TOL: f64 = 1.0;
const A_TOL: f64 = 1.0;
const GROWTH_TOL: f64 = 1000.0;

pub fn run() -> Result<(), String> {
    let grid = Arc::new(Grid::new([16; D], [H; D], ORIGIN));
    let config = BoundaryConfig::<D>::periodic();
    let values = BoundaryValues::<D>::zeros();

    let adm = schwarzschild::<D>(grid.clone(), M).map_err(|error| error.to_string())?;
    let mut bssn = BssnState::from_adm(&adm, &config, &values);

    let rho = Field::new(grid.clone(), 0.0);

    let initial_h = window_h(&bssn, &rho, &config, &values);
    if !initial_h.is_finite() || initial_h <= 0.0 {
        return Err(format!(
            "initial window Hamiltonian residual must be positive and finite, got {initial_h:.3e}"
        ));
    }

    let mut worst_h = initial_h;
    let mut final_h = initial_h;

    for step in 1..=STEPS {
        bssn = bssn.step_rk4(DT, &config, &values);
        check_finite(&bssn, step)?;

        final_h = window_h(&bssn, &rho, &config, &values);
        if !final_h.is_finite() {
            return Err(format!(
                "window Hamiltonian residual is not finite at step {step}"
            ));
        }
        worst_h = worst_h.max(final_h);
    }

    let (final_k, final_a) = window_k(&bssn);
    let growth = worst_h / initial_h;

    if final_k >= K_TOL {
        return Err(format!(
            "final max |K| = {final_k:.3e} is not below {K_TOL:.1}"
        ));
    }

    if final_a >= A_TOL {
        return Err(format!(
            "final max |A tilde| = {final_a:.3e} is not below {A_TOL:.1}"
        ));
    }

    if growth >= GROWTH_TOL {
        return Err(format!(
            "worst Hamiltonian residual grew by {growth:.3} (limit {GROWTH_TOL:.0})"
        ));
    }

    println!(
        "V7 BSSN boundedness: initial |H| {initial_h:.3e}, worst |H| {worst_h:.3e}, final |H| {final_h:.3e}, growth {growth:.3}, max|K| {final_k:.3e}, max|A| {final_a:.3e}"
    );

    Ok(())
}

fn in_window(coords: [f64; D]) -> bool {
    coords
        .iter()
        .all(|&x| (WINDOW_LOW..=WINDOW_HIGH).contains(&x))
}

fn window_h(
    bssn: &BssnState<D>,
    rho: &Field<f64, D>,
    config: &BoundaryConfig<D>,
    values: &BoundaryValues<D>,
) -> f64 {
    let adm = bssn.to_adm();
    let residual = hamiltonian_constraint(&adm.gamma, &adm.k, rho, config, values);

    let mut worst = 0.0f64;
    bssn.grid.for_each_index(|idx| {
        if in_window(bssn.grid.coords(idx)) {
            worst = worst.max(residual.get(idx).abs());
        }
    });

    worst
}

fn window_k(bssn: &BssnState<D>) -> (f64, f64) {
    let mut trace = 0.0f64;
    let mut atilde = 0.0f64;

    bssn.grid.for_each_index(|idx| {
        if !in_window(bssn.grid.coords(idx)) {
            return;
        }

        trace = trace.max(bssn.trace_k.get(idx).abs());

        let a = bssn.conformal_a.get(idx);
        for i in 0..D {
            for j in 0..D {
                atilde = atilde.max(a[(i, j)].abs());
            }
        }
    });

    (trace, atilde)
}

fn check_finite(bssn: &BssnState<D>, step: usize) -> Result<(), String> {
    let scalar_finite = |field: &Field<f64, D>| field.data.iter().all(|value| value.is_finite());
    let vector_finite = |field: &Field<Vector<D>, D>| {
        field
            .data
            .iter()
            .all(|value| (0..D).all(|i| value[i].is_finite()))
    };
    let tensor_finite = |field: &Field<Tensor2<D>, D>| {
        field
            .data
            .iter()
            .all(|value| (0..D).all(|i| (0..D).all(|j| value[(i, j)].is_finite())))
    };

    let finite = scalar_finite(&bssn.phi)
        && tensor_finite(&bssn.conformal_metric)
        && scalar_finite(&bssn.trace_k)
        && tensor_finite(&bssn.conformal_a)
        && vector_finite(&bssn.conformal_gamma)
        && scalar_finite(&bssn.lapse)
        && vector_finite(&bssn.shift);

    if finite {
        Ok(())
    } else {
        Err(format!("non-finite BSSN field at step {step}"))
    }
}
