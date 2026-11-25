// V2 Linear wave: a transverse traceless plane wave must move in the +x
// direction at the finite difference light speed without losing amplitude.
//
// Pass criteria and tolerances follow VALIDATION.md, V2.

use einstein_engine::geometry::metric::{ExtrinsicCurvature, LapseField, ShiftField, SliceMetric};
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::Grid;
use einstein_engine::numerics::rk4::AdmState;
use std::sync::Arc;

const D: usize = 3;

/// Wave parameters, matching engine/examples/evolution.rs.
const AMPLITUDE: f64 = 0.01;
const K: f64 = 2.0 * std::f64::consts::PI;

/// x spans 2.0 with 128 cells, so the wavelength 1.0 fits exactly twice and
/// k h = pi / 32. The transverse directions are uniform.
const NX: usize = 128;
const H: f64 = 2.0 / NX as f64;
const ORIGIN: [f64; D] = [-1.0, -0.2, -0.2];

/// dt = 0.0125 with h = 1/64 gives CFL 0.8, and 32 steps reach t = 0.4.
/// The crest then sits at x = -0.1, far from the periodic wrap.
const DT: f64 = 0.0125;
const STEPS: usize = 32;

/// First crest of cos(k (x - CREST0)).
const CREST0: f64 = -0.5;

/// VALIDATION.md V2: 0.5 percent in speed, 1 percent in amplitude.
const SPEED_TOL: f64 = 5e-3;
const AMPLITUDE_TOL: f64 = 1e-2;

pub fn run() -> Result<(), String> {
    let grid = Arc::new(Grid::new([NX, 8, 8], [H; D], ORIGIN));
    let config = BoundaryConfig::<D>::periodic();
    let values = BoundaryValues::<D>::zeros();

    let mut identity = Tensor2::<D>::zero();
    for i in 0..D {
        identity[(i, i)] = 1.0;
    }

    // The transverse traceless packet from the evolution example:
    //
    //   h_yy =  A cos(k (x - x0)),   h_zz = -A cos(k (x - x0))
    //
    // with K_ij = -1/2 d_t gamma_ij at t = 0, so the wave travels toward +x.
    let mut gamma: SliceMetric<D> = SliceMetric::new(grid.clone(), identity);
    let mut extrinsic: ExtrinsicCurvature<D> =
        ExtrinsicCurvature::new(grid.clone(), Tensor2::zero());

    grid.for_each_index(|idx| {
        let x = grid.coords(idx)[0];
        let phase = K * (x - CREST0);
        let wave = AMPLITUDE * phase.cos();
        let d_wave = AMPLITUDE * K * phase.sin();

        let mut g = identity;
        g[(1, 1)] += wave;
        g[(2, 2)] -= wave;
        gamma.set(idx, g);

        let mut k = Tensor2::<D>::zero();
        k[(1, 1)] = -0.5 * d_wave;
        k[(2, 2)] = 0.5 * d_wave;
        extrinsic.set(idx, k);
    });

    let lapse: LapseField<D> = LapseField::new(grid.clone(), 1.0);
    let shift: ShiftField<D> = ShiftField::new(grid.clone(), Vector::zero());

    let mut state = AdmState {
        grid: grid.clone(),
        lapse,
        shift,
        gamma,
        k: extrinsic,
    };

    if !adm_is_finite(&state) {
        return Err("initial non finite ADM field".to_string());
    }

    let (crest_start, amplitude_start) = crest(&state, CREST0);
    if amplitude_start <= 0.0 {
        return Err("initial wave amplitude is not positive".to_string());
    }

    for step in 1..=STEPS {
        state = state.step_rk4(DT, &config, &values);

        if !adm_is_finite(&state) {
            return Err(format!("non finite ADM field at step {step}"));
        }
    }

    let elapsed = STEPS as f64 * DT;
    let (crest_end, amplitude_end) = crest(&state, crest_start + elapsed);
    let speed = (crest_end - crest_start) / elapsed;

    let kh = K * H;
    let half = 0.5 * kh;
    let speed_fd = half.sin() / half;
    let speed_error = (speed - speed_fd).abs() / speed_fd;
    let amplitude_error = (amplitude_end - amplitude_start).abs() / amplitude_start;

    if crest_end <= crest_start {
        return Err(format!(
            "wave moved toward -x: crest {crest_start:.4} -> {crest_end:.4}",
        ));
    }

    if speed_error > SPEED_TOL {
        return Err(format!(
            "speed {speed:.5} differs from finite difference speed {speed_fd:.5} by {:.2}%",
            speed_error * 100.0,
        ));
    }

    if amplitude_error > AMPLITUDE_TOL {
        return Err(format!(
            "amplitude {amplitude_start:.5e} -> {amplitude_end:.5e}, error {:.2}%",
            amplitude_error * 100.0,
        ));
    }

    println!(
        "V2 Linear wave: crest {crest_start:.4} -> {crest_end:.4}, speed {speed:.5} (fd {speed_fd:.5}), amplitude {amplitude_start:.5e} -> {amplitude_end:.5e}"
    );

    Ok(())
}

/// Position and height of the h_yy crest nearest to `near`, refined
/// parabolically so the reading is not quantised to whole cells.
fn crest(state: &AdmState<D>, near: f64) -> (f64, f64) {
    let n = state.grid.shape[0];
    let mut best = 0usize;
    let mut best_value = f64::NEG_INFINITY;

    for j in 0..n {
        let x = state.grid.coords([j, 0, 0])[0];
        // Half a wavelength around the expected position keeps the search
        // on the crest we are tracking, away from the neighbouring crest.
        if (x - near).abs() > 0.5 {
            continue;
        }
        let value = state.gamma.get([j, 0, 0])[(1, 1)];
        if value > best_value {
            best_value = value;
            best = j;
        }
    }

    let before = state.gamma.get([(best + n - 1) % n, 0, 0])[(1, 1)];
    let after = state.gamma.get([(best + 1) % n, 0, 0])[(1, 1)];
    let denominator = before - 2.0 * best_value + after;

    let offset = if denominator.abs() < 1.0e-300 {
        0.0
    } else {
        0.5 * (before - after) / denominator
    };
    let peak = if denominator.abs() < 1.0e-300 {
        best_value
    } else {
        best_value - (before - after) * (before - after) / (8.0 * denominator)
    };

    let x = state.grid.coords([best, 0, 0])[0] + offset * state.grid.spacing[0];
    (x, peak - 1.0)
}

fn adm_is_finite(state: &AdmState<D>) -> bool {
    state.lapse.data.iter().all(|v| v.is_finite())
        && state
            .shift
            .data
            .iter()
            .all(|v| (0..D).all(|i| v[i].is_finite()))
        && state
            .gamma
            .data
            .iter()
            .all(|v| (0..D).all(|i| (0..D).all(|j| v[(i, j)].is_finite())))
        && state
            .k
            .data
            .iter()
            .all(|v| (0..D).all(|i| (0..D).all(|j| v[(i, j)].is_finite())))
}
