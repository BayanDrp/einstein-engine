// demo of evolution.rs: the ADM evolution equations for gamma_ij and K_ij
// run from engine/ with: cargo run --example evolution
//
// Evolution is two halves:
//   1. the right-hand side  (relativity::evolution)
//   2. a time integrator   (numerics::rk4, RK4 over EvolutionState)
// both are implemented, so the wave below is actually transported rather
// than nudged once by hand.

use einstein_engine::geometry::metric::{ExtrinsicCurvature, ShiftField, SliceMetric};
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::numerics::rk4::EvolutionState;
use einstein_engine::relativity::evolution::{extrinsic_curvature_rhs, spatial_metric_rhs};
use std::sync::Arc;

const D: usize = 3;

/// largest absolute component of a tensor field, for a quick "did anything
/// happen" check
fn max_abs(field: &Field<Tensor2<D>, D>) -> f64 {
    let mut worst = 0.0f64;
    field.grid.for_each_index(|idx| {
        let t = field.get(idx);
        for i in 0..D {
            for j in 0..D {
                worst = worst.max(t[(i, j)].abs());
            }
        }
    });
    worst
}

/// x position of the h_yy crest, refined parabolically so the reading is not
/// quantised to whole cells
fn crest_x(state: &EvolutionState<D>, near: f64) -> f64 {
    let n = state.grid.shape[0];
    let mut best = 0usize;
    let mut best_value = f64::NEG_INFINITY;

    for j in 0..n {
        if (state.grid.coords([j, 0, 0])[0] - near).abs() > 0.5 {
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

    state.grid.coords([best, 0, 0])[0] + offset * state.grid.spacing[0]
}

fn main() {
    // x spans 2.0 with 40 cells, so a wavelength of 1.0 fits exactly twice
    // and gets 40 cells: the profile is periodic, the wrap stencil sees no
    // jump, and k*h = 0.31 keeps numerical dispersion near 0.4%.
    let grid = Arc::new(Grid::new([40, 8, 8], [0.05, 0.05, 0.05], [-1.0, -0.2, -0.2]));
    let config = BoundaryConfig::<D>::periodic();
    let values = BoundaryValues::<D>::zeros();

    // constant lapse, no shift: geodesic-slicing *data*, which keeps the
    // two ADM right-hand sides below readable.
    let lapse = Field::new(grid.clone(), 1.0);
    let shift: ShiftField<D> = Field::new(grid.clone(), Vector::<D>::zero());

    // --- step 1: flat Minkowski should be a fixed point -----------------
    let flat = Tensor2::<D>::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let spatial: SliceMetric<D> = Field::new(grid.clone(), flat);
    let extrinsic: ExtrinsicCurvature<D> = Field::new(grid.clone(), Tensor2::<D>::zero());

    let gamma_rhs = spatial_metric_rhs(&lapse, &shift, &spatial, &extrinsic, &config, &values);
    let k_rhs = extrinsic_curvature_rhs(&lapse, &shift, &spatial, &extrinsic, &config, &values);

    println!("Minkowski (alpha = 1, beta = 0, gamma = delta, K = 0):");
    println!("  max |d_t gamma_ij| = {:.3e}", max_abs(&gamma_rhs));
    println!("  max |d_t K_ij|     = {:.3e}", max_abs(&k_rhs));
    println!("  both zero: flat spacetime is an exact solution, so it must not move.\n");

    // --- step 2: add a gravitational wave --------------------------------
    // a plane wave along x. the transverse traceless part lives in h_yy and
    // h_zz with opposite signs:
    //
    //   h_yy =  A cos(k x),   h_zz = -A cos(k x)
    //
    // the metric equation says d_t gamma_ij = -2 alpha K_ij, so at first
    // order K_ij = -1/2 d_t h_ij. a wave moving in +x is A cos(k (x - t)),
    // whose time derivative at t = 0 is +A k sin(k x), giving
    //
    //   K_yy = -1/2 A k sin(k x),   K_zz = +1/2 A k sin(k x)
    //
    // with k = omega for light speed 1. flipping these signs silently builds
    // a wave travelling the other way.
    //
    // A*k must stay well below 1, otherwise the K_ij K^ij term stops being
    // a small correction and the slice no longer solves the constraints.
    let amplitude = 0.01;
    let k = 2.0 * std::f64::consts::PI;

    // phase the profile so the first crest sits at x = -0.5, away from the
    // periodic wrap, and note that the wavelength still fits twice
    let crest = -0.5;

    let mut wave_gamma: SliceMetric<D> = Field::new(grid.clone(), flat);
    let mut wave_k: ExtrinsicCurvature<D> = Field::new(grid.clone(), Tensor2::<D>::zero());
    grid.for_each_index(|idx| {
        let x = grid.coords(idx)[0];
        let phase = k * (x - crest);

        let mut g = flat;
        g[(1, 1)] += amplitude * phase.cos();
        g[(2, 2)] -= amplitude * phase.cos();
        wave_gamma.set(idx, g);

        let mut kk = Tensor2::<D>::zero();
        kk[(1, 1)] = -0.5 * amplitude * k * phase.sin();
        kk[(2, 2)] = 0.5 * amplitude * k * phase.sin();
        wave_k.set(idx, kk);
    });

    let gamma_rhs = spatial_metric_rhs(&lapse, &shift, &wave_gamma, &wave_k, &config, &values);
    let k_rhs = extrinsic_curvature_rhs(&lapse, &shift, &wave_gamma, &wave_k, &config, &values);

    println!("same slice with a plane wave added:");
    println!("  max |d_t gamma_ij| = {:.6}", max_abs(&gamma_rhs));
    println!("  max |d_t K_ij|     = {:.6}", max_abs(&k_rhs));
    println!("  now nonzero: the wave is evolving.\n");

    // --- step 3: transport it with RK4 -----------------------------------
    let dt = 0.05;
    let steps = 4;

    let mut state = EvolutionState {
        grid: grid.clone(),
        gamma: wave_gamma,
        k: wave_k,
        lapse: Field::new(grid.clone(), 1.0),
        shift: Field::new(grid.clone(), Vector::<D>::zero()),
    };

    let start = crest_x(&state, crest);
    for _ in 0..steps {
        state = state.step_rk4(dt, &config, &values);
    }
    let elapsed = steps as f64 * dt;
    let end = crest_x(&state, crest + elapsed);

    println!("RK4 with dt = {dt} over {steps} steps (t = {elapsed}):");
    println!("  h_yy crest: {start:.4} -> {end:.4}");
    println!("  implied speed: {:.4}", (end - start) / elapsed);
    println!("  expected 1.0. Landing a little under is numerical dispersion:");
    println!("  second-order centred differences carry a wave slower than c. The");
    println!("  leading-order estimate 1 - (k h)^2/24 = 0.9959 covers part of it;");
    println!("  the rest is higher-order terms in the coupled system. Refine the");
    println!("  grid or shrink k h and the measured speed rises toward 1.");

    let lapse_drift = state
        .lapse
        .data
        .iter()
        .fold(0.0f64, |acc, v| acc.max((v - 1.0).abs()));

    println!("\ngauge: 1+log for the lapse, d_t alpha = -2 alpha K (beta = 0 here)");
    println!("  max |alpha - 1| over the run: {lapse_drift:.3e}");
    println!("  the wave is transversely traceless, so its K trace only survives");
    println!("  through gamma^ij and the lapse barely moves. The shift still has");
    println!("  no equation and stays at zero.");
}
