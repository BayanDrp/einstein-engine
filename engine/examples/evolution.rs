// demo of evolution.rs: the ADM evolution equations for gamma_ij and K_ij
// run from engine/ with: cargo run --example evolution
//
// Evolution is two halves:
//   1. the right-hand side  (this file, relativity::evolution)
//   2. a time integrator   (numerics::rk4, still a stub)
// so the step at the bottom is a hand-rolled forward Euler, good enough
// to see the equations act, not enough to trust quantitatively.

use einstein_engine::geometry::metric::{ExtrinsicCurvature, ShiftField, SliceMetric};
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
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

fn print_tensor(label: &str, t: &Tensor2<D>) {
    println!("  {label}");
    for i in 0..D {
        print!("   ");
        for j in 0..D {
            print!("{:10.5}", t[(i, j)]);
        }
        println!();
    }
}

fn main() {
    // 16^3 points, spacing 0.1: a wave of wavelength 1.0 fits ~10 cells
    let grid = Arc::new(Grid::new([16, 16, 16], [0.1, 0.1, 0.1], [-0.8, -0.8, -0.8]));
    let config = BoundaryConfig::<D>::periodic();
    let values = BoundaryValues::<D>::zeros();

    // geodesic slicing: no lapse variation, no shift. this is the
    // simplest gauge and keeps the two equations readable.
    let lapse: Field<f64, D> = Field::new(grid.clone(), 1.0);
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
    // a single plane wave along x. the transverse traceless part lives in
    // h_yy and h_zz with opposite signs:
    //
    //   h_yy =  A cos(k x),   h_zz = -A cos(k x)
    //
    // and K_ij = 1/2 d_t h_ij, so at t = 0:
    //
    //   K_yy = 1/2 A w sin(k x),   K_zz = -1/2 A w sin(k x)
    //
    // with w = k for light speed 1. these two are consistent: the metric
    // equation d_t gamma_ij = -2 K_ij then reproduces h's time derivative.
    let amplitude = 0.01;
    let k = 2.0 * std::f64::consts::PI; // wavelength 1.0
    let omega = k;

    let mut wave_gamma: SliceMetric<D> = Field::new(grid.clone(), flat);
    let mut wave_k: ExtrinsicCurvature<D> = Field::new(grid.clone(), Tensor2::<D>::zero());
    grid.for_each_index(|idx| {
        let x = grid.coords(idx)[0];
        let phase = k * x;

        let mut g = flat;
        g[(1, 1)] += amplitude * phase.cos();
        g[(2, 2)] -= amplitude * phase.cos();
        wave_gamma.set(idx, g);

        let mut kk = Tensor2::<D>::zero();
        kk[(1, 1)] = 0.5 * amplitude * omega * phase.sin();
        kk[(2, 2)] = -0.5 * amplitude * omega * phase.sin();
        wave_k.set(idx, kk);
    });

    let gamma_rhs = spatial_metric_rhs(&lapse, &shift, &wave_gamma, &wave_k, &config, &values);
    let k_rhs = extrinsic_curvature_rhs(&lapse, &shift, &wave_gamma, &wave_k, &config, &values);

    println!("same slice with a plane wave added:");
    println!("  max |d_t gamma_ij| = {:.6}", max_abs(&gamma_rhs));
    println!("  max |d_t K_ij|     = {:.6}", max_abs(&k_rhs));
    println!("  now nonzero: the wave is evolving.\n");

    // --- step 3: one forward Euler step ---------------------------------
    // gamma <- gamma + dt (d_t gamma),  K <- K + dt (d_t K)
    let dt = 0.01;
    let mut next_gamma: SliceMetric<D> = Field::new(grid.clone(), Tensor2::<D>::zero());
    let mut next_k: ExtrinsicCurvature<D> = Field::new(grid.clone(), Tensor2::<D>::zero());

    grid.for_each_index(|idx| {
        let g = wave_gamma.get(idx);
        let kk = wave_k.get(idx);
        let dg = gamma_rhs.get(idx);
        let dk = k_rhs.get(idx);

        let mut g2 = Tensor2::<D>::zero();
        let mut k2 = Tensor2::<D>::zero();
        for i in 0..D {
            for j in 0..D {
                g2[(i, j)] = g[(i, j)] + dt * dg[(i, j)];
                k2[(i, j)] = kk[(i, j)] + dt * dk[(i, j)];
            }
        }
        next_gamma.set(idx, g2);
        next_k.set(idx, k2);
    });

    // probe away from x = 0, where sin(kx) = 0 and the wave sits still
    let probe = [10, 8, 8];
    let x = grid.coords(probe)[0];
    println!("one Euler step of dt = {dt}, sampled at x = {x:.2}:");
    println!("  gamma_yy: {:.6} -> {:.6}", wave_gamma.get(probe)[(1, 1)], next_gamma.get(probe)[(1, 1)]);
    println!("  K_yy:     {:.6} -> {:.6}", wave_k.get(probe)[(1, 1)], next_k.get(probe)[(1, 1)]);

    print_tensor("d_t gamma_ij there:", gamma_rhs.get(probe));
    print_tensor("d_t K_ij there:", k_rhs.get(probe));

    println!("\nnext step is numerics::rk4 -- still a stub, so treat the Euler");
    println!("step above as illustrative rather than numerically meaningful.");
}
