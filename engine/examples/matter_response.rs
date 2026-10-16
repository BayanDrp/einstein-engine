//! Fixed-source matter-response diagnostic (NOT fully-coupled evolution).
//!
//! A uniform dust ball on a flat slice with K_ij = C delta_ij and
//! rho_0 = 6 C^2 / (16 pi) satisfies H = M = 0 exactly, so this starts from
//! genuine on-shell initial data with no elliptic solve (an FLRW patch).
//! The geometry is then evolved with the matter-aware BSSN integrator while
//! the sources are held fixed.
//!
//! Because fixed rho breaks continuity as soon as the volume moves, results
//! past t = 0 are a response diagnostic, not a cosmology run. What is
//! validated here: exact on-shell start, exact FLRW initial slopes, and
//! bounded (non-blowing-up) evolution with saved diagnostics.
//!
//! Run from engine/: cargo run --example matter_response
//! Writes matter_response.csv into the working directory.

use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::numerics::rk4::AdmState;
use einstein_engine::physics::stress_energy::dust;
use einstein_engine::relativity::bssn::BssnState;
use einstein_engine::relativity::matter::matter_sources;
use einstein_engine::validation::constraints::residuals;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;

const D: usize = 3;
// K_ij = C delta_ij (C > 0: contracting slice).
const C: f64 = 0.1;
// 6 C^2 = 16 pi rho_0  =>  H = 0 on the initial slice.
const RHO0: f64 = 6.0 * C * C / (16.0 * std::f64::consts::PI);
const H: f64 = 0.125;
const STEPS: usize = 16;

fn main() {
    let grid = Arc::new(Grid::new([8; D], [H; D], [0.5; D]));
    let config = BoundaryConfig::<D>::periodic();
    let values = BoundaryValues::<D>::zeros();

    // --- initial data: flat metric, constant mean curvature, uniform dust
    let mut identity = Tensor2::<D>::zero();
    for i in 0..D {
        identity[(i, i)] = 1.0;
    }
    let mut k0 = Tensor2::<D>::zero();
    for i in 0..D {
        for j in 0..D {
            k0[(i, j)] = if i == j { C } else { 0.0 };
        }
    }
    let adm = AdmState {
        grid: grid.clone(),
        gamma: Field::new(grid.clone(), identity),
        k: Field::new(grid.clone(), k0),
        lapse: Field::new(grid.clone(), 1.0),
        shift: Field::new(grid.clone(), Vector::<D>::zero()),
    };
    let t_field = dust::<4, D>(&Field::new(grid.clone(), RHO0));
    let src =
        matter_sources::<4, D>(&t_field, &adm.lapse, &adm.shift, &adm.gamma);

    // --- t = 0 validation: exactly on-shell (up to roundoff)
    let r0 = residuals(
        &adm.gamma,
        &adm.k,
        &src.rho,
        &src.momentum,
        &config,
        &values,
    );
    assert!(
        r0.hamiltonian_max < 1e-12,
        "initial H must vanish, got {}",
        r0.hamiltonian_max
    );
    assert!(
        r0.momentum_max < 1e-14,
        "initial M must vanish, got {}",
        r0.momentum_max
    );

    // --- t = 0 validation: FLRW initial slopes, d_t K = 12 pi rho_0,
    // d_t phi = -C/2, d_t alpha = -6 C, d_t gtilde = d_t Gamma = 0
    let bssn = BssnState::from_adm(&adm, &config, &values);
    let rhs0 = bssn.rhs_with_matter(&src, &config, &values);
    let pi = std::f64::consts::PI;
    grid.for_each_index(|idx| {
        assert!(
            (*rhs0.trace_k.get(idx) - 12.0 * pi * RHO0).abs() < 1e-12,
            "d_t K must equal 12 pi rho_0"
        );
        assert!(
            (*rhs0.phi.get(idx) + C / 2.0).abs() < 1e-12,
            "d_t phi must equal -C/2"
        );
        assert!(
            (*rhs0.lapse.get(idx) + 6.0 * C).abs() < 1e-12,
            "d_t alpha must equal -6 C"
        );
        let gm = rhs0.conformal_metric.get(idx);
        let gg = rhs0.conformal_gamma.get(idx);
        for i in 0..D {
            assert!(gg[i].abs() < 1e-12, "d_t Gamma must vanish");
            for j in 0..D {
                assert!(gm[(i, j)].abs() < 1e-12, "d_t gtilde must vanish");
            }
        }
    });

    // --- evolve with fixed sources, monitoring constraints
    let dt = 0.5 * H;
    let mut state = bssn;
    let mut out =
        File::create("matter_response.csv").expect("cannot create output csv");
    writeln!(out, "step,t,min_alpha,max_abs_K,Hmax,Mmax")
        .expect("cannot write csv header");
    println!("step | t     | min_alpha | max|K|   | |H|max  | |M|max");

    for step in 0..=STEPS {
        let a = state.to_adm();
        let r = residuals(&a.gamma, &a.k, &src.rho, &src.momentum, &config, &values);
        let mut min_alpha = f64::INFINITY;
        let mut max_k = 0.0f64;
        grid.for_each_index(|idx| {
            min_alpha = min_alpha.min(*state.lapse.get(idx));
            max_k = max_k.max(state.trace_k.get(idx).abs());
        });
        assert!(
            min_alpha.is_finite() && max_k.is_finite(),
            "non-finite state at step {step}"
        );
        assert!(min_alpha > 0.0, "lapse went non-positive at step {step}");
        writeln!(
            out,
            "{step},{:.6},{min_alpha:.6e},{max_k:.6e},{:.3e},{:.3e}",
            step as f64 * dt,
            r.hamiltonian_max,
            r.momentum_max,
        )
        .expect("cannot write csv row");
        println!(
            "{step:>4} | {t:5.3} | {min_alpha:.3e} | {max_k:.3e} | {h:.3e} | {m:.3e}",
            t = step as f64 * dt,
            h = r.hamiltonian_max,
            m = r.momentum_max,
        );

        if step < STEPS {
            state = state.step_rk4_with_matter(dt, &src, &config, &values);
        }
    }

    assert!(
        {
            let a = state.to_adm();
            let r = residuals(&a.gamma, &a.k, &src.rho, &src.momentum, &config, &values);
            r.hamiltonian_max < 1.0
        },
        "constraint blow-up over the run"
    );
    println!("done: diagnostics saved to matter_response.csv");
}
