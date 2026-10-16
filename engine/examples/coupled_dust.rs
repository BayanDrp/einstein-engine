//! Coupled dust+BSSN evolution (genuinely coupled, not fixed-source).
//!
//! Phase A: the same FLRW-patch dust as matter_response (flat metric,
//! K_ij = C delta_ij, uniform dust with 6 C^2 = 16 pi rho, exactly
//! on-shell), but here the dust density AND momentum are evolved states:
//! every RK4 stage rebuilds the matter sources from the intermediate dust
//! and geometry via BssnState::step_rk4_with_dust. With v = 0 the momentum
//! stays zero while rho dilutes with the volume, so |H| stays at roundoff
//! instead of drifting linearly as in the fixed-source run.
//!
//! Phase B: flat-space transport of a Gaussian bump at v = 0.2 on a frozen
//! Minkowski background (prescribed-background test, not self-consistent
//! evolution: the geometry is held fixed while matter moves, so the
//! constraints sit at their off-shell values by construction).
//!
//! Run from engine/: cargo run --example coupled_dust
//! Writes coupled_dust.csv and coupled_dust_transport.csv into the working
//! directory.

use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::numerics::rk4::AdmState;
use einstein_engine::physics::dust::DustState;
use einstein_engine::physics::dust_evolution::dust_step_rk4;
use einstein_engine::relativity::bssn::BssnState;
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
    phase_a_coupled_flrw();
    phase_b_transport();
}

fn phase_a_coupled_flrw() {
    let grid = Arc::new(Grid::new([8; D], [H; D], [0.5; D]));
    let config = BoundaryConfig::<D>::periodic();
    let values = BoundaryValues::<D>::zeros();

    // --- initial data: flat metric, constant mean curvature, uniform dust
    let mut identity = Tensor2::<D>::zero();
    let mut k0 = Tensor2::<D>::zero();
    for i in 0..D {
        identity[(i, i)] = 1.0;
        k0[(i, i)] = C;
    }
    let gamma = Field::new(grid.clone(), identity);
    let adm = AdmState {
        grid: grid.clone(),
        gamma: gamma.clone(),
        k: Field::new(grid.clone(), k0),
        lapse: Field::new(grid.clone(), 1.0),
        shift: Field::new(grid.clone(), Vector::<D>::zero()),
    };
    let dust = DustState::from_primitives(
        &Field::new(grid.clone(), RHO0),
        &Field::new(grid.clone(), Vector::<D>::zero()),
        &gamma,
    );

    // --- t = 0 validation: exactly on-shell (up to roundoff)
    let bssn = BssnState::from_adm(&adm, &config, &values);
    let a0 = bssn.to_adm();
    let rho0 = dust.rest_density(&a0.gamma);
    let zero_mom = Field::new(grid.clone(), Vector::<D>::zero());
    let r0 = residuals(&a0.gamma, &a0.k, &rho0, &zero_mom, &config, &values);
    assert!(
        r0.hamiltonian_max < 1.0e-12,
        "initial H must vanish, got {}",
        r0.hamiltonian_max
    );
    assert_eq!(r0.momentum_max, 0.0);

    // --- evolve geometry and dust together, monitoring constraints
    let dt = 0.5 * H;
    let mut bstate = bssn;
    let mut dstate = dust;
    let mut total0 = 0.0f64;
    dstate.grid.for_each_index(|idx| {
        total0 += dstate.densitized_rest_mass.get(idx);
    });
    total0 *= H.powi(3);

    let mut out =
        File::create("coupled_dust.csv").expect("cannot create output csv");
    writeln!(
        out,
        "step,t,dt,min_alpha,max_abs_K,rho_mean,rho_min,max_abs_v,Hmax,Mmax,mass_rel_err"
    )
    .expect("cannot write csv header");
    println!("phase A: coupled FLRW dust");
    println!("step | t     | min_alpha | max|K|   | rho_mean | rho_min  | max|v|   | |H|max  | |M|max  | mass_err");

    for step in 0..=STEPS {
        let a = bstate.to_adm();
        let rho = dstate.rest_density(&a.gamma);
        let vel = dstate.velocity(&a.gamma);
        let r = residuals(
            &a.gamma,
            &a.k,
            &rho,
            &zero_mom,
            &config,
            &values,
        );
        let mut min_alpha = f64::INFINITY;
        let mut max_k = 0.0f64;
        let mut rho_sum = 0.0f64;
        let mut rho_min = f64::INFINITY;
        let mut max_v = 0.0f64;
        let mut n = 0usize;
        grid.for_each_index(|idx| {
            min_alpha = min_alpha.min(*bstate.lapse.get(idx));
            max_k = max_k.max(bstate.trace_k.get(idx).abs());
            let rhov = *rho.get(idx);
            rho_sum += rhov;
            rho_min = rho_min.min(rhov);
            let vv = vel.get(idx);
            for i in 0..D {
                max_v = max_v.max(vv[i].abs());
            }
            n += 1;
        });
        let mut total = 0.0f64;
        grid.for_each_index(|idx| {
            total += dstate.densitized_rest_mass.get(idx);
        });
        total *= H.powi(3);
        let mass_err = (total - total0).abs() / total0;
        let rho_mean = rho_sum / n as f64;

        assert!(
            min_alpha.is_finite() && max_k.is_finite(),
            "non-finite state at step {step}"
        );
        assert!(min_alpha > 0.0, "lapse went non-positive at step {step}");
        assert!(mass_err == 0.0, "conserved mass drifted at step {step}");
        assert!(rho_min >= 0.0, "negative density at step {step}");

        writeln!(
            out,
            "{step},{:.6},{dt:.6e},{min_alpha:.6e},{max_k:.6e},{rho_mean:.6e},{rho_min:.6e},{max_v:.6e},{:.3e},{:.3e},{mass_err:.3e}",
            step as f64 * dt,
            r.hamiltonian_max,
            r.momentum_max,
        )
        .expect("cannot write csv row");
        println!(
            "{step:>4} | {t:5.3} | {min_alpha:.3e} | {max_k:.3e} | {rho_mean:.3e} | {rho_min:.3e} | {max_v:.3e} | {h:.3e} | {m:.3e} | {mass_err:.3e}",
            t = step as f64 * dt,
            h = r.hamiltonian_max,
            m = r.momentum_max,
        );

        if step < STEPS {
            let (nb, nd) = bstate
                .step_rk4_with_dust(&dstate, dt, &config, &values)
                .expect("dust velocity went invalid");
            bstate = nb;
            dstate = nd;
        }
    }

    let a = bstate.to_adm();
    let rho = dstate.rest_density(&a.gamma);
    let r = residuals(
        &a.gamma,
        &a.k,
        &rho,
        &zero_mom,
        &config,
        &values,
    );
    assert!(
        r.hamiltonian_max < 1.0e-9,
        "constraint blow-up over the run: |H| = {}",
        r.hamiltonian_max
    );
    println!("phase A done: diagnostics saved to coupled_dust.csv");
}

fn phase_b_transport() {
    // Gaussian bump advected at v = 0.2 on frozen Minkowski space.
    // Prescribed-background test: the geometry never responds, so H sits at
    // its off-shell value -16 pi rho by construction; what is validated is
    // rigid translation, velocity preservation, and mass conservation.
    let n = 32;
    let h = 1.0 / 32.0;
    let g = Arc::new(Grid::new([n; D], [h; D], [0.5; D]));
    let config = BoundaryConfig::<D>::periodic();
    let values = BoundaryValues::<D>::zeros();
    let sigma = 0.12;
    let amp = 1.0e-3;
    let vx = 0.2;

    let flat = {
        let mut identity = Tensor2::<D>::zero();
        for i in 0..D {
            identity[(i, i)] = 1.0;
        }
        Field::new(g.clone(), identity)
    };
    let lapse = Field::new(g.clone(), 1.0);
    let shift = Field::new(g.clone(), Vector::<D>::zero());

    let mut rho = Field::new(g.clone(), 0.0);
    g.for_each_index(|idx| {
        let c = g.coords(idx);
        let r2 =
            (c[0] - 1.0).powi(2) + (c[1] - 1.0).powi(2) + (c[2] - 1.0).powi(2);
        rho.set(idx, amp * (-r2 / (2.0 * sigma * sigma)).exp());
    });
    let mut vel_vec = Vector::<D>::zero();
    vel_vec[0] = vx;
    let dust = DustState::from_primitives(&rho, &Field::new(g.clone(), vel_vec), &flat);

    let dt = 0.5 * h;
    let steps = (0.5 / dt).round() as usize;
    let mut state = dust;
    for _ in 0..steps {
        state = dust_step_rk4(&state, dt, &lapse, &shift, &flat, &config, &values);
    }
    let t = steps as f64 * dt;

    let mut out = File::create("coupled_dust_transport.csv")
        .expect("cannot create output csv");
    writeln!(out, "t,profile_err,mass_rel_err,Hmax,max_abs_v")
        .expect("cannot write csv header");

    let w = 1.0 / (1.0 - vx * vx).sqrt();
    let mut worst = 0.0f64;
    let mut total = 0.0f64;
    let mut total0 = 0.0f64;
    let vel = state.velocity(&flat);
    let mut max_v_err = 0.0f64;
    g.for_each_index(|idx| {
        let c = g.coords(idx);
        let r2 = (c[0] - 1.0 - vx * t).powi(2)
            + (c[1] - 1.0).powi(2)
            + (c[2] - 1.0).powi(2);
        let want = amp * w * (-r2 / (2.0 * sigma * sigma)).exp();
        worst = worst.max((state.densitized_rest_mass.get(idx) - want).abs());
        total += state.densitized_rest_mass.get(idx);
        let v = vel.get(idx);
        max_v_err = max_v_err.max((v[0] - vx).abs());
        max_v_err = max_v_err.max(v[1].abs());
        max_v_err = max_v_err.max(v[2].abs());
    });
    // initial total recomputed from the analytic profile
    g.for_each_index(|idx| {
        let c = g.coords(idx);
        let r2 =
            (c[0] - 1.0).powi(2) + (c[1] - 1.0).powi(2) + (c[2] - 1.0).powi(2);
        total0 += amp * w * (-r2 / (2.0 * sigma * sigma)).exp();
    });
    let mass_err = ((total - total0) / total0).abs();

    let rho_now = state.rest_density(&flat);
    let r = residuals(
        &flat,
        &Field::new(g.clone(), Tensor2::<D>::zero()),
        &rho_now,
        &Field::new(g.clone(), Vector::<D>::zero()),
        &config,
        &values,
    );
    // H is off-shell here by construction (frozen flat geometry under a
    // moving bump): it must sit at -16 pi rho, i.e. order 16 pi amp.
    println!("phase B: flat transport, v = {vx}");
    println!("  profile_err = {worst:.3e}, mass_rel_err = {mass_err:.3e}");
    println!(
        "  Hmax = {:.3e} (off-shell, ~16 pi amp = {:.3e}), max|v - v0| = {max_v_err:.3e}",
        r.hamiltonian_max,
        16.0 * std::f64::consts::PI * amp,
    );
    assert!(worst < 1.0e-4, "transport profile mismatch {worst}");
    assert!(mass_err < 1.0e-10, "transport mass drift {mass_err}");
    assert!(max_v_err < 1.0e-2, "transport velocity drift {max_v_err}");
    writeln!(
        out,
        "{t:.6},{worst:.6e},{mass_err:.3e},{:.3e},{max_v_err:.3e}",
        r.hamiltonian_max,
    )
    .expect("cannot write csv row");
    println!("phase B done: diagnostics saved to coupled_dust_transport.csv");
}
