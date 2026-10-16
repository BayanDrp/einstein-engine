use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::physics::dust::DustState;
use einstein_engine::physics::dust_evolution::{dust_rhs, dust_step_rk4};
use einstein_engine::relativity::constraints::{hamiltonian_constraint, momentum_constraint};
use std::sync::Arc;

const D: usize = 3;
const C: f64 = 0.1;

fn grid(n: usize, h: f64) -> Arc<Grid<D>> {
    Arc::new(Grid::new([n; D], [h; D], [0.5; D]))
}

fn flat_metric(g: &Arc<Grid<D>>) -> Field<Tensor2<D>, D> {
    let mut identity = Tensor2::<D>::zero();
    for i in 0..D {
        identity[(i, i)] = 1.0;
    }
    Field::new(g.clone(), identity)
}

fn unit_lapse(g: &Arc<Grid<D>>) -> Field<f64, D> {
    Field::new(g.clone(), 1.0)
}

fn zero_shift(g: &Arc<Grid<D>>) -> Field<Vector<D>, D> {
    Field::new(g.clone(), Vector::<D>::zero())
}

fn periodic() -> (BoundaryConfig<D>, BoundaryValues<D>) {
    (BoundaryConfig::<D>::periodic(), BoundaryValues::<D>::zeros())
}

// Conserved Dtilde = sqrt(det gamma) rho W round-trips through the
// primitive rest density on a non-flat metric with nonzero velocity.
#[test]
fn conserved_roundtrips() {
    let g = grid(6, 0.2);
    let mut diag = Tensor2::<D>::zero();
    diag[(0, 0)] = 1.1;
    diag[(1, 1)] = 0.9;
    diag[(2, 2)] = 1.0;
    let metric = Field::new(g.clone(), diag);

    let mut vel_vec = Vector::<D>::zero();
    vel_vec[0] = 0.3;
    let vel = Field::new(g.clone(), vel_vec);
    let rho_in = Field::new(g.clone(), 2.0);

    let dust = DustState::from_primitives(&rho_in, &vel, &metric);
    let rho_out = dust.rest_density(&metric);
    let vel_out = dust.velocity(&metric);

    let mut worst = 0.0f64;
    g.for_each_index(|idx| {
        worst = worst.max((rho_out.get(idx) - 2.0).abs());
        worst = worst.max((vel_out.get(idx)[0] - 0.3).abs());
        worst = worst.max(vel_out.get(idx)[1].abs());
        worst = worst.max(vel_out.get(idx)[2].abs());
    });
    assert!(worst < 1.0e-12, "roundtrip mismatch {worst}");
}

// Uniform dust at rest on flat space has exactly zero RHS: all fluxes
// vanish pointwise, and every geometric source needs a gradient.
#[test]
fn homogeneous_rhs_vanishes() {
    let (config, values) = periodic();
    let g = grid(6, 0.2);
    let rho_in = Field::new(g.clone(), 1.5);
    let vel = Field::new(g.clone(), Vector::<D>::zero());
    let dust = DustState::from_primitives(&rho_in, &vel, &flat_metric(&g));

    let r = dust_rhs(&dust, 
        &unit_lapse(&g),
        &zero_shift(&g),
        &flat_metric(&g),
        &config,
        &values,
    );

    let mut worst_c = 0.0f64;
    let mut worst_v = 0.0f64;
    g.for_each_index(|idx| {
        worst_c = worst_c.max(r.densitized_rest_mass.get(idx).abs());
        let vv = r.densitized_momentum.get(idx);
        for i in 0..D {
            worst_v = worst_v.max(vv[i].abs());
        }
    });
    assert_eq!(worst_c, 0.0);
    assert_eq!(worst_v, 0.0);
}

// A Gaussian bump advected at constant velocity must translate rigidly:
// after t = 0.5 at v = 0.2 the profile matches the shifted analytic
// Gaussian, and the total conserved mass is unchanged to roundoff.
#[test]
fn bump_advects_at_velocity() {
    let (config, values) = periodic();
    let n = 32;
    let h = 1.0 / 32.0;
    let g = Arc::new(Grid::new([n; D], [h; D], [0.5; D]));
    let sigma = 0.12;

    let mut rho = Field::new(g.clone(), 0.0);
    g.for_each_index(|idx| {
        let c = g.coords(idx);
        let r2 = (c[0] - 1.0).powi(2) + (c[1] - 1.0).powi(2) + (c[2] - 1.0).powi(2);
        rho.set(idx, (-r2 / (2.0 * sigma * sigma)).exp());
    });
    let mut vel_vec = Vector::<D>::zero();
    vel_vec[0] = 0.2;
    let w = 1.0 / (1.0 - 0.2f64.powi(2)).sqrt();
    let dust = DustState::from_primitives(
        &rho,
        &Field::new(g.clone(), vel_vec),
        &flat_metric(&g),
    );

    let mut total0 = 0.0f64;
    g.for_each_index(|idx| {
        total0 += dust.densitized_rest_mass.get(idx);
    });

    let dt = 0.5 * h;
    let steps = (0.5 / dt).round() as usize;
    let mut state = dust;
    for _ in 0..steps {
        state = dust_step_rk4(&state, 
            dt,
            &unit_lapse(&g),
            &zero_shift(&g),
            &flat_metric(&g),
            &config,
            &values,
        );
    }
    let t = steps as f64 * dt;

    // total conserved mass: flux form on a periodic grid telescopes
    let mut total1 = 0.0f64;
    g.for_each_index(|idx| {
        total1 += state.densitized_rest_mass.get(idx);
    });
    assert!(
        ((total1 - total0) / total0).abs() < 1.0e-10,
        "mass drifted: {total0} -> {total1}"
    );

    // translated profile (conserved density carries one Lorentz factor)
    let mut worst = 0.0f64;
    g.for_each_index(|idx| {
        let c = g.coords(idx);
        let r2 = (c[0] - 1.0 - 0.2 * t).powi(2) + (c[1] - 1.0).powi(2) + (c[2] - 1.0).powi(2);
        let want = w * (-r2 / (2.0 * sigma * sigma)).exp();
        worst = worst.max((state.densitized_rest_mass.get(idx) - want).abs());
    });
    assert!(worst < 3.0e-2, "advected profile mismatch {worst}");

    // recovered velocity stays at the transport speed: momentum and
    // density disperse together, so their ratio is cleaner than either
    let vel_out = state.velocity(&flat_metric(&g));
    let mut worst_v = 0.0f64;
    g.for_each_index(|idx| {
        let v = vel_out.get(idx);
        worst_v = worst_v.max((v[0] - 0.2).abs());
        worst_v = worst_v.max(v[1].abs());
        worst_v = worst_v.max(v[2].abs());
    });
    assert!(worst_v < 1.0e-2, "transport velocity drifted {worst_v}");
}

// FLRW-patch data (flat metric, K_ij = C delta_ij, uniform dust with
// 6 C^2 = 16 pi rho) satisfies both constraints: H vanishes to roundoff
// and M vanishes exactly.
#[test]
fn flrw_dust_initial_data_is_on_shell() {
    let (config, values) = periodic();
    let g = grid(8, 0.125);
    let rho0 = 6.0 * C * C / (16.0 * std::f64::consts::PI);

    let mut k0 = Tensor2::<D>::zero();
    for i in 0..D {
        for j in 0..D {
            k0[(i, j)] = if i == j { C } else { 0.0 };
        }
    }

    let h = hamiltonian_constraint(
        &flat_metric(&g),
        &Field::new(g.clone(), k0),
        &Field::new(g.clone(), rho0),
        &config,
        &values,
    );
    let m = momentum_constraint(
        &flat_metric(&g),
        &Field::new(g.clone(), k0),
        &zero_shift(&g),
        &config,
        &values,
    );

    let mut worst_h = 0.0f64;
    let mut worst_m = 0.0f64;
    g.for_each_index(|idx| {
        worst_h = worst_h.max(h.get(idx).abs());
        let mv = m.get(idx);
        for i in 0..D {
            worst_m = worst_m.max(mv[i].abs());
        }
    });
    assert!(worst_h < 1.0e-12, "H = {worst_h}");
    assert_eq!(worst_m, 0.0);
}

// Dust at rest in a static curved lapse feels d(Ptilde_j)/dt = -rho d_j
// alpha: attraction toward small alpha with the Newtonian sign. A uniform
// shift drops out because only its derivatives enter the sources, so the
// beta = 0 and beta = const runs agree bit-for-bit.
#[test]
fn lapse_gradient_accelerates_dust() {
    let (config, values) = periodic();
    let n = 32;
    let h = 1.0 / 32.0;
    let g = Arc::new(Grid::new([n; D], [h; D], [0.5; D]));

    let mut lapse = Field::new(g.clone(), 0.0);
    g.for_each_index(|idx| {
        let x = g.coords(idx)[0];
        lapse.set(idx, 1.0 + 0.05 * (2.0 * std::f64::consts::PI * x).cos());
    });

    let rho_in = Field::new(g.clone(), 1.0);
    let vel = Field::new(g.clone(), Vector::<D>::zero());
    let dust = DustState::from_primitives(&rho_in, &vel, &flat_metric(&g));

    let run = |shift: &Field<Vector<D>, D>| {
        let r = dust_rhs(&dust, 
            &lapse,
            shift,
            &flat_metric(&g),
            &config,
            &values,
        );
        let mut worst = 0.0f64;
        g.for_each_index(|idx| {
            let x = g.coords(idx)[0];
            // -rho d_x alpha = +2 pi * 0.05 sin(2 pi x)
            let want = 2.0 * std::f64::consts::PI * 0.05
                * (2.0 * std::f64::consts::PI * x).sin();
            let got = r.densitized_momentum.get(idx);
            worst = worst.max((got[0] - want).abs());
            worst = worst.max(got[1].abs());
            worst = worst.max(got[2].abs());
        });
        worst
    };

    let plain = run(&zero_shift(&g));
    assert!(plain < 1.0e-2, "lapse force mismatch {plain}");

    let mut beta_vec = Vector::<D>::zero();
    beta_vec[0] = 0.1;
    let shifted = run(&Field::new(g.clone(), beta_vec));
    assert!(shifted < 1.0e-2, "lapse force mismatch with shift {shifted}");

    // uniform shift is invisible to the sources: identical RHS bit-for-bit
    let r0 = dust_rhs(&dust, &lapse, &zero_shift(&g), &flat_metric(&g), &config, &values);
    let r1 = dust_rhs(&dust, 
        &lapse,
        &Field::new(g.clone(), beta_vec),
        &flat_metric(&g),
        &config,
        &values,
    );
    let mut worst = 0.0f64;
    g.for_each_index(|idx| {
        for i in 0..D {
            let a = r0.densitized_momentum.get(idx)[i];
            let b = r1.densitized_momentum.get(idx)[i];
            if a != b {
                worst = worst.max((a - b).abs());
            }
        }
    });
    assert_eq!(worst, 0.0);
}

// Vacuum floor: non-positive conserved density recovers rho = 0, v = 0
// instead of NaNs.
#[test]
fn vacuum_floor_recovers_zeros() {
    let g = grid(4, 0.25);
    let mut mom_vec = Vector::<D>::zero();
    mom_vec[0] = 1.0;
    let dust = DustState {
        grid: g.clone(),
        densitized_rest_mass: Field::new(g.clone(), -0.5),
        densitized_momentum: Field::new(g.clone(), mom_vec),
    };

    let rho = dust.rest_density(&flat_metric(&g));
    let vel = dust.velocity(&flat_metric(&g));
    g.for_each_index(|idx| {
        assert_eq!(*rho.get(idx), 0.0);
        let v = vel.get(idx);
        for i in 0..D {
            assert_eq!(v[i], 0.0);
        }
    });
}
