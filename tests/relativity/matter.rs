use einstein_engine::geometry::tensor::{invert, Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::numerics::rk4::AdmState;
use einstein_engine::physics::stress_energy::{
    dust, dust_stress_energy_field, StressEnergy, StressEnergyField,
};
use einstein_engine::relativity::bssn::BssnState;
use einstein_engine::relativity::constraints::{hamiltonian_constraint, momentum_constraint};
use einstein_engine::relativity::matter::{four_velocity_covariant, matter_sources, MatterSources};
use std::sync::Arc;

const D: usize = 3;

fn grid() -> Arc<Grid<D>> {
    Arc::new(Grid::new([4; D], [0.25; D], [0.5; D]))
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

fn zero_curvature(g: &Arc<Grid<D>>) -> Field<Tensor2<D>, D> {
    Field::new(g.clone(), Tensor2::<D>::zero())
}

fn periodic() -> (BoundaryConfig<D>, BoundaryValues<D>) {
    (BoundaryConfig::<D>::periodic(), BoundaryValues::<D>::zeros())
}

fn matrix(t: &Tensor2<D>) -> [[f64; D]; D] {
    let mut m = [[0.0; D]; D];
    for i in 0..D {
        for j in 0..D {
            m[i][j] = t[(i, j)];
        }
    }
    m
}

fn max_diff_scalar(a: &Field<f64, D>, b: &Field<f64, D>) -> f64 {
    let mut worst = 0.0f64;
    a.grid.for_each_index(|idx| {
        worst = worst.max((a.get(idx) - b.get(idx)).abs());
    });
    worst
}

fn max_diff_tensor(a: &Field<Tensor2<D>, D>, b: &Field<Tensor2<D>, D>) -> f64 {
    let mut worst = 0.0f64;
    a.grid.for_each_index(|idx| {
        let x = a.get(idx);
        let y = b.get(idx);
        for i in 0..D {
            for j in 0..D {
                worst = worst.max((x[(i, j)] - y[(i, j)]).abs());
            }
        }
    });
    worst
}

fn max_diff_vector(a: &Field<Vector<D>, D>, b: &Field<Vector<D>, D>) -> f64 {
    let mut worst = 0.0f64;
    a.grid.for_each_index(|idx| {
        let x = a.get(idx);
        let y = b.get(idx);
        for i in 0..D {
            worst = worst.max((x[i] - y[i]).abs());
        }
    });
    worst
}

// Small-amplitude, smooth, periodic data off flat space with a nonzero
// shift, mirroring the BSSN consistency tests.
fn rippled(grid: Arc<Grid<D>>) -> AdmState<D> {
    let eps = 1.0e-2;
    let mut gamma = Field::new(grid.clone(), Tensor2::zero());
    let mut k = Field::new(grid.clone(), Tensor2::zero());
    let mut lapse = Field::new(grid.clone(), 0.0);
    let mut shift = Field::new(grid.clone(), Vector::zero());

    grid.for_each_index(|idx| {
        let c = grid.coords(idx);
        let (x, y, z) = (c[0], c[1], c[2]);

        let mut g = Tensor2::zero();
        g[(0, 0)] = 1.0 + eps * (2.0 * std::f64::consts::PI * x).cos();
        g[(1, 1)] = 1.0 + eps * (2.0 * std::f64::consts::PI * y).cos();
        g[(2, 2)] = 1.0 + eps * (2.0 * std::f64::consts::PI * z).cos();
        g[(0, 1)] = eps * (2.0 * std::f64::consts::PI * x).sin()
            * (2.0 * std::f64::consts::PI * y).sin();
        g[(1, 0)] = g[(0, 1)];

        let mut kk = Tensor2::zero();
        kk[(0, 0)] = 0.5 * eps * (2.0 * std::f64::consts::PI * x).cos();
        kk[(1, 1)] = 0.3 * eps * (2.0 * std::f64::consts::PI * y).cos();
        kk[(2, 2)] = 0.2 * eps * (2.0 * std::f64::consts::PI * z).cos();
        kk[(0, 1)] = 0.4 * eps * (2.0 * std::f64::consts::PI * y).sin();
        kk[(1, 0)] = kk[(0, 1)];

        lapse.set(idx, 1.0 + eps * (2.0 * std::f64::consts::PI * x).cos());

        let mut b = Vector::zero();
        b[0] = eps * (2.0 * std::f64::consts::PI * y).sin();
        b[1] = eps * (2.0 * std::f64::consts::PI * z).cos();
        b[2] = -eps * (2.0 * std::f64::consts::PI * x).sin();
        shift.set(idx, b);

        gamma.set(idx, g);
        k.set(idx, kk);
    });

    AdmState {
        grid: grid.clone(),
        gamma,
        k,
        lapse,
        shift,
    }
}

// Small, constant, symmetric T_mu_nu with every independent slot filled, so
// each source contribution (rho, S^i, S_ij, S) is independently nonzero.
fn small_stress(grid: &Arc<Grid<D>>) -> StressEnergyField<4, D> {
    let mut t = StressEnergy::<4>::zero();
    t.data[0][0] = 1.0e-3;
    t.data[0][1] = 2.0e-4;
    t.data[1][0] = 2.0e-4;
    t.data[0][2] = -1.0e-4;
    t.data[2][0] = -1.0e-4;
    t.data[1][1] = 5.0e-4;
    t.data[2][2] = 3.0e-4;
    t.data[3][3] = 2.0e-4;
    t.data[1][2] = 1.0e-4;
    t.data[2][1] = 1.0e-4;
    Field::new(grid.clone(), t)
}

// Flat space, unit lapse, zero shift, v = 0.6 along x: W = 1/sqrt(1 - 0.36)
// = 1.25, so u_0 = -1.25 and u_i = (0.75, 0, 0).
#[test]
fn minkowski_four_velocity() {
    let alpha = 1.0;
    let beta = Vector::<D>::zero();

    let mut spatial_metric = Tensor2::<D>::zero();
    for i in 0..D {
        spatial_metric[(i, i)] = 1.0;
    }

    let mut velocity = Vector::<D>::zero();
    velocity[0] = 0.6;

    let (u_0, u_cov) =
        four_velocity_covariant(alpha, &beta, &spatial_metric, &velocity).unwrap();

    assert!((u_0 + 1.25).abs() < 1e-12);
    assert!((u_cov[0] - 0.75).abs() < 1e-12);
    assert!(u_cov[1].abs() < 1e-12);
    assert!(u_cov[2].abs() < 1e-12);
}

// Dust at rest on Minkowski: T_00 = rho and nothing else, so E = rho and
// S^i, S_ij, S all vanish exactly.
#[test]
fn matter_sources_minkowski_dust_at_rest() {
    let g = grid();
    let rho0 = 2.5;
    let t = dust::<4, D>(&Field::new(g.clone(), rho0));
    let src = matter_sources::<4, D>(&t, &unit_lapse(&g), &zero_shift(&g), &flat_metric(&g));

    g.for_each_index(|idx| {
        assert_eq!(*src.rho.get(idx), rho0);
        let m = src.momentum.get(idx);
        let s = src.stress.get(idx);
        for i in 0..D {
            assert_eq!(m[i], 0.0);
            for j in 0..D {
                assert_eq!(s[(i, j)], 0.0);
            }
        }
        assert_eq!(*src.stress_trace.get(idx), 0.0);
    });
}

// Moving dust, v = 0.6 along x, rest density 2: W = 1.25 gives
// E = 2 W^2 = 3.125, S^x = 2 W^2 v = 1.875, S_xx = 2 (W v)^2 = 1.125.
#[test]
fn matter_sources_moving_dust_values() {
    let g = grid();
    let rho_field = Field::new(g.clone(), 2.0);
    let u_0 = Field::new(g.clone(), -1.25);
    let mut u_vec = Vector::<D>::zero();
    u_vec[0] = 0.75;
    let u_cov = Field::new(g.clone(), u_vec);

    let t = dust_stress_energy_field::<4, D>(&rho_field, &u_0, &u_cov);
    let src = matter_sources::<4, D>(&t, &unit_lapse(&g), &zero_shift(&g), &flat_metric(&g));

    g.for_each_index(|idx| {
        assert!((src.rho.get(idx) - 3.125).abs() < 1e-12);
        let m = src.momentum.get(idx);
        assert!((m[0] - 1.875).abs() < 1e-12);
        assert!(m[1].abs() < 1e-12);
        assert!(m[2].abs() < 1e-12);
        let s = src.stress.get(idx);
        assert!((s[(0, 0)] - 1.125).abs() < 1e-12);
        for i in 0..D {
            for j in 0..D {
                if i != 0 || j != 0 {
                    assert!(s[(i, j)].abs() < 1e-12);
                }
            }
        }
        assert!((src.stress_trace.get(idx) - 1.125).abs() < 1e-12);
    });
}

// Same physical state with a nonzero shift beta = (0.1, 0, 0): E, S^i and
// S are scalars built to be shift-covariant, so the identical numbers
// 3.125 / 1.875 / 1.125 must come out. This pins the beta^i T_0i and
// beta^i beta^j T_ij terms, which vanish in every zero-shift test.
#[test]
fn matter_sources_shift_terms_cancel() {
    let g = grid();
    let mut beta_vec = Vector::<D>::zero();
    beta_vec[0] = 0.1;
    let shift = Field::new(g.clone(), beta_vec);
    let mut vel_vec = Vector::<D>::zero();
    vel_vec[0] = 0.6;
    let vel = Field::new(g.clone(), vel_vec);

    let (u_0, u_cov) =
        einstein_engine::relativity::matter::four_velocity_covariant_field(
            &unit_lapse(&g),
            &shift,
            &flat_metric(&g),
            &vel,
        )
        .unwrap();
    let t = dust_stress_energy_field::<4, D>(&Field::new(g.clone(), 2.0), &u_0, &u_cov);
    let src = matter_sources::<4, D>(&t, &unit_lapse(&g), &shift, &flat_metric(&g));

    g.for_each_index(|idx| {
        assert!((src.rho.get(idx) - 3.125).abs() < 1e-12);
        let m = src.momentum.get(idx);
        assert!((m[0] - 1.875).abs() < 1e-12);
        assert!(m[1].abs() < 1e-12);
        assert!(m[2].abs() < 1e-12);
        assert!((src.stress_trace.get(idx) - 1.125).abs() < 1e-12);
    });
}

// The trace must use the inverse metric: with gamma = diag(1.1, 0.9, 1.0)
// and S_ij = delta_ij, S = 1/1.1 + 1/0.9 + 1. Tracing with gamma itself
// would give 3.0 instead.
#[test]
fn stress_trace_uses_inverse_metric() {
    let g = grid();
    let mut diag = Tensor2::<D>::zero();
    diag[(0, 0)] = 1.1;
    diag[(1, 1)] = 0.9;
    diag[(2, 2)] = 1.0;
    let metric = Field::new(g.clone(), diag);

    let mut unit = StressEnergy::<4>::zero();
    for i in 0..D {
        unit.data[i + 1][i + 1] = 1.0;
    }
    let t = Field::new(g.clone(), unit);
    let src = matter_sources::<4, D>(&t, &unit_lapse(&g), &zero_shift(&g), &metric);

    let want = 1.0 / 1.1 + 1.0 / 0.9 + 1.0;
    g.for_each_index(|idx| {
        assert!((src.stress_trace.get(idx) - want).abs() < 1e-12);
        assert_eq!(*src.rho.get(idx), 0.0);
        let m = src.momentum.get(idx);
        for i in 0..D {
            assert_eq!(m[i], 0.0);
        }
    });
}

// E enters H: flat slice, K = 0, uniform dust rho = 2.5 gives
// H = -16 pi rho exactly, since every geometric term vanishes.
#[test]
fn hamiltonian_shifts_by_16_pi_rho() {
    let g = grid();
    let (config, values) = periodic();
    let t = dust::<4, D>(&Field::new(g.clone(), 2.5));
    let src = matter_sources::<4, D>(&t, &unit_lapse(&g), &zero_shift(&g), &flat_metric(&g));
    let h = hamiltonian_constraint(
        &flat_metric(&g),
        &zero_curvature(&g),
        &src.rho,
        &config,
        &values,
    );

    let want = -16.0 * std::f64::consts::PI * 2.5;
    g.for_each_index(|idx| {
        assert!((h.get(idx) - want).abs() < 1e-12);
    });
}

// S^i enters M^i: flat slice, K = 0, moving dust with S^x = 1.875 gives
// M^x = -8 pi S^x, since the divergence of zero vanishes.
#[test]
fn momentum_matches_8_pi_s() {
    let g = grid();
    let (config, values) = periodic();
    let rho_field = Field::new(g.clone(), 2.0);
    let u_0 = Field::new(g.clone(), -1.25);
    let mut u_vec = Vector::<D>::zero();
    u_vec[0] = 0.75;
    let u_cov = Field::new(g.clone(), u_vec);

    let t = dust_stress_energy_field::<4, D>(&rho_field, &u_0, &u_cov);
    let src = matter_sources::<4, D>(&t, &unit_lapse(&g), &zero_shift(&g), &flat_metric(&g));
    let m = momentum_constraint(
        &flat_metric(&g),
        &zero_curvature(&g),
        &src.momentum,
        &config,
        &values,
    );

    let want = -8.0 * std::f64::consts::PI * 1.875;
    g.for_each_index(|idx| {
        let mv = m.get(idx);
        assert!((mv[0] - want).abs() < 1e-12);
        assert!(mv[1].abs() < 1e-12);
        assert!(mv[2].abs() < 1e-12);
    });
}

// Zero matter reproduces the vacuum RHS bit-for-bit on both formulations:
// the _with_matter wrappers delegate with identically zero sources.
#[test]
fn zero_matter_reproduces_vacuum_rhs() {
    let (config, values) = periodic();
    let g = grid();
    let adm = rippled(g.clone());
    let vac = MatterSources::vacuum(&g);

    let ar = adm.rhs(&config, &values);
    let am = adm.rhs_with_matter(&vac, &config, &values);
    assert_eq!(max_diff_tensor(&ar.gamma, &am.gamma), 0.0);
    assert_eq!(max_diff_tensor(&ar.k, &am.k), 0.0);
    assert_eq!(max_diff_scalar(&ar.lapse, &am.lapse), 0.0);
    assert_eq!(max_diff_vector(&ar.shift, &am.shift), 0.0);

    let bssn = BssnState::from_adm(&adm, &config, &values);
    let br = bssn.rhs(&config, &values);
    let bm = bssn.rhs_with_matter(&vac, &config, &values);
    assert_eq!(max_diff_scalar(&br.phi, &bm.phi), 0.0);
    assert_eq!(max_diff_tensor(&br.conformal_metric, &bm.conformal_metric), 0.0);
    assert_eq!(max_diff_scalar(&br.trace_k, &bm.trace_k), 0.0);
    assert_eq!(max_diff_tensor(&br.conformal_a, &bm.conformal_a), 0.0);
    assert_eq!(max_diff_vector(&br.conformal_gamma, &bm.conformal_gamma), 0.0);
    assert_eq!(max_diff_scalar(&br.lapse, &bm.lapse), 0.0);
}

// The ADM matter increment is exactly +4 pi alpha ((S - rho) gamma_ij -
// 2 S_ij), recomputed independently here from the sources and the state;
// gamma, lapse and shift RHS are untouched by matter.
#[test]
fn adm_matter_increment_matches_analytic() {
    let (config, values) = periodic();
    let g = grid();
    let adm = rippled(g.clone());
    let src = matter_sources::<4, D>(
        &small_stress(&g),
        &adm.lapse,
        &adm.shift,
        &adm.gamma,
    );

    let r0 = adm.rhs(&config, &values);
    let r1 = adm.rhs_with_matter(&src, &config, &values);

    assert_eq!(max_diff_tensor(&r0.gamma, &r1.gamma), 0.0);
    assert_eq!(max_diff_scalar(&r0.lapse, &r1.lapse), 0.0);
    assert_eq!(max_diff_vector(&r0.shift, &r1.shift), 0.0);

    let mut worst = 0.0f64;
    g.for_each_index(|idx| {
        let alpha = *adm.lapse.get(idx);
        let gamma = adm.gamma.get(idx);
        let rho = *src.rho.get(idx);
        let s = *src.stress_trace.get(idx);
        let sij = src.stress.get(idx);
        let k0 = r0.k.get(idx);
        let k1 = r1.k.get(idx);
        for i in 0..D {
            for j in 0..D {
                let want = alpha * 4.0 * std::f64::consts::PI
                    * ((s - rho) * gamma[(i, j)] - 2.0 * sij[(i, j)]);
                worst = worst.max((k1[(i, j)] - k0[(i, j)] - want).abs());
            }
        }
    });
    assert!(worst < 1.0e-12, "ADM matter increment mismatch {worst}");
}

// The BSSN matter increments are exactly +4 pi alpha (rho + s) on d_t K,
// -e^{-4 phi} alpha 8 pi (S_ij - gamma_ij s/3) on d_t A tilde, and
// -16 pi alpha gtilde^ij s_j on d_t Gamma^i; phi, gtilde and lapse RHS
// are untouched, and the A tilde increment is symmetric.
#[test]
fn bssn_matter_increments_match_analytic() {
    let (config, values) = periodic();
    let g = grid();
    let adm = rippled(g.clone());
    let src = matter_sources::<4, D>(
        &small_stress(&g),
        &adm.lapse,
        &adm.shift,
        &adm.gamma,
    );
    let bssn = BssnState::from_adm(&adm, &config, &values);

    let r0 = bssn.rhs(&config, &values);
    let r1 = bssn.rhs_with_matter(&src, &config, &values);

    assert_eq!(max_diff_scalar(&r0.phi, &r1.phi), 0.0);
    assert_eq!(max_diff_tensor(&r0.conformal_metric, &r1.conformal_metric), 0.0);
    assert_eq!(max_diff_scalar(&r0.lapse, &r1.lapse), 0.0);

    let pi = std::f64::consts::PI;
    let mut worst_k = 0.0f64;
    let mut worst_a = 0.0f64;
    let mut worst_g = 0.0f64;
    let mut worst_sym = 0.0f64;
    g.for_each_index(|idx| {
        let alpha = *bssn.lapse.get(idx);
        let phi = *bssn.phi.get(idx);
        let e4 = (4.0 * phi).exp();
        let em4 = 1.0 / e4;
        let gtilde = *bssn.conformal_metric.get(idx);
        let ginvt = invert(&matrix(&gtilde));
        let rho = *src.rho.get(idx);
        let s = *src.stress_trace.get(idx);
        let sij = src.stress.get(idx);
        let mom = src.momentum.get(idx);

        // d_t K increment
        let dk = r1.trace_k.get(idx) - r0.trace_k.get(idx);
        worst_k = worst_k.max((dk - alpha * 4.0 * pi * (rho + s)).abs());

        // d_t A tilde increment, with physical gamma_ij = e^{4 phi} gtilde_ij
        let a0 = r0.conformal_a.get(idx);
        let a1 = r1.conformal_a.get(idx);
        for i in 0..D {
            for j in 0..D {
                let want = -em4 * alpha * 8.0 * pi
                    * (sij[(i, j)] - e4 * gtilde[(i, j)] * s / 3.0);
                worst_a = worst_a.max((a1[(i, j)] - a0[(i, j)] - want).abs());
                worst_sym = worst_sym.max(
                    ((a1[(i, j)] - a0[(i, j)]) - (a1[(j, i)] - a0[(j, i)])).abs(),
                );
            }
        }

        // d_t Gamma^i increment, with s_j = gamma_jk S^k physical
        let mut s_cov = [0.0f64; D];
        for j in 0..D {
            for k in 0..D {
                s_cov[j] += e4 * gtilde[(j, k)] * mom[k];
            }
        }
        let g0 = r0.conformal_gamma.get(idx);
        let g1 = r1.conformal_gamma.get(idx);
        for i in 0..D {
            let mut want = 0.0;
            for j in 0..D {
                want += -16.0 * alpha * pi * ginvt[i][j] * s_cov[j];
            }
            worst_g = worst_g.max((g1[i] - g0[i] - want).abs());
        }
    });
    assert!(worst_k < 1.0e-12, "d_t K matter mismatch {worst_k}");
    assert!(worst_a < 1.0e-12, "d_t A matter mismatch {worst_a}");
    assert!(worst_g < 1.0e-12, "d_t Gamma matter mismatch {worst_g}");
    assert!(worst_sym < 1.0e-12, "A increment not symmetric {worst_sym}");
}

// The BSSN/ADM K identity survives matter: BSSN d_t K = ADM d_t K - alpha H,
// because both matter pieces reduce to the same on-shell combination.
#[test]
fn bssn_adm_k_identity_holds_with_matter() {
    let (config, values) = periodic();
    let g = grid();
    let mut adm = rippled(g.clone());
    adm.shift = Field::new(g.clone(), Vector::<D>::zero());
    let src = matter_sources::<4, D>(
        &small_stress(&g),
        &adm.lapse,
        &adm.shift,
        &adm.gamma,
    );

    let adm_rhs = adm.rhs_with_matter(&src, &config, &values);
    let bssn = BssnState::from_adm(&adm, &config, &values);
    let bssn_rhs = bssn.rhs_with_matter(&src, &config, &values);

    // H with the actual matter density: the identity needs H_matter,
    // not H_vacuum (they differ by exactly 16 pi rho).
    let h = hamiltonian_constraint(&adm.gamma, &adm.k, &src.rho, &config, &values);

    let mut worst = 0.0f64;
    g.for_each_index(|idx| {
        let gamma = adm.gamma.get(idx);
        let k = adm.k.get(idx);
        let ginv = invert(&matrix(&gamma));
        let mut k2 = 0.0;
        for a in 0..D {
            for b in 0..D {
                for c in 0..D {
                    for d in 0..D {
                        k2 += ginv[a][c] * ginv[b][d] * k[(c, d)] * k[(a, b)];
                    }
                }
            }
        }
        let mut adm_dtk = 2.0 * adm.lapse.get(idx) * k2;
        let kr = adm_rhs.k.get(idx);
        for i in 0..D {
            for j in 0..D {
                adm_dtk += ginv[i][j] * kr[(i, j)];
            }
        }
        let bssn_dtk = *bssn_rhs.trace_k.get(idx);
        let diff = (bssn_dtk - (adm_dtk - adm.lapse.get(idx) * h.get(idx))).abs();
        worst = worst.max(diff);
    });
    assert!(worst < 1.0e-4, "K identity with matter mismatch {worst}");
}
