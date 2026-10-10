use einstein_engine::geometry::christoffel::spatial_christoffel_symbols;
use einstein_engine::geometry::ricci::ricci_from_riemann;
use einstein_engine::geometry::riemann::riemann_from_christoffel;
use einstein_engine::geometry::tensor::{det, invert, Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::initial_data::schwarzschild::schwarzschild;
use einstein_engine::numerics::rk4::EvolutionState;
use einstein_engine::relativity::bssn::BssnState;
use einstein_engine::relativity::constraints::hamiltonian_constraint;
use std::sync::Arc;

const D: usize = 3;
const ORIGIN: [f64; D] = [0.5, 0.5, 0.5];

fn periodic() -> (BoundaryConfig<D>, BoundaryValues<D>) {
    (BoundaryConfig::<D>::periodic(), BoundaryValues::<D>::zeros())
}

fn grid(n: usize, h: f64) -> Arc<Grid<D>> {
    Arc::new(Grid::new([n; D], [h; D], ORIGIN))
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

fn max_abs_diff_tensor(a: &Field<Tensor2<D>, D>, b: &Field<Tensor2<D>, D>) -> f64 {
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

fn max_scalar_field(a: &Field<f64, D>) -> f64 {
    let mut worst = 0.0f64;
    a.grid.for_each_index(|idx| worst = worst.max(a.get(idx).abs()));
    worst
}

fn max_abs_tensor_field(a: &Field<Tensor2<D>, D>) -> f64 {
    let mut worst = 0.0f64;
    a.grid.for_each_index(|idx| {
        let x = a.get(idx);
        for i in 0..D {
            for j in 0..D {
                worst = worst.max(x[(i, j)].abs());
            }
        }
    });
    worst
}

fn max_abs_vector_field(a: &Field<Vector<D>, D>) -> f64 {
    let mut worst = 0.0f64;
    a.grid.for_each_index(|idx| {
        let x = a.get(idx);
        for i in 0..D {
            worst = worst.max(x[i].abs());
        }
    });
    worst
}

// Nontrivial, periodic, symmetric, positive-definite ADM data in every sense:
// non-flat metric, non-vanishing trace-full extrinsic curvature, curved lapse
// and a nonzero shift. Everything is period-matching so no stencil anywhere,
// including the faces, sees a jump.
fn admixtures(grid: Arc<Grid<D>>) -> EvolutionState<D> {
    let mut gamma = Field::new(grid.clone(), Tensor2::zero());
    let mut k = Field::new(grid.clone(), Tensor2::zero());
    let mut lapse = Field::new(grid.clone(), 0.0);
    let mut shift = Field::new(grid.clone(), Vector::zero());

    grid.for_each_index(|idx| {
        let c = grid.coords(idx);
        let (x, y, z) = (c[0], c[1], c[2]);

        let mut g = Tensor2::zero();
        for i in 0..D {
            g[(i, i)] = 1.0 + 0.1 * (2.0 * std::f64::consts::PI * c[i]).cos();
        }
        g[(0, 1)] = 0.05 * (2.0 * std::f64::consts::PI * x).sin() * (2.0 * std::f64::consts::PI * y).sin();
        g[(0, 2)] = 0.05 * (2.0 * std::f64::consts::PI * x).sin() * (2.0 * std::f64::consts::PI * z).sin();
        g[(1, 2)] = 0.05 * (2.0 * std::f64::consts::PI * y).sin() * (2.0 * std::f64::consts::PI * z).sin();
        g[(1, 0)] = g[(0, 1)];
        g[(2, 0)] = g[(0, 2)];
        g[(2, 1)] = g[(1, 2)];

        let mut kk = Tensor2::zero();
        for i in 0..D {
            kk[(i, i)] = 0.03 + 0.01 * (2.0 * std::f64::consts::PI * x).cos();
        }
        kk[(0, 1)] = 0.01 * (2.0 * std::f64::consts::PI * y).sin();
        kk[(1, 0)] = kk[(0, 1)];
        kk[(0, 2)] = -0.01 * (2.0 * std::f64::consts::PI * z).sin();
        kk[(2, 0)] = kk[(0, 2)];

        gamma.set(idx, g);
        k.set(idx, kk);
        lapse.set(idx, 1.0 + 0.05 * (2.0 * std::f64::consts::PI * x).cos());
        let mut b = Vector::zero();
        b[0] = 0.05 * (2.0 * std::f64::consts::PI * y).sin();
        b[1] = 0.05 * (2.0 * std::f64::consts::PI * z).cos();
        b[2] = -0.05 * (2.0 * std::f64::consts::PI * x).sin();
        shift.set(idx, b);
    });

    EvolutionState {
        grid: grid.clone(),
        gamma,
        k,
        lapse,
        shift,
    }
}

#[test]
fn minkowski_roundtrips_to_flat_adm_and_back() {
    let (config, values) = periodic();
    let bssn = BssnState::minkowski(grid(6, 0.25));

    let adm = bssn.to_adm();
    assert_eq!(max_scalar_field(&adm.lapse), 1.0);
    bssn.grid.for_each_index(|idx| {
        for i in 0..D {
            for j in 0..D {
                let want = if i == j { 1.0 } else { 0.0 };
                assert_eq!(adm.gamma.get(idx)[(i, j)], want);
                assert_eq!(adm.k.get(idx)[(i, j)], 0.0);
            }
        }
        assert_eq!(adm.shift.get(idx), &Vector::<D>::zero());
    });

    let back = BssnState::from_adm(&adm, &config, &values);
    assert_eq!(max_scalar_field(&back.phi), 0.0);
    back.grid.for_each_index(|idx| {
        for i in 0..D {
            for j in 0..D {
                let want = if i == j { 1.0 } else { 0.0 };
                assert_eq!(back.conformal_metric.get(idx)[(i, j)], want);
                assert_eq!(back.conformal_a.get(idx)[(i, j)], 0.0);
            }
            assert_eq!(back.conformal_gamma.get(idx)[i], 0.0);
        }
        assert_eq!(back.trace_k.get(idx), &0.0);
    });
}

// Isotropic Schwarzschild is conformally flat: gamma = psi^4 delta with
// psi = 1 + M/2r. Converting to BSSN must land on the conformal metric being
// exactly delta, phi = ln psi, vanishing A tilde, K and Gamma^i.
#[test]
fn schwarzschild_decomposes_into_conformal_flatness() {
    let (config, values) = periodic();
    let g = grid(10, 0.1);
    let admmass = 1.0;
    let adm = schwarzschild::<D>(g.clone(), admmass).unwrap();
    let bssn = BssnState::from_adm(&adm, &config, &values);

    g.for_each_index(|idx| {
        let c = g.coords(idx);
        let r = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        let psi = 1.0 + admmass / (2.0 * r);

        assert!((bssn.phi.get(idx) - psi.ln()).abs() < 1.0e-12);
        assert_eq!(bssn.trace_k.get(idx), &0.0);

        for i in 0..D {
            for j in 0..D {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((bssn.conformal_metric.get(idx)[(i, j)] - want).abs() < 1.0e-12);
                assert_eq!(bssn.conformal_a.get(idx)[(i, j)], 0.0);
            }
            assert!(bssn.conformal_gamma.get(idx)[i].abs() < 1.0e-13);
        }
    });
}

// ADM -> BSSN -> ADM must be the identity on the metric and curvature, and the
// algebraic constraints det(gtilde) = 1 and gtilde^ij A tilde_ij = 0 must hold
// pointwise after a conversion.
#[test]
fn arbitrary_adm_data_roundtrips_cleanly() {
    let (config, values) = periodic();
    let g = grid(8, 0.125);
    let adm = admixtures(g.clone());
    let bssn = BssnState::from_adm(&adm, &config, &values);
    let back = bssn.to_adm();

    assert!(max_abs_diff_tensor(&adm.gamma, &back.gamma) < 1.0e-12);
    assert!(max_abs_diff_tensor(&adm.k, &back.k) < 1.0e-12);

    let mut worst_lapse = 0.0f64;
    let mut worst_shift = 0.0f64;
    g.for_each_index(|idx| {
        worst_lapse = worst_lapse.max((adm.lapse.get(idx) - back.lapse.get(idx)).abs());
        let bs = back.shift.get(idx);
        let as_ = adm.shift.get(idx);
        for i in 0..D {
            worst_shift = worst_shift.max((bs[i] - as_[i]).abs());
        }
    });
    assert!(worst_lapse < 1.0e-12);
    assert!(worst_shift < 1.0e-12);

    g.for_each_index(|idx| {
        let gtilde = matrix(bssn.conformal_metric.get(idx));
        assert!((det(&gtilde) - 1.0).abs() < 1.0e-10);

        let ginvt = invert(&gtilde);
        let at = bssn.conformal_a.get(idx);
        let mut tr = 0.0;
        for i in 0..D {
            for j in 0..D {
                tr += ginvt[i][j] * at[(i, j)];
            }
        }
        assert!(tr.abs() < 1.0e-10, "gtilde^ij A tilde_ij = {tr}");
    });
}

// BSSN -> ADM -> BSSN is the identity provided det(gtilde) = 1 (which converts
// preserve); a conformal metric with unit determinant goes around exactly.
#[test]
fn bssn_to_adm_roundtrips() {
    let (config, values) = periodic();
    let g = grid(8, 0.125);
    let adm = admixtures(g);
    let bssn = BssnState::from_adm(&adm, &config, &values);
    let back = BssnState::from_adm(&bssn.to_adm(), &config, &values);

    let mut worst_phi = 0.0f64;
    bssn.grid.for_each_index(|idx| {
        worst_phi = worst_phi.max((bssn.phi.get(idx) - back.phi.get(idx)).abs());
    });
    assert!(worst_phi < 1.0e-12);
    assert!(max_abs_diff_tensor(&bssn.conformal_metric, &back.conformal_metric) < 1.0e-12);
    assert!(max_abs_diff_tensor(&bssn.conformal_a, &back.conformal_a) < 1.0e-12);
}

// Low-amplitude, smooth, periodic data off flat space. Both formulations then
// evolve a nearly linear problem, so agreement between ADM and BSSN is sharp
// instead of being buried under mesh truncation.
fn rippled(grid: Arc<Grid<D>>) -> EvolutionState<D> {
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
        g[(1, 2)] = eps * (2.0 * std::f64::consts::PI * y).sin()
            * (2.0 * std::f64::consts::PI * z).sin();
        g[(2, 1)] = g[(1, 2)];
        g[(0, 2)] = eps * (2.0 * std::f64::consts::PI * z).sin()
            * (2.0 * std::f64::consts::PI * x).sin();
        g[(2, 0)] = g[(0, 2)];

        let mut kk = Tensor2::zero();
        kk[(0, 0)] = 0.5 * eps * (2.0 * std::f64::consts::PI * x).cos();
        kk[(1, 1)] = 0.3 * eps * (2.0 * std::f64::consts::PI * y).cos();
        kk[(2, 2)] = 0.2 * eps * (2.0 * std::f64::consts::PI * z).cos();
        kk[(0, 1)] = 0.4 * eps * (2.0 * std::f64::consts::PI * y).sin();
        kk[(1, 0)] = kk[(0, 1)];
        kk[(0, 2)] = 0.3 * eps * (2.0 * std::f64::consts::PI * z).sin();
        kk[(2, 0)] = kk[(0, 2)];
        kk[(1, 2)] = 0.4 * eps * (2.0 * std::f64::consts::PI * x).sin();
        kk[(2, 1)] = kk[(1, 2)];

        lapse.set(idx, 1.0 + eps * (2.0 * std::f64::consts::PI * x).cos());

        let mut b = Vector::zero();
        b[0] = eps * (2.0 * std::f64::consts::PI * y).sin();
        b[1] = eps * (2.0 * std::f64::consts::PI * z).cos();
        b[2] = -eps * (2.0 * std::f64::consts::PI * x).sin();
        shift.set(idx, b);

        gamma.set(idx, g);
        k.set(idx, kk);
    });

    EvolutionState {
        grid: grid.clone(),
        gamma,
        k,
        lapse,
        shift,
    }
}

// Flat space is a fixed point of the BSSN system: every field's right-hand
// side vanishes exactly.
#[test]
fn bssn_rhs_minkowski_is_zero() {
    let (config, values) = periodic();
    let s = BssnState::minkowski(grid(8, 0.125));
    let r = s.rhs(&config, &values);

    assert_eq!(max_scalar_field(&r.phi), 0.0);
    assert_eq!(max_abs_tensor_field(&r.conformal_metric), 0.0);
    assert_eq!(max_scalar_field(&r.trace_k), 0.0);
    assert_eq!(max_abs_tensor_field(&r.conformal_a), 0.0);
    assert_eq!(max_abs_vector_field(&r.conformal_gamma), 0.0);
    assert_eq!(max_scalar_field(&r.lapse), 0.0);
    assert_eq!(max_abs_vector_field(&r.shift), 0.0);
}

// The conformal split of the physical Ricci tensor, R(gamma) = Rtilde + Rphi
// with gamma = e^{4 phi} gtilde, has to reproduce the Ricci tensor that the
// already-tested Christoffel -> Riemann -> Ricci pipeline computes directly
// from the physical metric. This pins down the product terms of Rtilde and
// every coefficient of Rphi on nontrivial data. The mismatch is pure mesh
// truncation, so it must converge at second order.
#[test]
fn conformal_ricci_split_reproduces_physical_ricci() {
    let (config, values) = periodic();

    let mismatch = |g: &Arc<Grid<D>>| -> f64 {
        let adm = admixtures(g.clone());
        let bssn = BssnState::from_adm(&adm, &config, &values);
        let (rt, rp) = bssn.conformal_ricci(&config, &values);

        let christoffel = spatial_christoffel_symbols(&adm.gamma, &config, &values);
        let riemann = riemann_from_christoffel::<D, D>(&christoffel, &config, &values);
        let rphys = ricci_from_riemann::<D, D>(&riemann);

        let mut worst = 0.0f64;
        g.for_each_index(|idx| {
            let (a, b) = (rt.get(idx), rp.get(idx));
            let physical = rphys.get(idx);
            for i in 0..D {
                for j in 0..D {
                    worst = worst.max((a.data[i][j] + b.data[i][j] - physical.data[i][j]).abs());
                }
            }
        });
        worst
    };

    let coarse = mismatch(&grid(16, 0.0625));
    let fine = mismatch(&grid(32, 0.03125));
    let ratio = coarse / fine;

    assert!(
        coarse > 0.0,
        "an exact zero would mean the test is not measuring truncation"
    );
    assert!(
        (ratio - 4.0).abs() < 1.0,
        "R split should converge at second order (ratio 4), got {ratio:.2}"
    );
    assert!(fine < 1.0e-2, "R split mismatch too large: {fine}");
}

// The BSSN and ADM systems describe the same physics. One RK4 step from
// identical data, taken on each side and mapped back through the conversion,
// must land on the same ADM state for gamma, lapse and shift -- those
// equations are exact reformulations with no constraint substitution. The K
// equation is different: BSSN's d_t K is ADM's d_t K with the Hamiltonian
// constraint substituted, d_t K_bssn = d_t K_adm - alpha * H, so that identity
// is checked directly. This exercises every BSSN evolution equation against
// the independently tested ADM equations.
#[test]
fn bssn_rhs_matches_adm_rhs_one_step() {
    let (config, values) = periodic();
    let g = grid(10, 0.1);
    let mut adm = rippled(g.clone());
    // the K identity below is cleanest with a frozen shift
    adm.shift = Field::new(g.clone(), Vector::zero());
    let dt = 1.0e-4;

    let adm_next = adm.step_rk4(dt, &config, &values);

    let bssn = BssnState::from_adm(&adm, &config, &values);
    let bssn_next = bssn.step_rk4(dt, &config, &values);
    let back = bssn_next.to_adm();

    assert!(
        max_abs_diff_tensor(&adm_next.gamma, &back.gamma) < 1.0e-6,
        "gamma mismatch {}",
        max_abs_diff_tensor(&adm_next.gamma, &back.gamma)
    );
    g.for_each_index(|idx| {
        let diff = (adm_next.lapse.get(idx) - back.lapse.get(idx)).abs();
        assert!(diff < 1.0e-6, "lapse mismatch {diff}");
        let bs = back.shift.get(idx);
        let as_ = adm_next.shift.get(idx);
        for i in 0..D {
            assert!((bs[i] - as_[i]).abs() < 1.0e-12, "shift mismatch");
        }
    });

    // d_t K identity: BSSN substitutes the Hamiltonian constraint.
    let zero_rho = Field::new(g.clone(), 0.0);
    let h = hamiltonian_constraint(&adm.gamma, &adm.k, &zero_rho, &config, &values);
    let adm_rhs = adm.rhs(&config, &values);
    let bssn_rhs = bssn.rhs(&config, &values);
    g.for_each_index(|idx| {
        let gamma = adm.gamma.get(idx);
        let k = adm.k.get(idx);
        let ginv = invert(&matrix(gamma));
        // K_ab K^ab
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
        // ADM d_t K = gamma^ij d_t K_ij + 2 alpha K_ab K^ab  (frozen shift)
        let mut adm_dtk = 2.0 * adm.lapse.get(idx) * k2;
        let kr = adm_rhs.k.get(idx);
        for i in 0..D {
            for j in 0..D {
                adm_dtk += ginv[i][j] * kr[(i, j)];
            }
        }
        let bssn_dtk = bssn_rhs.trace_k.get(idx);
        let alpha = adm.lapse.get(idx);
        let diff = (bssn_dtk - (adm_dtk - alpha * h.get(idx))).abs();
        assert!(diff < 1.0e-4, "K identity mismatch {diff}");
    });
}

// The static Schwarzschild slice is a fixed point of the BSSN system too: the
// lapse law, the conformal metric, the trace, A tilde and Gamma^i all have to
// vanish on it up to mesh truncation. The singularity sits in a corner of the
// box and the periodic faces wrap non-periodic data, so only the interior
// window x, y, z in [0.75, 1.25] is inspected, as in the ADM fixed-point
// test.
#[test]
fn schwarzschild_slice_is_a_bssn_fixed_point() {
    let (config, values) = periodic();
    let g = grid(24, 0.0416);
    let adm = schwarzschild::<D>(g.clone(), 0.05).unwrap();
    let bssn = BssnState::from_adm(&adm, &config, &values);
    let r = bssn.rhs(&config, &values);

    let mut worst_phi = 0.0f64;
    let mut worst_trace = 0.0f64;
    let mut worst_a = 0.0f64;
    let mut worst_gamma = 0.0f64;
    g.for_each_index(|idx| {
        let c = g.coords(idx);
        if !(c[0] >= 0.75 && c[0] <= 1.25 && c[1] >= 0.75 && c[1] <= 1.25 && c[2] >= 0.75 && c[2] <= 1.25) {
            return;
        }

        worst_phi = worst_phi.max(r.phi.get(idx).abs());
        worst_trace = worst_trace.max(r.trace_k.get(idx).abs());
        let at = r.conformal_a.get(idx);
        for i in 0..D {
            for j in 0..D {
                worst_a = worst_a.max(at[(i, j)].abs());
            }
        }
        let gn = r.conformal_gamma.get(idx);
        for i in 0..D {
            worst_gamma = worst_gamma.max(gn[i].abs());
        }
    });

    // d_t K is pure mesh truncation (the static slice makes the continuum
    // value vanish), so it sits at the truncation level like in the ADM
    // fixed-point test; the other four are exactly zero to roundoff.
    assert!(worst_phi < 1.0e-6, "d_t phi = {worst_phi}");
    assert!(worst_trace < 1.0e-3, "d_t K = {worst_trace}");
    assert!(worst_a < 1.0e-6, "d_t A tilde = {worst_a}");
    assert!(worst_gamma < 1.0e-6, "d_t Gamma = {worst_gamma}");
}