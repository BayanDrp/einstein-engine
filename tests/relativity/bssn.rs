use einstein_engine::geometry::tensor::{det, invert, Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::initial_data::schwarzschild::schwarzschild;
use einstein_engine::numerics::rk4::EvolutionState;
use einstein_engine::relativity::bssn::BssnState;
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