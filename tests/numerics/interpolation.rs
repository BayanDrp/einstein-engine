use einstein_engine::geometry::metric::{inverse_spacetime_metric, SpacetimeMetric};
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::numerics::interpolation::{
    interpolate_scalar, interpolate_tensor, interpolate_vector, InterpolationError,
};
use std::sync::Arc;

const D: usize = 3;

fn grid() -> Arc<Grid<D>> {
    Arc::new(Grid::new([8; D], [0.125; D], [0.5; D]))
}

// constant field: every sample, interior or face, returns the value.
#[test]
fn constant_field_exact() {
    let g = grid();
    let f = Field::new(g, 2.5);

    for p in [
        [0.5, 0.5, 0.5],
        [0.5625, 0.6875, 0.8125],
        [1.375, 1.375, 1.375],
        [1.0, 0.5, 1.375],
    ] {
        assert_eq!(interpolate_scalar(&f, p), Ok(2.5));
    }
}

// affine field: trilinear interpolation is exact up to roundoff.
#[test]
fn affine_field_exact_interior() {
    let g = grid();
    let mut f = Field::new(g.clone(), 0.0);
    g.for_each_index(|idx| {
        let c = g.coords(idx);
        f.set(idx, 1.0 + 2.0 * c[0] - 3.0 * c[1] + 0.5 * c[2]);
    });

    for p in [
        [0.56, 0.69, 0.81],
        [1.37, 1.31, 1.19],
        [0.8125, 1.0625, 0.9375],
        [1.0, 1.0, 1.0],
        [0.63, 1.31, 0.77],
    ] {
        let want = 1.0 + 2.0 * p[0] - 3.0 * p[1] + 0.5 * p[2];
        let got = interpolate_scalar(&f, p).unwrap();
        assert!(
            (got - want).abs() < 1e-12,
            "at {p:?}: got {got}, want {want}"
        );
    }
}

// sampling exactly on nodes returns the stored values.
#[test]
fn nodes_sample_exactly() {
    let g = grid();
    let mut f = Field::new(g.clone(), 0.0);
    g.for_each_index(|idx| {
        let c = g.coords(idx);
        f.set(idx, c[0] * c[1] + c[2]);
    });

    g.for_each_index(|idx| {
        let want = *f.get(idx);
        assert_eq!(interpolate_scalar(&f, g.coords(idx)), Ok(want));
    });
}

// the upper boundary belongs to the domain and samples exactly.
#[test]
fn upper_boundary_exact() {
    let g = grid();
    let mut f = Field::new(g.clone(), 0.0);
    g.for_each_index(|idx| {
        let c = g.coords(idx);
        f.set(idx, c[0] - 2.0 * c[1] + c[2]);
    });

    // max corner and face centers on the upper faces
    for p in [
        [1.375, 1.375, 1.375],
        [0.9375, 1.375, 1.0625],
        [1.375, 0.8125, 1.375],
        [1.0625, 1.1875, 1.375],
    ] {
        let want = p[0] - 2.0 * p[1] + p[2];
        let got = interpolate_scalar(&f, p).unwrap();
        assert!(
            (got - want).abs() < 1e-12,
            "at {p:?}: got {got}, want {want}"
        );
    }
}

// outside the closed node box (and NaN) is an error, never a panic.
#[test]
fn outside_domain_errors() {
    let g = grid();
    let f = Field::new(g, 1.0);

    for p in [
        [0.499, 1.0, 1.0],
        [1.0, 1.0, 1.376],
        [2.0, 2.0, 2.0],
        [-1.0, 0.5, 0.5],
        [f64::NAN, 1.0, 1.0],
        [1.0, f64::INFINITY, 1.0],
    ] {
        assert_eq!(
            interpolate_scalar(&f, p),
            Err(InterpolationError::OutsideDomain),
            "at {p:?}"
        );
    }
}

// vector and tensor fields interpolate component-wise, exactly on affine data.
#[test]
fn vector_tensor_affine_exact() {
    let g = grid();
    let mut vf = Field::new(g.clone(), Vector::<D>::zero());
    let mut tf = Field::new(g.clone(), Tensor2::<D>::zero());
    g.for_each_index(|idx| {
        let c = g.coords(idx);
        let mut v = Vector::<D>::zero();
        for i in 0..D {
            v[i] = 1.0 + (i as f64 + 1.0) * c[i];
        }
        vf.set(idx, v);
        let mut t = Tensor2::<D>::zero();
        for i in 0..D {
            for j in 0..D {
                t[(i, j)] = c[i] - 2.0 * c[j];
            }
        }
        tf.set(idx, t);
    });

    let p = [0.66, 0.91, 1.22];
    let v = interpolate_vector(&vf, p).unwrap();
    for i in 0..D {
        let want = 1.0 + (i as f64 + 1.0) * p[i];
        assert!((v[i] - want).abs() < 1e-12, "v[{i}]: got {}, want {want}", v[i]);
    }
    let t = interpolate_tensor(&tf, p).unwrap();
    for i in 0..D {
        for j in 0..D {
            let want = p[i] - 2.0 * p[j];
            assert!(
                (t[(i, j)] - want).abs() < 1e-12,
                "t[{i}][{j}]: got {}, want {want}",
                t[(i, j)]
            );
        }
    }
}

// a valid nontrivial metric inverts cleanly: g * g^-1 = identity.
#[test]
fn inverse_spacetime_metric_identity() {
    let mut shift = Vector::<D>::zero();
    shift[0] = 0.1;
    shift[1] = 0.05;
    shift[2] = -0.02;
    let mut spatial = Tensor2::<D>::zero();
    spatial[(0, 0)] = 1.1;
    spatial[(1, 1)] = 0.9;
    spatial[(2, 2)] = 1.0;
    spatial[(0, 1)] = 0.03;
    spatial[(1, 0)] = 0.03;
    let metric = SpacetimeMetric::from_3plus1(0.9, &shift, &spatial);

    let g = metric.as_matrix::<4>();
    let ginv = inverse_spacetime_metric::<D, 4>(&metric);

    for a in 0..4 {
        for b in 0..4 {
            let mut s = 0.0;
            for c in 0..4 {
                s += g[a][c] * ginv[c][b];
            }
            let want = if a == b { 1.0 } else { 0.0 };
            assert!(
                (s - want).abs() < 1e-12,
                "[{a}][{b}]: got {s}, want {want}"
            );
        }
    }
}

// zero lapse makes the 4-metric degenerate: inversion must fail loudly,
// not return garbage.
#[test]
#[should_panic(expected = "singular matrix")]
fn singular_metric_panics() {
    let spatial = Tensor2::<D>::zero();
    let metric = SpacetimeMetric::from_3plus1(
        0.0,
        &Vector::<D>::zero(),
        &spatial,
    );
    let _ = inverse_spacetime_metric::<D, 4>(&metric);
}
