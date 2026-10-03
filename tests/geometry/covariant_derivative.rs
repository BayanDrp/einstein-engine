use einstein_engine::geometry::christoffel::spatial_christoffel_symbols;
use einstein_engine::geometry::covariant_derivative::{
    covariant_deriv_covector, covariant_deriv_vector, covariant_divergence_mixed,
};
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use std::sync::Arc;

fn flat_spatial<const D: usize>(grid: Arc<Grid<D>>) -> Field<Tensor2<D>, D> {
    let column = |i: usize, j: usize| if i == j { 1.0 } else { 0.0 };
    let mut spatial: Field<Tensor2<D>, D> = Field::new(grid.clone(), Tensor2::<D>::zero());
    let fill = grid.clone();
    fill.for_each_index(|idx| {
        let mut g = Tensor2::<D>::zero();
        for i in 0..D {
            for j in 0..D {
                g[(i, j)] = column(i, j);
            }
        }
        spatial.set(idx, g);
    });
    spatial
}

// flat slice: Christoffel symbols vanish, so the covariant derivative
// of V^i = x^i (that is (x, y, z)) is the identity matrix, and the
// mixed divergence of T^i_j = delta^i_j is zero.
#[test]
fn cov_deriv_flat_identity() {
    let grid = Arc::new(Grid::new([5, 5, 5], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
    let spatial = flat_spatial::<3>(grid.clone());
    let config = BoundaryConfig::<3>::periodic();
    let values = BoundaryValues::<3>::zeros();
    let gamma = spatial_christoffel_symbols::<3>(&spatial, &config, &values);

    let mut v: Field<Vector<3>, 3> = Field::new(grid.clone(), Vector::<3>::zero());
    let fill = grid.clone();
    fill.for_each_index(|idx| {
        let c = grid.coords(idx);
        v.set(idx, Vector::new([c[0], c[1], c[2]]));
    });

    let dv = covariant_deriv_vector::<3>(&v, &gamma, &config, &values);
    let t = *dv.get([2, 2, 2]);
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
    for k in 0..3 {
        for i in 0..3 {
            let expected = if k == i { 1.0 } else { 0.0 };
            assert!(close(t[(k, i)], expected), "D_{k} V^{i}");
        }
    }
}

// steep 2D slice gamma_yy = 1 + x^2 (see spatial_curvature.rs).
// Gamma^y_xy = x/(1+x^2), Gamma^x_yy = -x.
// for V^i = (x, 0) at x = 1:  D V = [[1, 0], [0, 0.5]]
// for W_i = (x, 0) at x = 1:  D W = [[1, 0], [0, 1]]
#[test]
fn cov_deriv_curved_slice_values() {
    let grid = Arc::new(Grid::new([5, 3], [0.5, 0.5], [0.0, 0.0]));
    let mut spatial: Field<Tensor2<2>, 2> = Field::new(grid.clone(), Tensor2::<2>::zero());
    let fill = grid.clone();
    fill.for_each_index(|idx| {
        let x = grid.coords(idx)[0];
        let mut g = Tensor2::<2>::zero();
        g[(0, 0)] = 1.0;
        g[(1, 1)] = 1.0 + x * x;
        spatial.set(idx, g);
    });
    let config = BoundaryConfig::<2>::periodic();
    let values = BoundaryValues::<2>::zeros();
    let gamma = spatial_christoffel_symbols::<2>(&spatial, &config, &values);

    let mut v: Field<Vector<2>, 2> = Field::new(grid.clone(), Vector::<2>::zero());
    let fill_v = grid.clone();
    fill_v.for_each_index(|idx| {
        let x = grid.coords(idx)[0];
        v.set(idx, Vector::new([x, 0.0]));
    });

    let dv = covariant_deriv_vector::<2>(&v, &gamma, &config, &values);
    let t = *dv.get([2, 1]); // x = 1.0
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
    assert!(close(t[(0, 0)], 1.0)); // D_x V^x
    assert!(close(t[(0, 1)], 0.0)); // D_x V^y
    assert!(close(t[(1, 0)], 0.0)); // D_y V^x
    assert!(close(t[(1, 1)], 0.5)); // D_y V^y = Gamma^y_xy x = 0.5

    // same vector as a covector: W_i = (x, 0)
    let mut w: Field<Vector<2>, 2> = Field::new(grid.clone(), Vector::<2>::zero());
    let fill_w = grid.clone();
    fill_w.for_each_index(|idx| {
        let x = grid.coords(idx)[0];
        w.set(idx, Vector::new([x, 0.0]));
    });

    let dw = covariant_deriv_covector::<2>(&w, &gamma, &config, &values);
    let t = *dw.get([2, 1]);
    assert!(close(t[(0, 0)], 1.0)); // D_x W_x
    assert!(close(t[(0, 1)], 0.0)); // D_x W_y
    assert!(close(t[(1, 0)], 0.0)); // D_y W_x
    assert!(close(t[(1, 1)], 1.0)); // D_y W_y = x^2 = 1

    // divergence of the identity mixed tensor delta^i_j is zero
    let mut m: Field<Tensor2<2>, 2> = Field::new(grid.clone(), Tensor2::<2>::zero());
    let fill_m = grid.clone();
    fill_m.for_each_index(|idx| {
        let mut t = Tensor2::<2>::zero();
        t[(0, 0)] = 1.0;
        t[(1, 1)] = 1.0;
        m.set(idx, t);
    });
    let div = covariant_divergence_mixed::<2>(&m, &gamma, &config, &values);
    let out = *div.get([2, 1]);
    assert!(out[0].abs() < 1e-12, "div^0 = {}", out[0]);
    assert!(out[1].abs() < 1e-12, "div^1 = {}", out[1]);
}