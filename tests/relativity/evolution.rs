use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::numerics::derivatives::shift_derivative;
use einstein_engine::relativity::evolution::{extrinsic_curvature_rhs, spatial_metric_rhs};
use std::sync::Arc;

fn flat_spatial<const D: usize>(grid: Arc<Grid<D>>) -> Field<Tensor2<D>, D> {
    let mut spatial: Field<Tensor2<D>, D> = Field::new(grid.clone(), Tensor2::<D>::zero());
    grid.for_each_index(|idx| {
        let mut g = Tensor2::<D>::zero();
        for i in 0..D {
            g[(i, i)] = 1.0;
        }
        spatial.set(idx, g);
    });
    spatial
}

fn constant_lapse<const D: usize>(grid: Arc<Grid<D>>, value: f64) -> Field<f64, D> {
    let mut lapse = Field::zeros(grid.clone());
    grid.for_each_index(|idx| lapse.set(idx, value));
    lapse
}

fn zero_shift<const D: usize>(grid: Arc<Grid<D>>) -> Field<Vector<D>, D> {
    Field::new(grid.clone(), Vector::<D>::zero())
}

fn constant_tensor_field<const D: usize>(grid: Arc<Grid<D>>, f: impl Fn(usize, usize) -> f64) -> Field<Tensor2<D>, D> {
    let mut out: Field<Tensor2<D>, D> = Field::new(grid.clone(), Tensor2::<D>::zero());
    grid.for_each_index(|idx| {
        let mut t = Tensor2::<D>::zero();
        for i in 0..D {
            for j in 0..D {
                t[(i, j)] = f(i, j);
            }
        }
        out.set(idx, t);
    });
    out
}

// largest absolute component over the interior only. edge stencils see
// a wrapped neighbour for a field that is not periodic (like beta^x = b x),
// so they are not meaningful for these exact-value checks.
fn max_abs_interior<const D: usize>(field: &Field<Tensor2<D>, D>) -> f64 {
    let mut worst = 0.0f64;
    field.grid.for_each_index(|idx| {
        let on_edge = (0..D).any(|d| idx[d] == 0 || idx[d] + 1 == field.grid.shape[d]);
        if on_edge {
            return;
        }
        let t = field.get(idx);
        for i in 0..D {
            for j in 0..D {
                worst = worst.max(t[(i, j)].abs());
            }
        }
    });
    worst
}

// shift_derivative on its own:
// a constant field gives zero for any shift, a linear field gives the
// shift dotted with its gradient, and a zero shift gives zero.
#[test]
fn shift_derivative_basics() {
    let grid = Arc::new(Grid::new([5, 5], [0.5, 0.5], [0.0, 0.0]));
    let config = BoundaryConfig::<2>::periodic();
    let values = BoundaryValues::<2>::zeros();

    // constant field f = 3.0 with a non-zero shift
    let mut constant = Field::zeros(grid.clone());
    grid.for_each_index(|idx| constant.set(idx, 3.0));
    let mut shift = zero_shift::<2>(grid.clone());
    grid.for_each_index(|idx| shift.set(idx, Vector::new([2.0, -1.0])));
    let d = shift_derivative(&constant, &shift, &config, &values);
    grid.for_each_index(|idx| assert!(d.get(idx).abs() < 1e-12, "constant field"));

    // linear field f = x with shift beta^x = 2.0, so the answer is 2.0
    let mut linear = Field::zeros(grid.clone());
    grid.for_each_index(|idx| linear.set(idx, grid.coords(idx)[0]));
    grid.for_each_index(|idx| shift.set(idx, Vector::new([2.0, 0.0])));
    let d = shift_derivative(&linear, &shift, &config, &values);
    grid.for_each_index(|idx| {
        if (1..4).contains(&idx[0]) {
            assert!((*d.get(idx) - 2.0).abs() < 1e-12, "linear field");
        }
    });

    // zero shift kills the derivative
    let zero = zero_shift::<2>(grid.clone());
    let d = shift_derivative(&linear, &zero, &config, &values);
    grid.for_each_index(|idx| assert!(d.get(idx).abs() < 1e-12, "zero shift"));
}

// flat space with beta^x = b x. then gamma_ij = delta_ij is constant
// in ADM, so L_beta gamma_ij = 2 alpha K_ij and the rhs vanishes
// identically. this exercises all three Lie derivative terms at once.
#[test]
fn spatial_rhs_flat_drifting_shift_cancels() {
    let grid = Arc::new(Grid::new([7, 7], [0.5, 0.5], [-1.5, -1.5]));
    let spatial = flat_spatial::<2>(grid.clone());
    let lapse = constant_lapse::<2>(grid.clone(), 1.0);
    let b = 0.75;

    let mut shift: Field<Vector<2>, 2> = Field::new(grid.clone(), Vector::<2>::zero());
    grid.for_each_index(|idx| shift.set(idx, Vector::new([b * grid.coords(idx)[0], 0.0])));

    // 2 alpha K_ij = L_beta gamma_ij = 2 b for the xx component
    let extrinsic = constant_tensor_field::<2>(grid.clone(), |i, j| {
        if i == 0 && j == 0 { b } else { 0.0 }
    });

    let config = BoundaryConfig::<2>::periodic();
    let values = BoundaryValues::<2>::zeros();

    let rhs = spatial_metric_rhs::<2>(&lapse, &shift, &spatial, &extrinsic, &config, &values);
    let worst = max_abs_interior(&rhs);
    assert!(
        worst < 1e-12,
        "flat drifting shift should give zero, got {}",
        worst
    );
}

// with beta = 0 the Lie derivative vanishes and the rhs is -2 alpha K_ij
#[test]
fn spatial_rhs_zero_shift_is_minus_two_alpha_k() {
    let grid = Arc::new(Grid::new([4, 4], [1.0, 1.0], [0.0, 0.0]));
    let spatial = flat_spatial::<2>(grid.clone());
    let lapse = constant_lapse::<2>(grid.clone(), 0.8);
    let shift = zero_shift::<2>(grid.clone());
    let extrinsic = constant_tensor_field::<2>(grid.clone(), |i, j| {
        if i == j { 1.5 } else { 0.0 }
    });

    let config = BoundaryConfig::<2>::periodic();
    let values = BoundaryValues::<2>::zeros();

    let rhs = spatial_metric_rhs::<2>(&lapse, &shift, &spatial, &extrinsic, &config, &values);
    let alpha = 0.8;
    grid.for_each_index(|idx| {
        let t = rhs.get(idx);
        for i in 0..2 {
            for j in 0..2 {
                let expected = -2.0 * alpha * extrinsic.get(idx)[(i, j)];
                assert!((t[(i, j)] - expected).abs() < 1e-12);
            }
        }
    });
}

// curved slice gamma_yy = 1 + x^2 with beta^y = x.
// the rhs must stay symmetric in (i, j): gamma_ij and K_ij are
// symmetric tensors, so their time derivative has to be too.
// (this is the case an unsymmetrised -grad(beta_i) gets wrong)
#[test]
fn spatial_rhs_stays_symmetric_on_curved_slice() {
    let grid = Arc::new(Grid::new([5, 3], [0.5, 0.5], [0.0, 0.0]));
    let mut spatial: Field<Tensor2<2>, 2> = Field::new(grid.clone(), Tensor2::<2>::zero());
    grid.for_each_index(|idx| {
        let x = grid.coords(idx)[0];
        let mut g = Tensor2::<2>::zero();
        g[(0, 0)] = 1.0;
        g[(1, 1)] = 1.0 + x * x;
        spatial.set(idx, g);
    });

    let lapse = constant_lapse::<2>(grid.clone(), 1.0);
    let mut shift: Field<Vector<2>, 2> = Field::new(grid.clone(), Vector::<2>::zero());
    grid.for_each_index(|idx| shift.set(idx, Vector::new([0.0, grid.coords(idx)[0]])));

    // make K deliberately asymmetric in the input to prove the mirroring
    let extrinsic = constant_tensor_field::<2>(grid.clone(), |i, j| {
        if i == 0 && j == 1 {
            0.3
        } else if i == 1 && j == 0 {
            -0.7
        } else {
            0.0
        }
    });

    let config = BoundaryConfig::<2>::periodic();
    let values = BoundaryValues::<2>::zeros();

    let rhs = spatial_metric_rhs::<2>(&lapse, &shift, &spatial, &extrinsic, &config, &values);
    grid.for_each_index(|idx| {
        let t = rhs.get(idx);
        assert_eq!(t[(0, 1)], t[(1, 0)], "rhs must be symmetric");
    });
}

// isolate the advection term: gamma_yy = 1 + x^2 with a constant
// beta^x = 1. the shift gradients vanish, so the only surviving piece
// is beta^k d_k gamma_ij = d_x (1 + x^2) = 2x for the yy component.
#[test]
fn spatial_rhs_advection_term_alone() {
    let grid = Arc::new(Grid::new([5, 3], [0.5, 0.5], [0.0, 0.0]));
    let mut spatial: Field<Tensor2<2>, 2> = Field::new(grid.clone(), Tensor2::<2>::zero());
    grid.for_each_index(|idx| {
        let x = grid.coords(idx)[0];
        let mut g = Tensor2::<2>::zero();
        g[(0, 0)] = 1.0;
        g[(1, 1)] = 1.0 + x * x;
        spatial.set(idx, g);
    });

    let lapse = constant_lapse::<2>(grid.clone(), 1.0);
    let mut shift: Field<Vector<2>, 2> = Field::new(grid.clone(), Vector::<2>::zero());
    grid.for_each_index(|idx| shift.set(idx, Vector::new([1.0, 0.0])));

    let extrinsic: Field<Tensor2<2>, 2> = Field::new(grid.clone(), Tensor2::<2>::zero());

    let config = BoundaryConfig::<2>::periodic();
    let values = BoundaryValues::<2>::zeros();

    let rhs = spatial_metric_rhs::<2>(&lapse, &shift, &spatial, &extrinsic, &config, &values);
    let at_one = *rhs.get([2, 1]); // x = 1.0, central derivative of 1 + x^2 is exact
    assert!((at_one[(0, 0)] - 0.0).abs() < 1e-12, "xx");
    assert!((at_one[(1, 1)] - 2.0).abs() < 1e-12, "yy = 2x, got {}", at_one[(1, 1)]);
}
// ---------------------------------------------------------------------
// extrinsic curvature: d_t K_ij
// ---------------------------------------------------------------------

// flat slice, beta = 0, alpha = 1, K = 0: nothing happens.
#[test]
fn k_rhs_minkowski_static_zero() {
    let grid = Arc::new(Grid::new([5, 5], [0.5, 0.5], [-1.0, -1.0]));
    let spatial = flat_spatial::<2>(grid.clone());
    let lapse = constant_lapse::<2>(grid.clone(), 1.0);
    let shift = zero_shift::<2>(grid.clone());
    let extrinsic: Field<Tensor2<2>, 2> = Field::new(grid.clone(), Tensor2::<2>::zero());

    let config = BoundaryConfig::<2>::periodic();
    let values = BoundaryValues::<2>::zeros();

    let rhs = extrinsic_curvature_rhs::<2>(&lapse, &shift, &spatial, &extrinsic, &config, &values);
    assert!(max_abs_interior(&rhs) < 1e-12, "got {}", max_abs_interior(&rhs));
}

// beta = 0, alpha = 1, flat slice (so R_ij = 0): the rhs is the
// extrinsic curvature part alone,
//
//   d_t K_ij = -2 K_ik K^k_j + K K_ij
//
// with K = [[1, 1], [1, 2]] that gives [[-1, -3], [-3, -4]].
#[test]
fn k_rhs_quadratic_term_in_k() {
    let grid = Arc::new(Grid::new([5, 5], [0.5, 0.5], [-1.0, -1.0]));
    let spatial = flat_spatial::<2>(grid.clone());
    let lapse = constant_lapse::<2>(grid.clone(), 1.0);
    let shift = zero_shift::<2>(grid.clone());
    let extrinsic = constant_tensor_field::<2>(grid.clone(), |i, j| match (i, j) {
        (0, 0) => 1.0,
        (0, 1) | (1, 0) => 1.0,
        (1, 1) => 2.0,
        _ => 0.0,
    });

    let config = BoundaryConfig::<2>::periodic();
    let values = BoundaryValues::<2>::zeros();

    let rhs = extrinsic_curvature_rhs::<2>(&lapse, &shift, &spatial, &extrinsic, &config, &values);
    let expected = [[-1.0, -3.0], [-3.0, -4.0]];

    grid.for_each_index(|idx| {
        if idx[0] == 0 || idx[0] == grid.shape[0] - 1 {
            return;
        }
        if idx[1] == 0 || idx[1] == grid.shape[1] - 1 {
            return;
        }
        let t = rhs.get(idx);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (t[(i, j)] - expected[i][j]).abs() < 1e-12,
                    "K rhs ({i},{j}): got {} want {}",
                    t[(i, j)],
                    expected[i][j]
                );
            }
        }
    });
}

// flat slice, K = 0, beta = 0, lapse alpha = 1 + x^2. the lapse term
// is all that survives:
//
//   d_t K_ij = -alpha nabla_i nabla_j alpha
//
// on a flat slice Gamma = 0, so nabla_i nabla_j alpha = d_i d_j alpha.
// for alpha = 1 + x^2 that is d_xx alpha = 2 exactly, giving
// d_t K_xx = -2 (1 + x^2) and everything else zero.
#[test]
fn k_rhs_lapse_second_derivative() {
    let grid = Arc::new(Grid::new([7, 3], [0.5, 0.5], [0.0, 0.0]));
    let spatial = flat_spatial::<2>(grid.clone());

    let mut lapse = Field::zeros(grid.clone());
    grid.for_each_index(|idx| {
        let x = grid.coords(idx)[0];
        lapse.set(idx, 1.0 + x * x);
    });

    let shift = zero_shift::<2>(grid.clone());
    let extrinsic: Field<Tensor2<2>, 2> = Field::new(grid.clone(), Tensor2::<2>::zero());

    let config = BoundaryConfig::<2>::periodic();
    let values = BoundaryValues::<2>::zeros();

    let rhs = extrinsic_curvature_rhs::<2>(&lapse, &shift, &spatial, &extrinsic, &config, &values);

    // Only the deep interior is trustworthy here. A second derivative is
    // built by differentiating a first derivative, so it needs a two cell
    // halo of valid data: at index 1 the Hessian stencil reaches index 0,
    // whose gradient already wrapped against the periodic boundary. Check
    // 2..=n-3.
    for i in 2..grid.shape[0] - 2 {
        let x = grid.coords([i, 1])[0];
        let t = *rhs.get([i, 1]);
        let expected = -2.0 * (1.0 + x * x);
        assert!(
            (t[(0, 0)] - expected).abs() < 1e-12,
            "x = {x}: got {} want {expected}",
            t[(0, 0)]
        );
        assert!(t[(1, 1)].abs() < 1e-12, "yy component should vanish");
        assert!(t[(0, 1)].abs() < 1e-12, "xy component should vanish");
    }
}

// the curvature piece must actually contribute: on the surface of
// revolution slice gamma_yy = 1 + x^2 the Ricci tensor is nonzero, so
// with alpha = 2 and K = beta = 0 the rhs is just 2 R_ij and cannot
// vanish. this checks R_ij is wired into the K equation at all.
#[test]
fn k_rhs_picks_up_spatial_ricci() {
    let grid = Arc::new(Grid::new([9, 3], [0.25, 0.5], [0.0, 0.0]));
    let mut spatial: Field<Tensor2<2>, 2> = Field::new(grid.clone(), Tensor2::<2>::zero());
    grid.for_each_index(|idx| {
        let x = grid.coords(idx)[0];
        let mut g = Tensor2::<2>::zero();
        g[(0, 0)] = 1.0;
        g[(1, 1)] = 1.0 + x * x;
        spatial.set(idx, g);
    });

    let lapse = constant_lapse::<2>(grid.clone(), 2.0);
    let shift = zero_shift::<2>(grid.clone());
    let extrinsic: Field<Tensor2<2>, 2> = Field::new(grid.clone(), Tensor2::<2>::zero());

    let config = BoundaryConfig::<2>::periodic();
    let values = BoundaryValues::<2>::zeros();

    let rhs = extrinsic_curvature_rhs::<2>(&lapse, &shift, &spatial, &extrinsic, &config, &values);
    let t = *rhs.get([4, 1]); // x = 1.0
    // For ds^2 = dx^2 + f(x)^2 dy^2 the Gaussian curvature is -f''/f, so
    // R^(3) = -2 f''/f. With f = sqrt(1 + x^2), f'' = (1 + x^2)^(-3/2), giving
    // R^(3) = -2/(1 + x^2)^2 = -1/2 at x = 1. In two dimensions
    // R_ij = (R/2) gamma_ij, so R_xx = -1/4 and R_yy = -1/2 at x = 1,
    // and multiplying by alpha = 2 gives -1/2 and -1.
    // central differences carry O(h^2) error, hence the 0.05 tolerance.
    assert!((t[(0, 0)] + 0.5).abs() < 0.05, "xx: got {}", t[(0, 0)]);
    assert!((t[(1, 1)] + 1.0).abs() < 0.05, "yy: got {}", t[(1, 1)]);
}

// like the metric equation, the K rhs has to stay symmetric in (i, j).
// we feed in a deliberately asymmetric K to prove the mirroring works.
#[test]
fn k_rhs_stays_symmetric_on_curved_slice() {
    let grid = Arc::new(Grid::new([5, 3], [0.5, 0.5], [0.0, 0.0]));
    let mut spatial: Field<Tensor2<2>, 2> = Field::new(grid.clone(), Tensor2::<2>::zero());
    grid.for_each_index(|idx| {
        let x = grid.coords(idx)[0];
        let mut g = Tensor2::<2>::zero();
        g[(0, 0)] = 1.0;
        g[(1, 1)] = 1.0 + x * x;
        spatial.set(idx, g);
    });

    let mut lapse = Field::zeros(grid.clone());
    grid.for_each_index(|idx| {
        let x = grid.coords(idx)[0];
        lapse.set(idx, 1.0 + x * x);
    });

    let mut shift: Field<Vector<2>, 2> = Field::new(grid.clone(), Vector::<2>::zero());
    grid.for_each_index(|idx| shift.set(idx, Vector::new([0.3, grid.coords(idx)[0]])));

    let extrinsic = constant_tensor_field::<2>(grid.clone(), |i, j| match (i, j) {
        (0, 0) => 0.4,
        (0, 1) => 0.3,
        (1, 0) => -0.7,
        (1, 1) => -0.2,
        _ => 0.0,
    });

    let config = BoundaryConfig::<2>::periodic();
    let values = BoundaryValues::<2>::zeros();

    let rhs = extrinsic_curvature_rhs::<2>(&lapse, &shift, &spatial, &extrinsic, &config, &values);
    grid.for_each_index(|idx| {
        let t = rhs.get(idx);
        assert_eq!(t[(0, 1)], t[(1, 0)], "K rhs must be symmetric");
    });
}
