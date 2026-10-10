use einstein_engine::geometry::metric::{
    ExtrinsicCurvature,
    LapseField,
    ShiftField,
    SliceMetric,
};
use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::numerics::rk4::EvolutionState;
use einstein_engine::relativity::evolution::extrinsic_curvature_rhs;
use einstein_engine::relativity::gauge::{lapse_rhs, shift_rhs};
use std::sync::Arc;

const D: usize = 2;

fn config() -> (BoundaryConfig<D>, BoundaryValues<D>) {
    (BoundaryConfig::<D>::periodic(), BoundaryValues::<D>::zeros())
}

fn flat(grid: Arc<Grid<D>>) -> SliceMetric<D> {
    let mut spatial: SliceMetric<D> = Field::new(grid.clone(), Tensor2::<D>::zero());
    grid.for_each_index(|idx| {
        let mut g = Tensor2::<D>::zero();
        for i in 0..D {
            g[(i, i)] = 1.0;
        }
        spatial.set(idx, g);
    });
    spatial
}

fn constant_tensor(grid: Arc<Grid<D>>, f: impl Fn(usize, usize) -> f64) -> ExtrinsicCurvature<D> {
    let mut out: ExtrinsicCurvature<D> = Field::new(grid.clone(), Tensor2::<D>::zero());
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

fn constant_lapse(grid: Arc<Grid<D>>, value: f64) -> LapseField<D> {
    let mut lapse: LapseField<D> = Field::new(grid.clone(), 0.0);
    grid.for_each_index(|idx| lapse.set(idx, value));
    lapse
}

fn zero_shift(grid: Arc<Grid<D>>) -> ShiftField<D> {
    Field::new(grid, Vector::<D>::zero())
}

fn worst(field: &LapseField<D>) -> f64 {
    field.data.iter().fold(0.0f64, |acc, v| acc.max(v.abs()))
}

fn extent(field: &LapseField<D>) -> (f64, f64) {
    field
        .data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(*v), hi.max(*v))
        })
}

fn worst_tensor_diff(left: &ExtrinsicCurvature<D>, right: &ExtrinsicCurvature<D>) -> f64 {
    left.data
        .iter()
        .zip(&right.data)
        .fold(0.0f64, |acc, (a, b)| {
            let mut worst = acc;
            for i in 0..D {
                for j in 0..D {
                    worst = worst.max((a[(i, j)] - b[(i, j)]).abs());
                }
            }
            worst
        })
}

// K = 0 everywhere: 1+log has nothing to do, so the lapse derivative is zero.
#[test]
fn lapse_rhs_vanishes_on_minkowski_data() {
    let grid = Arc::new(Grid::new([5, 5], [0.5, 0.5], [0.0, 0.0]));
    let (boundary, boundary_values) = config();
    let spatial = flat(grid.clone());
    let lapse = constant_lapse(grid.clone(), 1.0);
    let shift = zero_shift(grid.clone());
    let extrinsic: ExtrinsicCurvature<D> = Field::new(grid.clone(), Tensor2::<D>::zero());

    let rhs = lapse_rhs(&lapse, &shift, &spatial, &extrinsic, &boundary, &boundary_values);
    assert!(worst(&rhs) < 1e-14, "got {}", worst(&rhs));
}

// Flat slice so γ^{ij} = δ^{ij}, constant K = diag(1, 2), lapse 0.5.
// The trace is 3, so ∂t α = -2 * 0.5 * 3 = -3 at every point, exactly.
#[test]
fn lapse_rhs_is_minus_two_alpha_trace_k() {
    let grid = Arc::new(Grid::new([5, 5], [0.5, 0.5], [0.0, 0.0]));
    let (boundary, boundary_values) = config();
    let spatial = flat(grid.clone());
    let lapse = constant_lapse(grid.clone(), 0.5);
    let shift = zero_shift(grid.clone());
    let extrinsic = constant_tensor(grid.clone(), |i, j| {
        if i == 0 && j == 0 {
            1.0
        } else if i == 1 && j == 1 {
            2.0
        } else {
            0.0
        }
    });

    let rhs = lapse_rhs(&lapse, &shift, &spatial, &extrinsic, &boundary, &boundary_values);
    grid.for_each_index(|idx| {
        let got = *rhs.get(idx);
        assert!((got + 3.0).abs() < 1e-14, "at {idx:?}: got {got}");
    });
}

// The trace must be taken with γ^{ij}, not δ^{ij}. On the anisotropic slice
// γ_xx = 1 + g, γ_yy = 1 - g with K = diag(g'/2, -g'/2) the δ^{ij} trace
// vanishes identically, while the true one is -g g' / (1 - g^2).
#[test]
fn lapse_rhs_traces_with_the_inverse_metric() {
    // spacing 0.1 puts the samples off the nodes of a wave with wavelength
    // 0.5, so g * g' is genuinely nonzero somewhere. A spacing of half the
    // wavelength would alias every point onto a node.
    let grid = Arc::new(Grid::new([9, 3], [0.1, 0.5], [0.0, 0.0]));
    let (boundary, boundary_values) = config();
    let lapse = constant_lapse(grid.clone(), 1.0);
    let shift = zero_shift(grid.clone());

    let amplitude = 0.002;
    let wave_number = 4.0 * std::f64::consts::PI;

    let mut spatial: SliceMetric<D> = Field::new(grid.clone(), Tensor2::<D>::zero());
    let mut extrinsic: ExtrinsicCurvature<D> = Field::new(grid.clone(), Tensor2::<D>::zero());

    grid.for_each_index(|idx| {
        let x = grid.coords(idx)[0];
        let g = amplitude * (wave_number * x).sin();
        let gp = amplitude * wave_number * (wave_number * x).cos();

        let mut gamma = Tensor2::<D>::zero();
        gamma[(0, 0)] = 1.0 + g;
        gamma[(1, 1)] = 1.0 - g;
        spatial.set(idx, gamma);

        let mut k = Tensor2::<D>::zero();
        k[(0, 0)] = 0.5 * gp;
        k[(1, 1)] = -0.5 * gp;
        extrinsic.set(idx, k);
    });

    let rhs = lapse_rhs(&lapse, &shift, &spatial, &extrinsic, &boundary, &boundary_values);

    let mut worst_error = 0.0f64;
    let mut largest_expected = 0.0f64;
    let mut worst_x = 0.0f64;

    grid.for_each_index(|idx| {
        let x = grid.coords(idx)[0];
        let g = amplitude * (wave_number * x).sin();
        let gp = amplitude * wave_number * (wave_number * x).cos();

        // K = 0.5 g'/(1 + g) - 0.5 g'/(1 - g) = -g g' / (1 - g^2)
        let trace_k = -g * gp / (1.0 - g * g);
        let expected = -2.0 * trace_k;

        if expected.abs() > largest_expected {
            largest_expected = expected.abs();
        }

        let got = *rhs.get(idx);
        if (got - expected).abs() > worst_error {
            worst_error = (got - expected).abs();
            worst_x = x;
        }
    });

    assert!(
        worst_error < 1e-14,
        "worst error {worst_error} at x = {worst_x}"
    );

    // δ^{ij} would give trace = 0.5 g' - 0.5 g' = 0 everywhere, so the two
    // answers only differ where g * g' does not vanish.
    assert!(
        largest_expected > 1e-6,
        "test needs a slice where the inverse metric matters, peak was {largest_expected}"
    );
}

// With K = 0 the whole equation is the advection term: alpha = x with a
// constant beta^x = b gives ∂t α = b in the interior, where a centred
// difference of a linear field is exact. The edges see a wrapped neighbour,
// since alpha = x is not periodic, so they are skipped.
#[test]
fn lapse_rhs_advects_along_the_shift() {
    let grid = Arc::new(Grid::new([5, 3], [0.5, 0.5], [0.0, 0.0]));
    let (boundary, boundary_values) = config();
    let spatial = flat(grid.clone());
    let extrinsic: ExtrinsicCurvature<D> = Field::new(grid.clone(), Tensor2::<D>::zero());

    let b = 2.0;
    let mut lapse: LapseField<D> = Field::new(grid.clone(), 0.0);
    let mut shift = zero_shift(grid.clone());
    grid.for_each_index(|idx| {
        lapse.set(idx, grid.coords(idx)[0]);
        shift.set(idx, Vector::new([b, 0.0]));
    });

    let rhs = lapse_rhs(&lapse, &shift, &spatial, &extrinsic, &boundary, &boundary_values);
    for i in 1..grid.shape[0] - 1 {
        for j in 0..grid.shape[1] {
            let got = *rhs.get([i, j]);
            assert!((got - b).abs() < 1e-12, "at index {i}: got {got} want {b}");
        }
    }
}

// No shift equation exists yet, so the derivative is exactly zero everywhere.
#[test]
fn shift_rhs_is_zero() {
    let grid = Arc::new(Grid::new([4, 4], [1.0, 1.0], [0.0, 0.0]));
    let mut shift = zero_shift(grid.clone());
    grid.for_each_index(|idx| shift.set(idx, Vector::new([1.0, -3.0])));

    let rhs = shift_rhs(&shift);
    assert!(rhs.data.iter().all(|v| v == &Vector::<D>::zero()));
}

// The gauge is wired into EvolutionState::rhs, not just sitting in gauge.rs.
// Flat slice, alpha = 1, K = diag(1, 1) has trace 2, so the returned lapse
// derivative is -4 at every point.
#[test]
fn state_rhs_uses_the_gauge() {
    let grid = Arc::new(Grid::new([5, 5], [0.5, 0.5], [0.0, 0.0]));
    let (boundary, boundary_values) = config();

    let gamma = flat(grid.clone());
    let k = constant_tensor(grid.clone(), |i, j| if i == j { 1.0 } else { 0.0 });
    let lapse = constant_lapse(grid.clone(), 1.0);
    let shift = zero_shift(grid.clone());

    let state = EvolutionState::<D> {
        grid: grid.clone(),
        gamma,
        k,
        lapse,
        shift,
    };

    let rhs = state.rhs(&boundary, &boundary_values);

    grid.for_each_index(|idx| {
        let got = *rhs.lapse.get(idx);
        assert!((got + 4.0).abs() < 1e-14, "at {idx:?}: got {got}");
    });

    assert!(rhs.shift.data.iter().all(|v| v == &Vector::<D>::zero()));

    // rhs() delegates the gauge to the state-level methods
    assert_eq!(
        state.lapse_rhs(&boundary, &boundary_values).data,
        rhs.lapse.data
    );
    assert_eq!(state.shift_rhs().data, rhs.shift.data);
}

// A step must not leave the faces wherever RK4 happened to put them. With a
// uniform trace K the lapse wants to move everywhere, so a Dirichlet face is
// only still at 0.5 because the step re-imposed it.
#[test]
fn rk4_step_reenforces_dirichlet_faces() {
    let grid = Arc::new(Grid::new([5, 3], [0.5, 0.5], [0.0, 0.0]));
    let boundary = BoundaryConfig::<D>::dirichlet();
    let boundary_values = BoundaryValues::<D>::uniform(0.5);

    let state = EvolutionState::<D> {
        grid: grid.clone(),
        gamma: flat(grid.clone()),
        k: constant_tensor(grid.clone(), |i, j| if i == j { 1.0 } else { 0.0 }),
        lapse: constant_lapse(grid.clone(), 1.0),
        shift: zero_shift(grid.clone()),
    };

    let next = state.step_rk4(0.05, &boundary, &boundary_values);

    let mut interior_moved = false;
    grid.for_each_index(|idx| {
        let on_face = (0..D).any(|d| idx[d] == 0 || idx[d] + 1 == grid.shape[d]);
        let alpha = *next.lapse.get(idx);

        if on_face {
            assert_eq!(alpha, 0.5, "face at {idx:?} is not pinned");
        } else {
            interior_moved |= alpha < 1.0;
        }
    });

    assert!(interior_moved, "the interior should have evolved");
}

// Neumann faces are extrapolated from the adjacent interior point, so after a
// step the face must equal its neighbour exactly.
#[test]
fn rk4_step_reenforces_neumann_faces() {
    let grid = Arc::new(Grid::new([5, 3], [0.5, 0.5], [0.0, 0.0]));
    let boundary = BoundaryConfig::<D>::neumann();
    let boundary_values = BoundaryValues::<D>::uniform(0.0);

    let state = EvolutionState::<D> {
        grid: grid.clone(),
        gamma: flat(grid.clone()),
        k: constant_tensor(grid.clone(), |i, j| if i == j { 1.0 } else { 0.0 }),
        lapse: constant_lapse(grid.clone(), 1.0),
        shift: zero_shift(grid.clone()),
    };

    let next = state.step_rk4(0.05, &boundary, &boundary_values);

    for j in 1..grid.shape[1] - 1 {
        assert_eq!(
            *next.lapse.get([0, j]),
            *next.lapse.get([1, j]),
            "lower x face did not follow the interior"
        );
        assert_eq!(
            *next.lapse.get([grid.shape[0] - 1, j]),
            *next.lapse.get([grid.shape[0] - 2, j]),
            "upper x face did not follow the interior"
        );
    }
}

// Boundary enforcement is scoped to the lapse. A scalar per face cannot say
// anything about the components of a vector or rank-2 field, so applying it
// here would silently overwrite them: a uniform 0.25 would turn the shift
// into (0.25, 0.25) and pin the metric's faces to 0.25, making the metric
// singular. Neither must happen.
#[test]
fn boundary_enforcement_leaves_the_metric_and_shift_alone() {
    let grid = Arc::new(Grid::new([5, 3], [0.5, 0.5], [0.0, 0.0]));
    let boundary = BoundaryConfig::<D>::dirichlet();
    let boundary_values = BoundaryValues::<D>::uniform(0.25);

    let state = EvolutionState::<D> {
        grid: grid.clone(),
        gamma: flat(grid.clone()),
        k: constant_tensor(grid.clone(), |i, j| if i == j { 1.0 } else { 0.0 }),
        lapse: constant_lapse(grid.clone(), 1.0),
        shift: zero_shift(grid.clone()),
    };

    let next = state.step_rk4(0.05, &boundary, &boundary_values);

    // the lapse, a genuine scalar, is pinned to the prescribed value
    grid.for_each_index(|idx| {
        let on_face = (0..D).any(|d| idx[d] == 0 || idx[d] + 1 == grid.shape[d]);
        if on_face {
            assert_eq!(*next.lapse.get(idx), 0.25, "lapse face at {idx:?}");
        }
    });

    // The shift has no evolution equation at all, so it can only move if the
    // scalar value were applied to its faces. It must stay exactly zero.
    assert!(
        next.shift.data.iter().all(|v| v == &Vector::<D>::zero()),
        "the shift was overwritten by the scalar boundary value"
    );

    // The metric is not pinned to the scalar value either: a blind Dirichlet
    // application would set every face component to 0.25, off-diagonals
    // included. We only check the invariants that do not depend on how the
    // derivative stencils treat their own faces: the metric stays symmetric,
    // and its diagonal is not the prescribed scalar.
    grid.for_each_index(|idx| {
        let on_face = (0..D).any(|d| idx[d] == 0 || idx[d] + 1 == grid.shape[d]);
        let g = next.gamma.get(idx);
        assert!(
            (g[(0, 1)] - g[(1, 0)]).abs() < 1e-12,
            "gamma is no longer symmetric at {idx:?}"
        );
        if on_face {
            assert_ne!(g[(0, 0)], 0.25, "gamma[0,0] pinned at {idx:?}");
        }
    });
}

// The loop the earlier tests leave open: 1+log makes the lapse non-uniform,
// and that lapse then changes the geometry through the -alpha nabla_i nabla_j
// alpha term in the K equation.
//
// The data is diagnostic, not constraint-satisfying: the point is only that
// the gauge output reaches the evolution equation, which is a statement about
// the wiring, not about a physical solution.
//
// The wave data elsewhere is traceless, so it barely moves the lapse. Here
// the trace is 1 + x^2, so d_t alpha = -2 alpha (1 + x^2) is large and curved,
// giving a lapse gradient with a real second derivative after one step.
#[test]
fn gauge_made_lapse_feeds_back_into_the_k_equation() {
    let grid = Arc::new(Grid::new([9, 3], [0.25, 0.5], [0.0, 0.0]));
    let (boundary, boundary_values) = config();

    let mut spatial: SliceMetric<D> = Field::new(grid.clone(), Tensor2::<D>::zero());
    let mut extrinsic: ExtrinsicCurvature<D> = Field::new(grid.clone(), Tensor2::<D>::zero());

    grid.for_each_index(|idx| {
        let x = grid.coords(idx)[0];

        let mut gamma = Tensor2::<D>::zero();
        for i in 0..D {
            gamma[(i, i)] = 1.0;
        }
        spatial.set(idx, gamma);

        let mut k = Tensor2::<D>::zero();
        k[(0, 0)] = 0.5 * (1.0 + x * x);
        k[(1, 1)] = 0.5 * (1.0 + x * x);
        extrinsic.set(idx, k);
    });

    let state = EvolutionState::<D> {
        grid: grid.clone(),
        gamma: spatial,
        k: extrinsic,
        lapse: constant_lapse(grid.clone(), 1.0),
        shift: zero_shift(grid.clone()),
    };

    // step 1: with beta = 0 and a trace K = 1 + x^2, d_t alpha = -2 alpha K is
    // non-trivial, and it is spatially varying because K is
    let gauge_rhs = state.lapse_rhs(&boundary, &boundary_values);
    let (glo, ghi) = extent(&gauge_rhs);
    assert!(
        ghi - glo > 0.1,
        "d_t alpha should vary in space, got [{glo}, {ghi}]"
    );

    // step 2: one step therefore leaves the lapse clearly non-uniform
    let evolved = state.step_rk4(0.05, &boundary, &boundary_values);
    let (lo, hi) = extent(&evolved.lapse);
    assert!(
        hi - lo > 0.1,
        "the gauge should have made the lapse clearly non-uniform, got [{lo}, {hi}]"
    );

    // step 3: same geometry, two different lapses: only the gauge-made one has
    // a gradient, so any difference is the gauge feeding back
    let flat_lapse = constant_lapse(grid.clone(), 1.0);
    let rhs_gauge = extrinsic_curvature_rhs(
        &evolved.lapse,
        &evolved.shift,
        &evolved.gamma,
        &evolved.k,
        &boundary,
        &boundary_values,
    );
    let rhs_flat = extrinsic_curvature_rhs(
        &flat_lapse,
        &evolved.shift,
        &evolved.gamma,
        &evolved.k,
        &boundary,
        &boundary_values,
    );

    let diff = worst_tensor_diff(&rhs_gauge, &rhs_flat);
    println!(
        "gauge loop: lapse spread = {:.4}, worst d_t K difference = {:.4}",
        hi - lo,
        diff
    );
    assert!(
        diff > 1e-2,
        "the gauge-made lapse should change d_t K, worst difference was {diff}"
    );
}
