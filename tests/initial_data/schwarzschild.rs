use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::initial_data::schwarzschild::schwarzschild;
use einstein_engine::numerics::rk4::EvolutionState;
use einstein_engine::relativity::constraints::hamiltonian_constraint;
use einstein_engine::validation::constraints::residuals;
use std::sync::Arc;

const D: usize = 3;
const MASS: f64 = 1.0;

// The puncture sits at the coordinate origin and the grid lives in the octant
// x, y, z >= 0.5, so every sample lies outside the horizon (r > M/2 = 0.5).
const ORIGIN: [f64; D] = [0.5, 0.5, 0.5];

fn periodic() -> (BoundaryConfig<D>, BoundaryValues<D>) {
    (BoundaryConfig::<D>::periodic(), BoundaryValues::<D>::zeros())
}

fn grid(shape: usize, h: f64) -> Arc<Grid<D>> {
    Arc::new(Grid::new([shape; D], [h; D], ORIGIN))
}

fn zero_rho(grid: &Arc<Grid<D>>) -> Field<f64, D> {
    Field::new(grid.clone(), 0.0)
}

fn zero_momentum(grid: &Arc<Grid<D>>) -> Field<Vector<D>, D> {
    Field::new(grid.clone(), Vector::<D>::zero())
}

// Every field is the analytic isotropic-Schwarzschild value pointwise:
// gamma_ij = psi^4 delta_ij with psi = 1 + M/2r, the Schwarzschild lapse, and
// K_ij = beta^i = 0.
#[test]
fn matches_the_isotropic_form_pointwise() {
    let grid = grid(8, 0.25);
    let state = schwarzschild::<D>(grid.clone(), MASS).expect("valid data");

    grid.for_each_index(|idx| {
        let c = grid.coords(idx);
        let r = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        let psi = 1.0 + MASS / (2.0 * r);
        let psi4 = psi * psi * psi * psi;
        let alpha = (1.0 - MASS / (2.0 * r)) / (1.0 + MASS / (2.0 * r));

        let g = state.gamma.get(idx);
        for i in 0..D {
            assert!(
                (g[(i, i)] - psi4).abs() < 1e-12,
                "gamma[{i},{i}] at {idx:?}: got {}, want {psi4}",
                g[(i, i)]
            );
            for j in 0..D {
                if i != j {
                    assert_eq!(g[(i, j)], 0.0, "gamma[{i},{j}] at {idx:?} should be 0");
                }
            }
        }

        assert!(
            (state.lapse.get(idx) - alpha).abs() < 1e-12,
            "lapse at {idx:?}: got {}, want {alpha}",
            state.lapse.get(idx)
        );
        assert_eq!(state.k.get(idx), &Tensor2::<D>::zero());
        assert_eq!(state.shift.get(idx), &Vector::<D>::zero());
    });
}

// The slice is a moment of time symmetry: K = 0 makes the momentum constraint
// identically zero, down to the last bit (a divergence of zero is zero).
#[test]
fn momentum_constraint_is_exactly_zero() {
    let grid = grid(8, 0.25);
    let state = schwarzschild::<D>(grid.clone(), MASS).unwrap();
    let (config, values) = periodic();

    let r = residuals(
        &state.gamma,
        &state.k,
        &zero_rho(&grid),
        &zero_momentum(&grid),
        &config,
        &values,
    );

    assert_eq!(r.momentum_max, 0.0, "K = 0 must give exactly zero momentum");
}

// Largest |H| over a fixed physical window, so two grids of different spacing
// are compared on the same region rather than on different cell counts.
fn window_hamiltonian_max(state: &EvolutionState<D>, lo: f64, hi: f64) -> f64 {
    let (config, values) = periodic();
    let h = hamiltonian_constraint(
        &state.gamma,
        &state.k,
        &zero_rho(&state.grid),
        &config,
        &values,
    );

    let mut worst = 0.0f64;
    state.grid.for_each_index(|idx| {
        let c = state.grid.coords(idx);
        if c.iter().all(|&x| x >= lo && x <= hi) {
            worst = worst.max(h.get(idx).abs());
        }
    });
    worst
}

// The point of this data: gamma is conformally flat and psi is harmonic away
// from the puncture, so R^(3) = 0 exactly in the continuum and the whole
// residual is finite-difference truncation. Halving h must quarter it, which
// is what proves the data is on-shell rather than accidentally small.
//
// Measured over x, y, z in [0.75, 1.25], away from the periodic faces.
#[test]
fn hamiltonian_residual_converges_second_order() {
    let coarse = schwarzschild::<D>(grid(16, 1.0 / 16.0), MASS).unwrap();
    let fine = schwarzschild::<D>(grid(32, 1.0 / 32.0), MASS).unwrap();

    let (lo, hi) = (0.75, 1.25);
    let e_coarse = window_hamiltonian_max(&coarse, lo, hi);
    let e_fine = window_hamiltonian_max(&fine, lo, hi);
    let ratio = e_coarse / e_fine;

    println!(
        "H residual: coarse(h=1/16) = {e_coarse:.4e}, fine(h=1/32) = {e_fine:.4e}, ratio = {ratio:.3}"
    );

    assert!(
        e_coarse > 0.0,
        "an exact zero would mean the test is not measuring truncation"
    );
    assert!(
        (3.0..5.0).contains(&ratio),
        "residual ratio was {ratio}, expected ~4 for a second-order operator"
    );
}

// Guard rails: the constructor refuses data it cannot represent.
#[test]
fn rejects_a_puncture_inside_the_grid() {
    // origin at zero puts r = 0 on the corner grid point
    let bad = Arc::new(Grid::new([4, 4, 4], [0.25; D], [0.0; D]));
    assert!(schwarzschild::<D>(bad, MASS).is_err());
}

#[test]
fn rejects_nonpositive_or_nonfinite_mass() {
    assert!(schwarzschild::<D>(grid(4, 0.25), 0.0).is_err());
    assert!(schwarzschild::<D>(grid(4, 0.25), -1.0).is_err());
    assert!(schwarzschild::<D>(grid(4, 0.25), f64::NAN).is_err());
}

#[test]
fn rejects_dimensions_other_than_three() {
    let flat = Arc::new(Grid::new([4, 4], [0.25, 0.25], [1.0, 1.0]));
    assert!(schwarzschild::<2>(flat, MASS).is_err());
}
