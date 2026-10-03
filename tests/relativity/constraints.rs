use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use einstein_engine::relativity::constraints::{hamiltonian_constraint, momentum_constraint};
use std::sync::Arc;

// flat slice, no extrinsic curvature, no matter:
// H = R^(3) + K^2 - K_ij K^ij - 16 pi rho = 0 - 0 - 0 - 0 = 0.
#[test]
fn hamiltonian_minkowski_vacuum_zero() {
    let grid = Arc::new(Grid::new([5, 5, 5], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
    let spatial: Field<Tensor2<3>, 3> = Field::new(
        grid.clone(),
        Tensor2::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    let extrinsic: Field<Tensor2<3>, 3> = Field::new(grid.clone(), Tensor2::<3>::zero());
    let rho: Field<f64, 3> = Field::new(grid.clone(), 0.0);
    let config = BoundaryConfig::<3>::periodic();
    let values = BoundaryValues::<3>::zeros();

    let h = hamiltonian_constraint(&spatial, &extrinsic, &rho, &config, &values);
    let v = *h.get([2, 2, 2]);
    assert!(v.abs() < 1e-12);
}

// same flat Minkowski data but with matter density rho = 2:
// the K and R terms stay zero, so H = -16 pi rho = -32 pi.
#[test]
fn hamiltonian_minkowski_with_matter() {
    let grid = Arc::new(Grid::new([5, 5, 5], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
    let spatial: Field<Tensor2<3>, 3> = Field::new(
        grid.clone(),
        Tensor2::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    let extrinsic: Field<Tensor2<3>, 3> = Field::new(grid.clone(), Tensor2::<3>::zero());
    let rho: Field<f64, 3> = Field::new(grid.clone(), 2.0);
    let config = BoundaryConfig::<3>::periodic();
    let values = BoundaryValues::<3>::zeros();

    let h = hamiltonian_constraint(&spatial, &extrinsic, &rho, &config, &values);
    let v = *h.get([2, 2, 2]);
    assert!((v + 32.0 * std::f64::consts::PI).abs() < 1e-12);
}

// a curved slice with K_ij = 0 and no matter violates the constraint:
// H = R^(3) = about -2 at the center (the slice bends without matter
// to curve it). this is what monitoring the constraints detects.
#[test]
fn hamiltonian_curved_slice_violation() {
    let grid = Arc::new(Grid::new([21, 21], [0.1, 0.1], [-1.0, -1.0]));
    let mut spatial: Field<Tensor2<2>, 2> = Field::new(grid.clone(), Tensor2::<2>::zero());
    let fill = grid.clone();
    fill.for_each_index(|idx| {
        let x = grid.coords(idx)[0];
        let mut g = Tensor2::<2>::zero();
        g[(0, 0)] = 1.0;
        g[(1, 1)] = 1.0 + x * x;
        spatial.set(idx, g);
    });
    let extrinsic: Field<Tensor2<2>, 2> = Field::new(grid.clone(), Tensor2::<2>::zero());
    let rho: Field<f64, 2> = Field::new(grid.clone(), 0.0);
    let config = BoundaryConfig::<2>::periodic();
    let values = BoundaryValues::<2>::zeros();

    let h = hamiltonian_constraint(&spatial, &extrinsic, &rho, &config, &values);
    let v = *h.get([10, 10]); // x = 0, y = 0
    assert!((v + 2.0).abs() < 0.01, "H = {v}, expected about -2");
}

// zero extrinsic curvature and zero momentum: M^i = 0.
#[test]
fn momentum_minkowski_vacuum_zero() {
    let grid = Arc::new(Grid::new([5, 5, 5], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
    let spatial: Field<Tensor2<3>, 3> = Field::new(
        grid.clone(),
        Tensor2::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    let extrinsic: Field<Tensor2<3>, 3> = Field::new(grid.clone(), Tensor2::<3>::zero());
    let momentum: Field<Vector<3>, 3> = Field::new(grid.clone(), Vector::<3>::zero());
    let config = BoundaryConfig::<3>::periodic();
    let values = BoundaryValues::<3>::zeros();

    let m = momentum_constraint(&spatial, &extrinsic, &momentum, &config, &values);
    let v = *m.get([2, 2, 2]);
    assert_eq!(v, Vector::<3>::zero());
}

// flat slice with K_xy = x (so K^y_x = x): the divergence of the
// trace-free part is nonzero, so without matter M^y = d_x K^y_x = 1.
#[test]
fn momentum_curvature_violation() {
    let grid = Arc::new(Grid::new([5, 3, 3], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
    let spatial: Field<Tensor2<3>, 3> = Field::new(
        grid.clone(),
        Tensor2::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    let mut extrinsic: Field<Tensor2<3>, 3> = Field::new(grid.clone(), Tensor2::<3>::zero());
    let fill = grid.clone();
    fill.for_each_index(|idx| {
        let x = grid.coords(idx)[0];
        let mut k = Tensor2::<3>::zero();
        k[(0, 1)] = x;
        k[(1, 0)] = x;
        extrinsic.set(idx, k);
    });
    let momentum: Field<Vector<3>, 3> = Field::new(grid.clone(), Vector::<3>::zero());
    let config = BoundaryConfig::<3>::periodic();
    let values = BoundaryValues::<3>::zeros();

    let m = momentum_constraint(&spatial, &extrinsic, &momentum, &config, &values);
    let v = *m.get([2, 1, 1]); // x = 1.0
    assert!((v[0]).abs() < 1e-12, "M^x = {}", v[0]);
    assert!((v[1] - 1.0).abs() < 1e-12, "M^y = {}", v[1]);
    assert!((v[2]).abs() < 1e-12, "M^z = {}", v[2]);
}

// the same slice now carries momentum density S^y = 1/(8 pi),
// which exactly balances the curvature: M^y -> 0. this checks
// the -8 pi S^i coupling of matter to geometry.
#[test]
fn momentum_satisfied_with_matter() {
    let grid = Arc::new(Grid::new([5, 3, 3], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
    let spatial: Field<Tensor2<3>, 3> = Field::new(
        grid.clone(),
        Tensor2::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    let mut extrinsic: Field<Tensor2<3>, 3> = Field::new(grid.clone(), Tensor2::<3>::zero());
    let fill_k = grid.clone();
    fill_k.for_each_index(|idx| {
        let x = grid.coords(idx)[0];
        let mut k = Tensor2::<3>::zero();
        k[(0, 1)] = x;
        k[(1, 0)] = x;
        extrinsic.set(idx, k);
    });
    let mut momentum: Field<Vector<3>, 3> = Field::new(grid.clone(), Vector::<3>::zero());
    let fill_s = grid.clone();
    fill_s.for_each_index(|idx| {
        let mut s = Vector::<3>::zero();
        s[1] = 1.0 / (8.0 * std::f64::consts::PI);
        momentum.set(idx, s);
    });
    let config = BoundaryConfig::<3>::periodic();
    let values = BoundaryValues::<3>::zeros();

    let m = momentum_constraint(&spatial, &extrinsic, &momentum, &config, &values);
    let v = *m.get([2, 1, 1]); // x = 1.0
    assert!(v[0].abs() < 1e-12, "M^x = {}", v[0]);
    assert!(v[1].abs() < 1e-12, "M^y = {}", v[1]);
    assert!(v[2].abs() < 1e-12, "M^z = {}", v[2]);
}