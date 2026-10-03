use einstein_engine::geometry::curvature::spatial_ricci_scalar;
use einstein_engine::geometry::tensor::Tensor2;
use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
use einstein_engine::grid::{Field, Grid};
use std::sync::Arc;

// a flat spatial slice: gamma_ij = delta_ij is constant, so every
// Christoffel symbol (and all the curvature above it) is zero.
#[test]
fn spatial_curvature_flat_zero() {
    let grid = Arc::new(Grid::new([5, 5, 5], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
    let spatial: Field<Tensor2<3>, 3> = Field::new(
        grid.clone(),
        Tensor2::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    let config = BoundaryConfig::<3>::periodic();
    let values = BoundaryValues::<3>::zeros();

    let r = spatial_ricci_scalar::<3>(&spatial, &config, &values);

    let v = *r.get([2, 2, 2]);
    assert!(v.abs() < 1e-12);
}

// a 2D slice with gamma_xx = 1 and gamma_yy = 1 + x^2. this is the
// "surface of revolution" metric ds^2 = dx^2 + f(x)^2 dy^2 with
// f = sqrt(1 + x^2), whose Gaussian curvature is K = -f''/f = -1,
// so R^(3) = 2K = -2. checked at the middle point x = 0; the grid
// uses central differences so there is a tiny O(h^2) error.
#[test]
fn spatial_curvature_surface_of_revolution() {
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
    let config = BoundaryConfig::<2>::periodic();
    let values = BoundaryValues::<2>::zeros();

    let r = spatial_ricci_scalar::<2>(&spatial, &config, &values);

    let v = *r.get([10, 10]); // x = 0, y = 0
    assert!((v + 2.0).abs() < 0.01, "R^(3) = {v}, expected about -2");
}