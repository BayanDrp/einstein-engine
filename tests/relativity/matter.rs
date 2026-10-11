use einstein_engine::geometry::tensor::{Tensor2, Vector};
use einstein_engine::relativity::matter::four_velocity_covariant;

const D: usize = 3;

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
