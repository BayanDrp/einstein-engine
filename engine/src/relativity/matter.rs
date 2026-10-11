use crate::geometry::tensor::{Tensor2, Vector};
use crate::grid::Field;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FourVelocityError {
    NonFiniteInput,
    InvalidLapse,
    InvalidSpeed,
}

/// Build the covariant four-velocity (u_0, u_i) for any spatial dimension.
///
/// velocity: physical 3-velocity v^i measured by Eulerian observers.
/// alpha: lapse (must be positive).
/// beta: contravariant shift vector beta^i.
/// spatial_metric: gamma_ij.
///
/// Returns the time component u_0 alongside the spatial covariant components
/// u_i, split because a D+1-sized array is not expressible with stable
/// const-generics.
///
/// Uses c = 1 and the (-+++) metric signature, so that u^mu u_mu = -1 with
/// u^0 = W / alpha and u^i = W (v^i - beta^i / alpha).
pub fn four_velocity_covariant<const D: usize>(
    alpha: f64,
    beta: &Vector<D>,
    spatial_metric: &Tensor2<D>,
    velocity: &Vector<D>,
) -> Result<(f64, Vector<D>), FourVelocityError> {
    if !alpha.is_finite() {
        return Err(FourVelocityError::NonFiniteInput);
    }
    for i in 0..D {
        if !beta[i].is_finite() || !velocity[i].is_finite() {
            return Err(FourVelocityError::NonFiniteInput);
        }
        for j in 0..D {
            if !spatial_metric[(i, j)].is_finite() {
                return Err(FourVelocityError::NonFiniteInput);
            }
        }
    }

    if alpha <= 0.0 {
        return Err(FourVelocityError::InvalidLapse);
    }

    // Lower the spatial velocity index:
    // v_i = gamma_ij * v^j
    let mut v_cov: Vector<D> = Vector::zero();

    for i in 0..D {
        for j in 0..D {
            v_cov[i] += spatial_metric[(i, j)] * velocity[j];
        }
    }

    // v^2 = gamma_ij * v^i * v^j
    let mut v_squared = 0.0;

    for i in 0..D {
        v_squared += velocity[i] * v_cov[i];
    }

    if !v_squared.is_finite()
        || !(0.0..1.0).contains(&v_squared)
    {
        return Err(FourVelocityError::InvalidSpeed);
    }

    // Lorentz factor
    let w = 1.0 / (1.0 - v_squared).sqrt();

    // beta^i * v_i
    let mut beta_dot_v = 0.0;

    for i in 0..D {
        beta_dot_v += beta[i] * v_cov[i];
    }

    // Covariant four-velocity (u_0, u_i)
    let u_0 = w * (-alpha + beta_dot_v);
    let mut u_cov: Vector<D> = Vector::zero();

    for i in 0..D {
        u_cov[i] = w * v_cov[i];
    }

    Ok((u_0, u_cov))
}

/// Build the covariant four-velocity (u_0, u_i) on the whole grid.
///
/// Applies `four_velocity_covariant` pointwise; the first failing point
/// aborts the field build and its error is returned.
pub fn four_velocity_covariant_field<const D: usize>(
    alpha: &Field<f64, D>,
    beta: &Field<Vector<D>, D>,
    spatial_metric: &Field<Tensor2<D>, D>,
    velocity: &Field<Vector<D>, D>,
) -> Result<(Field<f64, D>, Field<Vector<D>, D>), FourVelocityError> {
    assert_eq!(beta.grid.shape, alpha.grid.shape);
    assert_eq!(spatial_metric.grid.shape, alpha.grid.shape);
    assert_eq!(velocity.grid.shape, alpha.grid.shape);

    let mut u_0 = Field::new(alpha.grid.clone(), 0.0);
    let mut u_cov = Field::new(alpha.grid.clone(), Vector::zero());
    let mut err: Option<FourVelocityError> = None;

    alpha.grid.for_each_index(|idx| {
        if err.is_some() {
            return;
        }
        match four_velocity_covariant(
            *alpha.get(idx),
            beta.get(idx),
            spatial_metric.get(idx),
            velocity.get(idx),
        ) {
            Ok((t, s)) => {
                u_0.set(idx, t);
                u_cov.set(idx, s);
            }
            Err(e) => err = Some(e),
        }
    });

    match err {
        Some(e) => Err(e),
        None => Ok((u_0, u_cov)),
    }
}