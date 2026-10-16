//! Dust evolution equations (the numerics half of [`crate::physics::dust`]).
//!
//! d(Dtilde)/dt = -d_i [Dtilde (alpha v^i - beta^i)] and
//! d(Ptilde_j)/dt = -d_i [Ptilde_j (alpha v^i - beta^i)] + source_j with
//! source_j = (alpha sqrt(gamma)/2) T^{mu nu} d_j g_{munu},
//! T^{00} = rho W^2/alpha^2,
//! T^{0k} = rho W^2 (v^k-beta^k/alpha)/alpha,
//! T^{kl} = rho W^2 (v^k-beta^k/alpha)(v^l-beta^l/alpha).
//!
//! The geometry is taken as fixed over each derivative evaluation: coupled
//! steppers pass the intermediate-stage geometry. Central differences, so
//! a uniform field has exactly zero RHS.

use crate::geometry::metric::{LapseField, ShiftField, SliceMetric};
use crate::geometry::tensor::{det, Tensor2, Vector};
use crate::grid::boundary::{BoundaryConfig, BoundaryValues};
use crate::grid::Field;
use crate::numerics::derivatives::central_partial;
use crate::physics::dust::DustState;

/// copy one (i, j) component of a tensor field into a scalar field
fn tensor_component<const D: usize>(
    field: &Field<Tensor2<D>, D>,
    i: usize,
    j: usize,
) -> Field<f64, D> {
    let mut out = Field::zeros(field.grid.clone());
    field
        .grid
        .for_each_index(|idx| out.set(idx, field.get(idx)[(i, j)]));
    out
}

/// Right-hand side of the dust evolution equations on fixed geometry.
pub fn dust_rhs<const D: usize>(
    state: &DustState<D>,
    lapse: &LapseField<D>,
    shift: &ShiftField<D>,
    spatial: &SliceMetric<D>,
    boundary: &BoundaryConfig<D>,
    boundary_values: &BoundaryValues<D>,
) -> DustState<D> {
    let (rho_f, vel_f) = state.primitives(spatial);

    // continuity: d_t Dtilde = -d_i F^i, F^i = Dtilde (alpha v^i - beta^i)
    let mut div = Field::new(state.grid.clone(), 0.0);
    for i in 0..D {
        let mut flux = Field::new(state.grid.clone(), 0.0);
        state.grid.for_each_index(|idx| {
            flux.set(
                idx,
                state.densitized_rest_mass.get(idx)
                    * (*lapse.get(idx) * vel_f.get(idx)[i] - shift.get(idx)[i]),
            );
        });
        let d_flux = central_partial(&flux, i, boundary, boundary_values);
        state.grid.for_each_index(|idx| {
            div.set(idx, div.get(idx) + d_flux.get(idx));
        });
    }
    let mut drhs = Field::new(state.grid.clone(), 0.0);
    state.grid.for_each_index(|idx| {
        drhs.set(idx, -div.get(idx));
    });

    // metric derivative fields for the momentum source
    let mut d_alpha: Vec<Field<f64, D>> = Vec::with_capacity(D);
    for k in 0..D {
        d_alpha.push(central_partial(lapse, k, boundary, boundary_values));
    }
    // d_k beta_m with beta_m = gamma_mk beta^k
    let mut d_beta_cov: Vec<Vec<Field<f64, D>>> = Vec::with_capacity(D);
    for m in 0..D {
        let mut beta_m = Field::new(state.grid.clone(), 0.0);
        state.grid.for_each_index(|idx| {
            let mut s = 0.0;
            for k in 0..D {
                s += spatial.get(idx)[(m, k)] * shift.get(idx)[k];
            }
            beta_m.set(idx, s);
        });
        let mut row = Vec::with_capacity(D);
        for k in 0..D {
            row.push(central_partial(&beta_m, k, boundary, boundary_values));
        }
        d_beta_cov.push(row);
    }
    // d_k (beta_m beta^m)
    let mut betasq = Field::new(state.grid.clone(), 0.0);
    state.grid.for_each_index(|idx| {
        let mut s = 0.0;
        for m in 0..D {
            let mut low = 0.0;
            for k in 0..D {
                low += spatial.get(idx)[(m, k)] * shift.get(idx)[k];
            }
            s += low * shift.get(idx)[m];
        }
        betasq.set(idx, s);
    });
    let mut d_betasq: Vec<Field<f64, D>> = Vec::with_capacity(D);
    for k in 0..D {
        d_betasq.push(central_partial(&betasq, k, boundary, boundary_values));
    }
    // d_k gamma_mn
    let mut d_gamma: Vec<Vec<Vec<Field<f64, D>>>> = Vec::with_capacity(D);
    for m in 0..D {
        let mut by_n = Vec::with_capacity(D);
        for n in 0..D {
            let comp = tensor_component(spatial, m, n);
            let mut by_k = Vec::with_capacity(D);
            for k in 0..D {
                by_k.push(central_partial(&comp, k, boundary, boundary_values));
            }
            by_n.push(by_k);
        }
        d_gamma.push(by_n);
    }

    // momentum flux divergence: d_i G^i_j, G^i_j = Ptilde_j (alpha v^i - beta^i)
    let mut div_mom: Vec<Field<f64, D>> = Vec::with_capacity(D);
    for _ in 0..D {
        div_mom.push(Field::new(state.grid.clone(), 0.0));
    }
    for i in 0..D {
        for j in 0..D {
            let mut flux = Field::new(state.grid.clone(), 0.0);
            state.grid.for_each_index(|idx| {
                flux.set(
                    idx,
                    state.densitized_momentum.get(idx)[j]
                        * (*lapse.get(idx) * vel_f.get(idx)[i] - shift.get(idx)[i]),
                );
            });
            let d_flux = central_partial(&flux, i, boundary, boundary_values);
            state.grid.for_each_index(|idx| {
                let v = *div_mom[j].get(idx) + *d_flux.get(idx);
                div_mom[j].set(idx, v);
            });
        }
    }

    let mut mrhs = Field::new(state.grid.clone(), Vector::<D>::zero());
    state.grid.for_each_index(|idx| {
        let alpha = *lapse.get(idx);
        let beta = *shift.get(idx);
        let r = *rho_f.get(idx);
        let vel = *vel_f.get(idx);

        let mut gmat = [[0.0f64; D]; D];
        for i in 0..D {
            for j in 0..D {
                gmat[i][j] = spatial.get(idx)[(i, j)];
            }
        }
        let gdet = det(&gmat);
        assert!(gdet > 0.0, "the spatial metric must be positive definite");
        let sqrtg = gdet.sqrt();

        let mut v_squared = 0.0;
        for i in 0..D {
            for j in 0..D {
                v_squared += vel[i] * spatial.get(idx)[(i, j)] * vel[j];
            }
        }
        let w = 1.0 / (1.0 - v_squared).sqrt();

        let t00 = r * w * w / (alpha * alpha);
        let mut t0 = [0.0f64; D];
        for k in 0..D {
            t0[k] = r * w * w * (vel[k] - beta[k] / alpha) / alpha;
        }

        let mut mv = Vector::<D>::zero();
        for j in 0..D {
            let mut src = t00
                * (-2.0 * alpha * *d_alpha[j].get(idx) + *d_betasq[j].get(idx));
            for k in 0..D {
                src += 2.0 * t0[k] * *d_beta_cov[k][j].get(idx);
            }
            for k in 0..D {
                for l in 0..D {
                    let tkl = r * w * w * (vel[k] - beta[k] / alpha)
                        * (vel[l] - beta[l] / alpha);
                    src += tkl * *d_gamma[k][l][j].get(idx);
                }
            }
            src *= 0.5 * alpha * sqrtg;
            mv[j] = -*div_mom[j].get(idx) + src;
        }
        mrhs.set(idx, mv);
    });

    DustState {
        grid: state.grid.clone(),
        densitized_rest_mass: drhs,
        densitized_momentum: mrhs,
    }
}

/// Advance dust alone by one RK4 timestep with frozen geometry.
///
/// Used for transport tests; the coupled BSSN stepper is what advances
/// matter together with the geometry.
pub fn dust_step_rk4<const D: usize>(
    state: &DustState<D>,
    dt: f64,
    lapse: &LapseField<D>,
    shift: &ShiftField<D>,
    spatial: &SliceMetric<D>,
    boundary: &BoundaryConfig<D>,
    boundary_values: &BoundaryValues<D>,
) -> DustState<D> {
    assert!(dt.is_finite() && dt > 0.0);

    let y0 = state.clone();

    let k1 = dust_rhs(&y0, lapse, shift, spatial, boundary, boundary_values);
    let y2 = y0.axpy(0.5 * dt, &k1);
    let k2 = dust_rhs(&y2, lapse, shift, spatial, boundary, boundary_values);
    let y3 = y0.axpy(0.5 * dt, &k2);
    let k3 = dust_rhs(&y3, lapse, shift, spatial, boundary, boundary_values);
    let y4 = y0.axpy(dt, &k3);
    let k4 = dust_rhs(&y4, lapse, shift, spatial, boundary, boundary_values);

    let mut next = y0;
    next.add_scaled_in_place(dt / 6.0, &k1);
    next.add_scaled_in_place(dt / 3.0, &k2);
    next.add_scaled_in_place(dt / 3.0, &k3);
    next.add_scaled_in_place(dt / 6.0, &k4);
    next
}
