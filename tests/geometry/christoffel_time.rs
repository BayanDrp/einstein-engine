#[cfg(test)]
mod tests {
    use einstein_engine::geometry::christoffel::{
        christoffel_symbols, christoffel_symbols_with_time_derivatives,
    };
    use einstein_engine::geometry::metric::{temporal_metric_derivative, SpacetimeMetric};
    use einstein_engine::geometry::tensor::{Tensor2, Vector};
    use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
    use einstein_engine::grid::{Field, Grid};
    use einstein_engine::relativity::evolution::spatial_metric_rhs;
    use einstein_engine::relativity::gauge::{lapse_rhs, shift_rhs};
    use std::sync::Arc;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // flat Minkowski metric but with the new constructor fed all-zero
    // derivative fields: d_t g_ab = 0, so the time-aware assembly must
    // reproduce the static constructor entry by entry.
    #[test]
    fn zero_time_derivatives_reproduce_static_constructor() {
        let grid = Arc::new(Grid::new([5, 3, 3], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
        let mut lapse: Field<f64, 3> = Field::zeros(grid.clone());
        for i in 0..5 {
            for j in 0..3 {
                for k in 0..3 {
                    lapse.set([i, j, k], 1.0 + i as f64 * 0.5);
                }
            }
        }
        let shift: Field<Vector<3>, 3> = Field::new(grid.clone(), Vector::new([0.5, 0.0, 0.0]));
        let spatial: Field<Tensor2<3>, 3> = Field::new(
            grid.clone(),
            Tensor2::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        );
        let dgamma_dt: Field<Tensor2<3>, 3> = Field::new(grid.clone(), Tensor2::zero());
        let dshift_dt: Field<Vector<3>, 3> = Field::new(grid.clone(), Vector::zero());
        let dlapse_dt: Field<f64, 3> = Field::new(grid.clone(), 0.0);
        let config = BoundaryConfig::<3>::periodic();
        let values = BoundaryValues::<3>::zeros();

        let static_g = christoffel_symbols::<3, 4>(&lapse, &shift, &spatial, &config, &values);
        let timed_g = christoffel_symbols_with_time_derivatives::<3, 4>(
            &lapse, &shift, &spatial, &dgamma_dt, &dshift_dt, &dlapse_dt, &config, &values,
        );

        for i in 0..5 {
            for j in 0..3 {
                for k in 0..3 {
                    assert_eq!(*static_g.get([i, j, k]), *timed_g.get([i, j, k]));
                }
            }
        }
    }

    // the pointwise helper must agree with a central finite difference of
    // the full spacetime metric built by `from_3plus1` along the time
    // axis, for inhomogeneous one-point data. All profiles are linear in
    // t, so g_ab(t) is polynomial and the central-difference error stays
    // far below 1e-8 (the only truncation term comes from the cubic
    // g_00, scaled by h^2 = 1e-8).
    #[test]
    fn temporal_metric_derivative_matches_metric_finite_difference() {
        let alpha_at = |t: f64| 1.0 + 0.3 * t;
        let beta_at = |t: f64| Vector::new([0.2 + 0.4 * t, -0.1 - 0.6 * t, 0.3 * t]);
        let gamma_at = |t: f64| {
            Tensor2::new([
                [1.0 + 0.5 * t, 0.15 + 0.25 * t, 0.05 - 0.1 * t],
                [0.15 + 0.25 * t, 1.0 + 0.35 * t, -0.2 + 0.15 * t],
                [0.05 - 0.1 * t, -0.2 + 0.15 * t, 1.0 + 0.1 * t],
            ])
        };

        let t0 = 0.7;
        let h = 1e-4;

        let g_plus =
            SpacetimeMetric::from_3plus1(alpha_at(t0 + h), &beta_at(t0 + h), &gamma_at(t0 + h))
                .as_matrix::<4>();
        let g_minus =
            SpacetimeMetric::from_3plus1(alpha_at(t0 - h), &beta_at(t0 - h), &gamma_at(t0 - h))
                .as_matrix::<4>();

        // A_ij = d_t gamma_ij and B^i = d_t beta^i at t0: exact
        // derivatives of the linear profiles above
        let a = Tensor2::new([[0.5, 0.25, -0.1], [0.25, 0.35, 0.15], [-0.1, 0.15, 0.1]]);
        let b_dot = Vector::new([0.4, -0.6, 0.3]);

        let analytic = temporal_metric_derivative::<3, 4>(
            alpha_at(t0),
            &beta_at(t0),
            &gamma_at(t0),
            &a,
            &b_dot,
            0.3,
        );

        for au in 0..4 {
            for bu in 0..4 {
                let fd = (g_plus[au][bu] - g_minus[au][bu]) / (2.0 * h);
                assert!(
                    close(analytic[au][bu], fd, 1e-8),
                    "dg[{}][{}] analytic {} vs finite difference {}",
                    au,
                    bu,
                    analytic[au][bu],
                    fd
                );
            }
        }
    }

    // FLRW-style expanding spatially flat slice: gamma_ij = a^2 delta_ij
    // with a = 1.2 uniform, alpha = 1, beta = 0, and A_ij = d_t gamma_ij
    // = 2 a adot delta_ij with adot = 0.5. The exact Christoffel symbols
    // of this background are
    //
    //   Gamma^0_ij = a adot delta_ij = 0.6 delta_ij
    //   Gamma^i_0j = (adot / a) delta^i_j = 0.4166... delta^i_j
    //
    // and everything else zero. The spatial metric is uniform, so every
    // spatial derivative is exactly zero and only the pointwise time
    // terms survive.
    #[test]
    fn flrw_expanding_slice_analytic_symbols() {
        let grid = Arc::new(Grid::new([3, 3, 3], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
        let lapse: Field<f64, 3> = Field::new(grid.clone(), 1.0);
        let shift: Field<Vector<3>, 3> = Field::new(grid.clone(), Vector::zero());

        let a = 1.2;
        let adot = 0.5;
        let mut spatial: Field<Tensor2<3>, 3> = Field::new(grid.clone(), Tensor2::zero());
        let mut dgamma_dt: Field<Tensor2<3>, 3> = Field::new(grid.clone(), Tensor2::zero());
        grid.for_each_index(|idx| {
            let mut g = Tensor2::<3>::zero();
            let mut da = Tensor2::<3>::zero();
            for i in 0..3 {
                g[(i, i)] = a * a;
                da[(i, i)] = 2.0 * a * adot;
            }
            spatial.set(idx, g);
            dgamma_dt.set(idx, da);
        });
        let dshift_dt: Field<Vector<3>, 3> = Field::new(grid.clone(), Vector::zero());
        let dlapse_dt: Field<f64, 3> = Field::new(grid.clone(), 0.0);

        let config = BoundaryConfig::<3>::periodic();
        let values = BoundaryValues::<3>::zeros();

        let g = christoffel_symbols_with_time_derivatives::<3, 4>(
            &lapse, &shift, &spatial, &dgamma_dt, &dshift_dt, &dlapse_dt, &config, &values,
        );

        let flat = adot / a;
        let aadot = a * adot;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let c = *g.get([i, j, k]);
                    for aa in 0..4 {
                        for bb in 0..4 {
                            for cc in 0..4 {
                                let expected = if aa == 0 && bb == cc && bb >= 1 {
                                    aadot
                                } else if aa >= 1
                                    && ((bb == 0 && cc == aa) || (cc == 0 && bb == aa))
                                {
                                    flat
                                } else {
                                    0.0
                                };
                                assert!(
                                    close(c.data[aa][bb][cc], expected, 1e-12),
                                    "at point [{}, {}, {}]: Gamma^[{}][{}][{}] = {} but expected {}",
                                    i, j, k, aa, bb, cc, c.data[aa][bb][cc], expected
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    // the same expanding slice, but with the time derivatives produced by
    // the real ADM machinery: A_ij from spatial_metric_rhs (which is
    // -2 alpha K_ij here, K_ij = -0.3 delta_ij on gamma = 1.44 delta_ij),
    // alpha_dot from the 1+log gauge (-2 alpha K = 1.25 for K = -0.625)
    // and the fixed shift (zero). The exact symbols are
    //
    //   Gamma^0_00 = adot_eff / alpha = 1.25
    //   Gamma^0_ij = A_ij / 2 = 0.3 delta_ij
    //   Gamma^i_0j = gamma^ik A_jk / 2 = 0.2083... delta^i_j
    //
    // with gamma^ik = delta^ik / a^2 = 1 / 1.44. This pins the wiring
    // between the evolution equations and the time-aware Christoffel
    // construction.
    #[test]
    fn evolution_rhs_feeds_time_derivative_constructor() {
        let grid = Arc::new(Grid::new([3, 3, 3], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
        let lapse: Field<f64, 3> = Field::new(grid.clone(), 1.0);
        let shift: Field<Vector<3>, 3> = Field::new(grid.clone(), Vector::zero());

        let mut gamma_data = Tensor2::<3>::zero();
        let mut k_data = Tensor2::<3>::zero();
        for i in 0..3 {
            gamma_data[(i, i)] = 1.44;
            k_data[(i, i)] = -0.3;
        }
        let spatial: Field<Tensor2<3>, 3> = Field::new(grid.clone(), gamma_data);
        let extrinsic: Field<Tensor2<3>, 3> = Field::new(grid.clone(), k_data);

        let config = BoundaryConfig::<3>::periodic();
        let values = BoundaryValues::<3>::zeros();

        let dgamma_dt = spatial_metric_rhs(&lapse, &shift, &spatial, &extrinsic, &config, &values);
        let dlapse_dt = lapse_rhs(&lapse, &shift, &spatial, &extrinsic, &config, &values);
        let dshift_dt = shift_rhs(&shift);

        // the evolvers must have produced the expected A_ij and alpha_dot
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let a = *dgamma_dt.get([i, j, k]);
                    for c in 0..3 {
                        for d in 0..3 {
                            let expected = if c == d { 0.6 } else { 0.0 };
                            assert!(
                                close(a[(c, d)], expected, 1e-12),
                                "A[({},{})] at [{}, {}, {}] = {}",
                                c,
                                d,
                                i,
                                j,
                                k,
                                a[(c, d)]
                            );
                        }
                    }
                    assert!(
                        close(*dlapse_dt.get([i, j, k]), 1.25, 1e-12),
                        "alpha_dot at [{}, {}, {}] = {}",
                        i,
                        j,
                        k,
                        *dlapse_dt.get([i, j, k])
                    );
                }
            }
        }

        let g = christoffel_symbols_with_time_derivatives::<3, 4>(
            &lapse, &shift, &spatial, &dgamma_dt, &dshift_dt, &dlapse_dt, &config, &values,
        );

        let flat = 0.6 / 2.0 / 1.44;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let c = *g.get([i, j, k]);
                    for aa in 0..4 {
                        for bb in 0..4 {
                            for cc in 0..4 {
                                let expected = if aa == 0 && bb == 0 && cc == 0 {
                                    1.25
                                } else if aa == 0 && bb == cc && bb >= 1 {
                                    0.3
                                } else if aa >= 1
                                    && ((bb == 0 && cc == aa) || (cc == 0 && bb == aa))
                                {
                                    flat
                                } else {
                                    0.0
                                };
                                assert!(
                                    close(c.data[aa][bb][cc], expected, 1e-12),
                                    "at point [{}, {}, {}]: Gamma^[{}][{}][{}] = {} but expected {}",
                                    i, j, k, aa, bb, cc, c.data[aa][bb][cc], expected
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    // a NaN planted in one A_ij component at one grid point must poison
    // the symbols at that point rather than silently wash out, and must
    // stay local: the time-derivative fields are never differentiated in
    // space, so neighbouring points are untouched.
    //
    // The slice carries a boost (beta_x = 0.5) so g^{a0} != 0 for
    // a = 0 and a = 1, and the inverse spreads the damage as well.
    // Because NaN * 0 = NaN in IEEE arithmetic, the NaN in A_12 reaches
    // every d_t g_ab block: d_t g_12 and d_t g_21 directly, then every
    // d_t g_0i forming that row (A_ij beta^j multiplies NaN by zero), and
    // d_t g_00 through A_ij beta^i beta^j. The NaN entries of the
    // d_t g_ab matrix are therefore (00), (02), (03), (20), (30), (23),
    // (32) in (row, col), and every Gamma^a_bc whose lower index pair
    // (b, c) matches one of those pairs is NaN, while all other (b, c)
    // combinations stay finite.
    #[test]
    fn nan_in_time_derivative_poisons_point() {
        let grid = Arc::new(Grid::new([5, 3, 3], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
        let mut lapse: Field<f64, 3> = Field::zeros(grid.clone());
        for i in 0..5 {
            for j in 0..3 {
                for k in 0..3 {
                    lapse.set([i, j, k], 1.0 + i as f64 * 0.5);
                }
            }
        }
        let shift: Field<Vector<3>, 3> = Field::new(grid.clone(), Vector::new([0.5, 0.0, 0.0]));
        let spatial: Field<Tensor2<3>, 3> = Field::new(
            grid.clone(),
            Tensor2::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        );

        let mut dgamma_dt: Field<Tensor2<3>, 3> = Field::new(grid.clone(), Tensor2::zero());
        let target = [2usize, 1, 1];
        let mut a = *dgamma_dt.get(target);
        a[(1, 2)] = f64::NAN; // d_t gamma_23
        a[(2, 1)] = f64::NAN; // d_t gamma_32, symmetric partner
        dgamma_dt.set(target, a);
        let dshift_dt: Field<Vector<3>, 3> = Field::new(grid.clone(), Vector::zero());
        let dlapse_dt: Field<f64, 3> = Field::new(grid.clone(), 0.0);

        let config = BoundaryConfig::<3>::periodic();
        let values = BoundaryValues::<3>::zeros();

        let g = christoffel_symbols_with_time_derivatives::<3, 4>(
            &lapse, &shift, &spatial, &dgamma_dt, &dshift_dt, &dlapse_dt, &config, &values,
        );

        let c = *g.get(target);
        // the (row, col) pairs of d_t g_ab that carry NaN
        let poisoned_pairs = [(0, 0), (0, 2), (0, 3), (2, 0), (3, 0), (2, 3), (3, 2)];
        for aa in 0..4 {
            for bb in 0..4 {
                for cc in 0..4 {
                    let expect_nan = poisoned_pairs.contains(&(bb, cc));
                    if expect_nan {
                        assert!(
                            c.data[aa][bb][cc].is_nan(),
                            "Gamma^[{}][{}][{}] should be NaN",
                            aa,
                            bb,
                            cc
                        );
                    } else {
                        assert!(
                            c.data[aa][bb][cc].is_finite(),
                            "Gamma^[{}][{}][{}] should be finite",
                            aa,
                            bb,
                            cc
                        );
                    }
                }
            }
        }

        // the neighbour is untouched: clean data in, clean symbols out
        let neighbour = *g.get([3, 1, 1]);
        for aa in 0..4 {
            for bb in 0..4 {
                for cc in 0..4 {
                    assert!(neighbour.data[aa][bb][cc].is_finite());
                }
            }
        }
    }

    // the three derivative fields must share the grid of the lapse; a
    // mismatched grid must be rejected up front.
    #[test]
    #[should_panic(expected = "same grid")]
    fn mismatched_dgamma_dt_grid_panics() {
        let grid = Arc::new(Grid::new([3, 3, 3], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
        let other = Arc::new(Grid::new([3, 3, 3], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
        let lapse: Field<f64, 3> = Field::new(grid.clone(), 1.0);
        let shift: Field<Vector<3>, 3> = Field::new(grid.clone(), Vector::zero());
        let spatial: Field<Tensor2<3>, 3> = Field::new(
            grid.clone(),
            Tensor2::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        );
        let dgamma_dt: Field<Tensor2<3>, 3> = Field::new(other.clone(), Tensor2::zero());
        let dshift_dt: Field<Vector<3>, 3> =
            Field::new(other.clone(), Vector::new([0.3, 0.0, 0.0]));
        let dlapse_dt: Field<f64, 3> = Field::new(other, 0.4);
        let config = BoundaryConfig::<3>::periodic();
        let values = BoundaryValues::<3>::zeros();

        let _ = christoffel_symbols_with_time_derivatives::<3, 4>(
            &lapse, &shift, &spatial, &dgamma_dt, &dshift_dt, &dlapse_dt, &config, &values,
        );
    }
}
