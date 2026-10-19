#[cfg(test)]
mod tests {
    use einstein_engine::geometry::christoffel::christoffel_symbols;
    use einstein_engine::geometry::metric::{LapseField, ShiftField, SliceMetric};
    use einstein_engine::geometry::tensor::{det, invert, Tensor2, Vector};
    use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
    use einstein_engine::grid::{Field, Grid};
    use einstein_engine::numerics::interpolation::interpolate_christoffel;
    use einstein_engine::numerics::rk4::AdmState;
    use einstein_engine::physics::geodesics::{bssn_stage_connections, geodesic_rhs, StageConnections};
    use einstein_engine::physics::particles::Particles;
    use einstein_engine::relativity::bssn::{stage_physical_adm, BssnState};
    use einstein_engine::relativity::matter::MatterSources;
    use std::sync::Arc;

    const D: usize = 3;
    const N: usize = 4;

    fn periodic() -> (BoundaryConfig<D>, BoundaryValues<D>) {
        (BoundaryConfig::<D>::periodic(), BoundaryValues::<D>::zeros())
    }

    fn identity() -> Tensor2<D> {
        let mut t = Tensor2::<D>::zero();
        for i in 0..D {
            t[(i, i)] = 1.0;
        }
        t
    }

    // a uniform homogeneous BSSN slice:
    //   phi = 0.15, gtilde = delta, A tilde = diag(0.1, -0.05, -0.05),
    //   K = -0.25, alpha = 1, beta = (0.2, -0.1, 0.05)
    // Every field is spatially constant so all central differences
    // evaluate to exact zeros and the only motion is the stage-rate one.
    fn uniform_dyn_state(grid: Arc<Grid<D>>) -> BssnState<D> {
        let mut s = BssnState::<D>::minkowski(grid);
        s.grid.for_each_index(|idx| {
            s.phi.set(idx, 0.15);

            let mut at = Tensor2::<D>::zero();
            at[(0, 0)] = 0.1;
            at[(1, 1)] = -0.05;
            at[(2, 2)] = -0.05;
            s.conformal_a.set(idx, at);

            s.trace_k.set(idx, -0.25);

            let mut b = Vector::<D>::zero();
            b[0] = 0.2;
            b[1] = -0.1;
            b[2] = 0.05;
            s.shift.set(idx, b);
        });
        s
    }

    // a uniform expanding slice in the vacuum BSSN system, with the
    // algebraic constraints satisfied exactly (gtilde = delta has det 1,
    // A tilde = 0): gtilde stays delta and A tilde stays zero through
    // every stage, while phi, alpha and K evolve genuinely.
    fn uniform_expanding_state(grid: Arc<Grid<D>>) -> BssnState<D> {
        let mut s = BssnState::<D>::minkowski(grid);
        s.grid.for_each_index(|idx| {
            s.phi.set(idx, 0.1);
            s.trace_k.set(idx, -0.3);
        });
        s
    }

    fn bssn_state_eq(a: &BssnState<D>, b: &BssnState<D>, tol: f64, what: &str) {
        a.grid.for_each_index(|idx| {
            assert!(
                (*a.phi.get(idx) - *b.phi.get(idx)).abs() <= tol,
                "{what}: phi differs at {:?}",
                idx
            );
            assert!(
                (*a.trace_k.get(idx) - *b.trace_k.get(idx)).abs() <= tol,
                "{what}: trace_k differs at {:?}",
                idx
            );
            assert!(
                (*a.lapse.get(idx) - *b.lapse.get(idx)).abs() <= tol,
                "{what}: lapse differs at {:?}",
                idx
            );
            for i in 0..D {
                for j in 0..D {
                    assert!(
                        (a.conformal_metric.get(idx)[(i, j)] - b.conformal_metric.get(idx)[(i, j)]).abs()
                            <= tol,
                        "{what}: conformal metric ({}, {}) differs at {:?}",
                        i,
                        j,
                        idx
                    );
                    assert!(
                        (a.conformal_a.get(idx)[(i, j)] - b.conformal_a.get(idx)[(i, j)]).abs()
                            <= tol,
                        "{what}: conformal a ({}, {}) differs at {:?}",
                        i,
                        j,
                        idx
                    );
                }
                assert_eq!(
                    a.shift.get(idx)[i],
                    b.shift.get(idx)[i],
                    "{what}: shift[{}] differs at {:?}",
                    i,
                    idx
                );
                assert_eq!(
                    a.conformal_gamma.get(idx)[i],
                    b.conformal_gamma.get(idx)[i],
                    "{what}: gamma fn [{}] differs at {:?}",
                    i,
                    idx
                );
            }
        });
    }

    // test-local replica of `BssnState::enforce_constraints`, so the hand
    // assembly of one RK4 step can go through the same algebraic
    // projections without touching private methods
    fn project_constraints(s: &mut BssnState<D>) {
        s.grid.for_each_index(|idx| {
            let gtilde = *s.conformal_metric.get(idx);
            let mut m = [[0.0f64; D]; D];
            for i in 0..D {
                for j in 0..D {
                    m[i][j] = gtilde[(i, j)];
                }
            }
            let d = det(&m);
            let scale = d.powf(-1.0 / 3.0);
            let mut g = Tensor2::<D>::zero();
            for i in 0..D {
                for j in 0..D {
                    g[(i, j)] = scale * gtilde[(i, j)];
                }
            }
            s.conformal_metric.set(idx, g);

            let gi = invert(&matrix(&g));
            let at = *s.conformal_a.get(idx);
            let mut tr = 0.0;
            for i in 0..D {
                for j in 0..D {
                    tr += gi[i][j] * at[(i, j)];
                }
            }
            let mut a = Tensor2::<D>::zero();
            for i in 0..D {
                for j in 0..D {
                    a[(i, j)] = at[(i, j)] - g[(i, j)] * tr / 3.0;
                }
            }
            s.conformal_a.set(idx, a);
        });
    }

    fn matrix<const D: usize>(t: &Tensor2<D>) -> [[f64; D]; D] {
        let mut m = [[0.0; D]; D];
        for i in 0..D {
            for j in 0..D {
                m[i][j] = t[(i, j)];
            }
        }
        m
    }

    // hand-rolled RK4 through the public geodesic primitives, with the
    // same fused final combination the engine stepper uses (float
    // addition is not associative, so a chained axpy accumulation can
    // differ in the last ulp)
    fn staged_hand_rk4<const D: usize, const N: usize>(
        start: &einstein_engine::physics::geodesics::GeodesicState<D>,
        dt: f64,
        conns: &StageConnections<N, D>,
    ) -> einstein_engine::physics::geodesics::GeodesicState<D> {
        use einstein_engine::physics::geodesics::GeodesicState;
        let axpy = |s: &GeodesicState<D>,
                    a: f64,
                    d: &GeodesicState<D>|
         -> GeodesicState<D> {
            let mut out = *s;
            for i in 0..D {
                out.pos[i] += a * d.pos[i];
                out.vel[i] += a * d.vel[i];
            }
            out
        };

        let k1 = geodesic_rhs::<D, N>(
            start,
            &interpolate_christoffel(&conns.connections[0], start.pos).unwrap(),
        );
        let y2 = axpy(start, 0.5 * dt, &k1);
        let k2 = geodesic_rhs::<D, N>(
            &y2,
            &interpolate_christoffel(&conns.connections[1], y2.pos).unwrap(),
        );
        let y3 = axpy(start, 0.5 * dt, &k2);
        let k3 = geodesic_rhs::<D, N>(
            &y3,
            &interpolate_christoffel(&conns.connections[2], y3.pos).unwrap(),
        );
        let y4 = axpy(start, dt, &k3);
        let k4 = geodesic_rhs::<D, N>(
            &y4,
            &interpolate_christoffel(&conns.connections[3], y4.pos).unwrap(),
        );

        let mut expected = *start;
        for i in 0..D {
            expected.pos[i] += dt / 6.0 * k1.pos[i]
                + dt / 3.0 * k2.pos[i]
                + dt / 3.0 * k3.pos[i]
                + dt / 6.0 * k4.pos[i];
            expected.vel[i] += dt / 6.0 * k1.vel[i]
                + dt / 3.0 * k2.vel[i]
                + dt / 3.0 * k3.vel[i]
                + dt / 6.0 * k4.vel[i];
        }
        expected
    }

    // 1. ADM/BSSN physical metric agreement for equivalent initial data:
    //    from_adm -> to_adm is the identity on gamma_ij, K_ij, alpha and
    //    beta, and the algebraic constraints det(gtilde) = 1 and
    //    trace(gtilde^-1 A tilde) = 0 hold on the BSSN side.
    #[test]
    fn adm_bssn_round_trip_agrees() {
        let grid = Arc::new(Grid::new([7, 7, 7], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
        let (config, values) = periodic();

        // physical slice: spatially varying gamma_xx (smooth-periodic),
        // constant gamma_yy/gamma_zz, trace-free A plus a trace part,
        // boosted constant shift, constant lapse
        let mut gamma: SliceMetric<D> = Field::new(grid.clone(), Tensor2::zero());
        let mut k: Field<Tensor2<D>, D> = Field::new(grid.clone(), Tensor2::zero());
        grid.for_each_index(|idx| {
            let x = idx[0] as f64 * grid.spacing[0];
            let fx = 1.2 + 0.15 * (2.0 * std::f64::consts::PI * x / 3.0).sin();
            let mut g = Tensor2::<D>::zero();
            g[(0, 0)] = fx;
            g[(1, 1)] = 1.1;
            g[(2, 2)] = 0.95;
            gamma.set(idx, g);

            let mut kk = Tensor2::<D>::zero();
            // kk = 0.07 * gamma + trace-free wrt gamma: 0.18 - 0.18 + 0
            kk[(0, 0)] = fx * 0.25;
            kk[(1, 1)] = 1.1 * -0.11;
            kk[(2, 2)] = 0.95 * 0.07;
            k.set(idx, kk);
        });

        let adm = AdmState {
            grid: grid.clone(),
            gamma,
            k,
            lapse: LapseField::new(grid.clone(), 1.05),
            shift: ShiftField::new(
                grid.clone(),
                Vector::new([0.25, 0.03, -0.1]),
            ),
        };

        let bssn = BssnState::<D>::from_adm(&adm, &config, &values);

        // algebraic constraints of the split
        bssn.grid.for_each_index(|idx| {
            let gtilde = *bssn.conformal_metric.get(idx);
            let d = det(&matrix(&gtilde));
            assert!(
                (d - 1.0).abs() < 1e-12,
                "det gtilde must be 1, got {d} at {:?}",
                idx
            );

            let gi = invert(&matrix(&gtilde));
            let at = *bssn.conformal_a.get(idx);
            let mut tr = 0.0;
            for i in 0..D {
                for j in 0..D {
                    tr += gi[i][j] * at[(i, j)];
                }
            }
            assert!(
                tr.abs() < 1e-12,
                "trace of A tilde must vanish, got {tr} at {:?}",
                idx
            );
        });

        // round trip
        let back = bssn.to_adm();
        back.grid.for_each_index(|idx| {
            for i in 0..D {
                for j in 0..D {
                    assert!(
                        (adm.gamma.get(idx)[(i, j)] - back.gamma.get(idx)[(i, j)]).abs() < 1e-12,
                        "gamma round trip differs at ({i},{j}), {:?}",
                        idx
                    );
                    assert!(
                        (adm.k.get(idx)[(i, j)] - back.k.get(idx)[(i, j)]).abs() < 1e-12,
                        "K round trip differs at ({i},{j}), {:?}",
                        idx
                    );
                }
            }
            assert_eq!(*adm.lapse.get(idx), *back.lapse.get(idx));
            assert_eq!(adm.shift.get(idx)[0], back.shift.get(idx)[0]);
            assert_eq!(adm.shift.get(idx)[1], back.shift.get(idx)[1]);
            assert_eq!(adm.shift.get(idx)[2], back.shift.get(idx)[2]);
        });
    }

    // 2. the derived physical metric rate e^(4 phi)(4 d_t phi gtilde +
    //    d_t gtilde) against a finite difference of the actual physical
    //    metric composition along the BSSN stage direction; the lapse and
    //    shift rates are exact because axpy is linear in those fields.
    #[test]
    fn bssn_stage_time_derivatives_match_finite_difference() {
        let grid = Arc::new(Grid::new([3, 3, 3], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
        let (config, values) = periodic();

        let y0 = uniform_dyn_state(grid.clone());
        let k1 = y0.rhs(&config, &values);

        let (_adm, dgamma_dt, dshift_dt, dlapse_dt) = stage_physical_adm::<D>(&y0, &k1);

        // central difference along the stage direction y +- eps * k
        let eps = 1e-3;
        let y_plus = y0.axpy(eps, &k1);
        let y_minus = y0.axpy(-eps, &k1);
        let g_plus = y_plus.to_adm().gamma;
        let g_minus = y_minus.to_adm().gamma;

        y0.grid.for_each_index(|idx| {
            for i in 0..D {
                for j in 0..D {
                    let fd = (g_plus.get(idx)[(i, j)] - g_minus.get(idx)[(i, j)]) / (2.0 * eps);
                    let derived = dgamma_dt.get(idx)[(i, j)];
                    assert!(
                        (fd - derived).abs() < 1e-7,
                        "d_t gamma_{}{}: fd {fd} vs derived {derived} at {:?}",
                        i,
                        j,
                        idx
                    );
                }
            }

            // lapse rate: axpy is linear in the lapse, so the finite
            // difference is exact
            let fd_lapse = (*y_plus.lapse.get(idx) - *y_minus.lapse.get(idx)) / (2.0 * eps);
            assert!(
                (fd_lapse - *dlapse_dt.get(idx)).abs() < 1e-13,
                "d_t alpha: fd {fd_lapse} vs derived {} at {:?}",
                *dlapse_dt.get(idx),
                idx
            );

            // shift rate: frozen, both sides exactly zero
            for i in 0..D {
                assert_eq!(dshift_dt.get(idx)[i], 0.0);
            }
        });
    }

    // 3. connection construction in the static limit: a BSSN slice with
    //    K = A tilde = 0, alpha = 1, beta = 0 and a spatially curved
    //    gtilde has a first stage rate with d_t phi = 0 and
    //    d_t gtilde = -2 alpha A tilde = 0, so stage bundle 0 must equal
    //    the plain static constructor on the same physical geometry.
    //    Bundles 1..3 diverge because the A tilde equation feeds on the
    //    slice Ricci tensor (R tilde + R phi), which is what makes the
    //    coupled path genuinely live; that divergence is asserted too.
    #[test]
    fn bssn_static_limit_matches_adm_christoffels() {
        let grid = Arc::new(Grid::new([9, 4, 4], [0.25, 0.25, 0.25], [0.0, 0.0, 0.0]));
        let (config, values) = periodic();

        let mut y0 = BssnState::<D>::minkowski(grid.clone());
        y0.grid.for_each_index(|idx| {
            let x = idx[0] as f64 * grid.spacing[0];
            let mut g = identity();
            g[(0, 0)] = 1.2 + 0.1 * (std::f64::consts::PI * x).sin();
            y0.conformal_metric.set(idx, g);
            y0.phi.set(idx, 0.3);
        });

        let stages = y0.step_rk4_stages(0.05, &config, &values);
        let connections = bssn_stage_connections::<D, N>(&stages, &config, &values);

        let adm0 = stages.y0.to_adm();
        let static_ref = christoffel_symbols::<D, N>(
            &adm0.lapse,
            &adm0.shift,
            &adm0.gamma,
            &config,
            &values,
        );

        // stage 0: the static limit reproduces the static constructor
        // including its Christoffel-parity, exactly at every point
        let bundle0 = &connections.connections[0];
        grid.for_each_index(|idx| {
            assert_eq!(
                *bundle0.get(idx),
                *static_ref.get(idx),
                "static-limit bundle 0 diverges from the static constructor at {:?}",
                idx
            );
        });

        // later stages: the slice Ricci tensor drives A tilde, A tilde
        // drives d_t gtilde, and the later bundles must feel it
        let mut max_div = 0.0f64;
        for bundle in connections.connections.iter().skip(1) {
            grid.for_each_index(|idx| {
                let a = bundle.get(idx);
                let b = static_ref.get(idx);
                for sa in 0..N {
                    for sb in 0..N {
                        for sc in 0..N {
                            max_div = max_div.max((a.data[sa][sb][sc] - b.data[sa][sb][sc]).abs());
                        }
                    }
                }
            });
        }
        assert!(
            max_div > 1e-9,
            "later bundles must measurably leave the static limit, max |delta| = {max_div}"
        );
    }

    // 4. stage structure of the new API on uniform data, where every
    //    enforcement is an exact no-op (gtilde stays delta, A tilde stays
    //    zero, det stays exactly 1), so all eight stage slots can be
    //    compared against a hand assembly through the public `axpy`,
    //    `rhs` and this file's constraint projections, and `collapse`
    //    against the classic stepper bit-for-bit.
    //    `collapse` against the classic stepper bit-for-bit.
    #[test]
    fn bssn_stage_taxonomy_matches_hand_assembly() {
        let grid = Arc::new(Grid::new([3, 3, 3], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
        let (config, values) = periodic();
        let dt = 0.02;

        let x0 = uniform_expanding_state(grid.clone());
        let steps = x0.step_rk4_stages(dt, &config, &values);

        // every rate is the right-hand side of the state it is paired
        // with, bit for bit (identical rhs calls on identical states)
        let k1_ref = x0.rhs(&config, &values);
        bssn_state_eq(&steps.k1, &k1_ref, 0.0, "k1");

        let mut y2c = x0.axpy(0.5 * dt, &steps.k1);
        project_constraints(&mut y2c);
        bssn_state_eq(&steps.y2, &y2c, 0.0, "y2");
        let k2_ref = y2c.rhs(&config, &values);
        bssn_state_eq(&steps.k2, &k2_ref, 0.0, "k2");

        let mut y3c = x0.axpy(0.5 * dt, &steps.k2);
        project_constraints(&mut y3c);
        bssn_state_eq(&steps.y3, &y3c, 0.0, "y3");
        let k3_ref = y3c.rhs(&config, &values);
        bssn_state_eq(&steps.k3, &k3_ref, 0.0, "k3");

        let mut y4c = x0.axpy(dt, &steps.k3);
        project_constraints(&mut y4c);
        bssn_state_eq(&steps.y4, &y4c, 0.0, "y4");
        let k4_ref = y4c.rhs(&config, &values);
        bssn_state_eq(&steps.k4, &k4_ref, 0.0, "k4");

        // collapse reproduces the classic stepper exactly
        let next = steps.collapse(&config, &values);
        let next_ref = x0.step_rk4_with_matter(
            dt,
            &MatterSources::vacuum(&x0.grid),
            &config,
            &values,
        );
        bssn_state_eq(&next, &next_ref, 0.0, "collapse");

        // the vacuum wrapper really is the vacuum specialization
        let vac = x0.step_rk4_stages(dt, &config, &values);
        bssn_state_eq(&vac.y0, &steps.y0, 0.0, "vacuum wrapper y0");
        bssn_state_eq(&vac.k1, &steps.k1, 0.0, "vacuum wrapper k1");
    }

    // 5. correct BSSN stage-to-connection pairing: on the uniform
    //    expanding slice every stage keeps gtilde = delta and
    //    A tilde = 0 exactly, so bundle i must encode, at stage time i,
    //      Gamma^x_0x = 2 d_t phi(y_i)      (since d_t gamma = 4 d_t phi gamma)
    //      Gamma^0_00 = d_t alpha(y_i) / alpha(y_i)
    //    and nothing else. The four stage rates differ at O(dt) in d_t
    //    phi, so a cross-wired pairing would fail these assertions.
    //    The spatial-spatial components must vanish exactly.
    #[test]
    fn bssn_bundles_carry_their_stage_rates() {
        let grid = Arc::new(Grid::new([3, 3, 3], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
        let (config, values) = periodic();

        let x0 = uniform_expanding_state(grid.clone());
        let steps = x0.step_rk4_stages(0.02, &config, &values);
        let connections = bssn_stage_connections::<D, N>(&steps, &config, &values);
        assert_eq!(connections.dt, steps.dt);

        let ys = [&steps.y0, &steps.y2, &steps.y3, &steps.y4];
        let ks = [&steps.k1, &steps.k2, &steps.k3, &steps.k4];

        // stage purity: the algebraic slice stays exactly flat/homogeneous
        for (i, y) in ys.iter().enumerate() {
            y.grid.for_each_index(|idx| {
                let g = *y.conformal_metric.get(idx);
                for a in 0..D {
                    for b in 0..D {
                        let want = if a == b { 1.0 } else { 0.0 };
                        assert_eq!(
                            g[(a, b)],
                            want,
                            "stage {i}: gtilde lost its delta form at {:?}",
                            idx
                        );
                    }
                }
                let at = *y.conformal_a.get(idx);
                for a in 0..D {
                    for b in 0..D {
                        assert_eq!(
                            at[(a, b)],
                            0.0,
                            "stage {i}: A tilde acquired a component at {:?}",
                            idx
                        );
                    }
                }
            });
        }

        // the four stage rate values must actually differ, otherwise the
        // pairing assertions below would be insensitive to wiring errors
        for a in 0..4 {
            for b in a + 1..4 {
                let diff = (*ks[a].phi.get([1, 1, 1]) - *ks[b].phi.get([1, 1, 1])).abs();
                assert!(
                    diff > 5e-8,
                    "stage phi rates too close: k{a} vs k{b}: {diff}"
                );
            }
        }

        for i in 0..4 {
            let bundle = connections.connections[i].get([1, 1, 1]);
            let phi_dot = *ks[i].phi.get([1, 1, 1]);
            let lapse_dot = *ks[i].lapse.get([1, 1, 1]);
            let alpha = *ys[i].lapse.get([1, 1, 1]);

            // Gamma^x_0x = 1/2 d_t gamma_xx / gamma_xx
            //   d_t gamma_xx = e^{4 phi} * 4 d_t phi
            //   gamma_xx = e^{4 phi}
            // -> Gamma^x_0x = 2 d_t phi
            assert!(
                (bundle.data[1][0][1] - 2.0 * phi_dot).abs() < 1e-12,
                "bundle {i}: Gamma^x_0x = {} vs 2 d_t phi = {}",
                bundle.data[1][0][1],
                2.0 * phi_dot
            );

            // Gamma^0_00 = alpha_dot / alpha (1+log gauge)
            assert!(
                (bundle.data[0][0][0] - lapse_dot / alpha).abs() < 1e-12,
                "bundle {i}: Gamma^0_00 = {} vs d_t alpha/alpha = {}",
                bundle.data[0][0][0],
                lapse_dot / alpha
            );

            // spatial-spatial components: flat uniform slice, exactly zero
            for a in 1..N {
                for b in 1..N {
                    for c in 1..N {
                        assert_eq!(
                            bundle.data[a][b][c],
                            0.0,
                            "bundle {i}: spatial component ({a},{b},{c}) not zero"
                        );
                    }
                }
            }
        }
    }

    // 6. a time-dependent geodesic run on the genuinely evolving BSSN
    //    path: two chained steps of the real vacuum BSSN evolution
    //    (growing phi through -alpha K/6, 1+log lapse through -2 alpha K),
    //    stage connections built from the BSSN stage pairs, particles
    //    stepped through the shared batch-commit machinery, and the
    //    committed states compared against a hand-rolled RK4 over the
    //    same bundles through the public primitives.
    #[test]
    fn bssn_coupled_geodesic_run_matches_hand_and_form() {
        let grid = Arc::new(Grid::new([4, 4, 4], [0.25, 0.25, 0.25], [0.0, 0.0, 0.0]));
        let (config, values) = periodic();

        let x0 = uniform_expanding_state(grid.clone());
        let dt = 0.01;

        let stages = x0.step_rk4_with_matter_stages(
            dt,
            &MatterSources::vacuum(&x0.grid),
            &config,
            &values,
        );
        let connections = bssn_stage_connections::<D, N>(&stages, &config, &values);

        // stage purity across the coupling path as well
        for (i, y) in [&stages.y0, &stages.y2, &stages.y3, &stages.y4]
            .iter()
            .enumerate()
        {
            y.grid.for_each_index(|idx| {
                let g = *y.conformal_metric.get(idx);
                assert_eq!(g[(0, 0)], 1.0, "stage {i} gtilde_xx");
                assert_eq!(g[(1, 1)], 1.0, "stage {i} gtilde_yy");
                let at = *y.conformal_a.get(idx);
                assert_eq!(at[(0, 0)], 0.0, "stage {i} A tilde_xx");
            });
        }

        let start = einstein_engine::physics::geodesics::GeodesicState {
            pos: [0.5, 0.5, 0.5],
            vel: [0.1, 0.0, 0.01],
        };

        let mut particles: Particles<D> = Particles::new();
        particles.spawn(0.0, start).unwrap();

        particles.step_all_with_connections(0.0, &connections).unwrap();

        let committed = particles.get(0).unwrap();
        let expected = staged_hand_rk4::<D, N>(&start, dt, &connections);
        assert_eq!(committed.state.pos, expected.pos, "committed pos vs hand RK4");
        assert_eq!(committed.state.vel, expected.vel, "committed vel vs hand RK4");

        // clock tracks the ADM-sized step exactly
        assert_eq!(committed.time, dt);

        // second chained step from the collapsed BSSN state
        let x1 = stages.collapse(&config, &values);
        let stages2 = x1.step_rk4_with_matter_stages(
            dt,
            &MatterSources::vacuum(&x0.grid),
            &config,
            &values,
        );
        let connections2 = bssn_stage_connections::<D, N>(&stages2, &config, &values);
        particles
            .step_all_with_connections(committed.time, &connections2)
            .unwrap();

        let committed = particles.get(0).unwrap();
        let expected2 = staged_hand_rk4::<D, N>(
            &expected,
            connections2.dt,
            &connections2,
        );
        assert_eq!(committed.state.pos, expected2.pos, "step 2 pos vs hand RK4");
        assert_eq!(committed.state.vel, expected2.vel, "step 2 vel vs hand RK4");
        assert!((committed.time - 2.0 * dt).abs() < 1e-15);

        // the expansion must actually act on the trajectory. In this
        // 1+log gauge with a frozen shift the layer is expanding
        // (K < 0) while the lapse grows (alpha_dot/alpha = -2K = 0.6):
        // the bundle rates fix
        //   d v^i/dt = v^i * (Gamma^0_00 - 2 * (2 - |v|^2) * d_t phi)
        // with Gamma^0_00 = 0.6 and d_t phi = 0.05, so the gauge term
        // dominates the metric term and the coordinate velocity grows
        // by O(0.4 * v * t) >> 1e-6 over two steps.
        assert!(
            committed.state.vel[0] > start.vel[0] + 1e-6,
            "x velocity must be gauge-accelerated, got {} vs {}",
            committed.state.vel[0],
            start.vel[0]
        );
    }
}
