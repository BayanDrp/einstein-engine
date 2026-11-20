#[cfg(test)]
mod tests {
    use einstein_engine::geometry::christoffel::{
        christoffel_symbols, christoffel_symbols_with_time_derivatives, ChristoffelField,
    };
    use einstein_engine::geometry::metric::{
        ExtrinsicCurvature, LapseField, ShiftField, SliceMetric,
    };
    use einstein_engine::geometry::tensor::{Tensor2, Vector};
    use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
    use einstein_engine::grid::{Field, Grid};
    use einstein_engine::numerics::interpolation::interpolate_christoffel;
    use einstein_engine::numerics::rk4::AdmState;
    use einstein_engine::physics::geodesics::{
        geodesic_rhs, geodesic_stages_step_rk4, geodesic_step_rk4, stage_connections,
        GeodesicState, StageConnections,
    };
    use einstein_engine::physics::particles::{ParticleError, ParticleStatus, Particles};
    use einstein_engine::relativity::matter::MatterSources;
    use std::sync::Arc;

    const D: usize = 3;
    const N: usize = 4;

    fn periodic() -> (BoundaryConfig<D>, BoundaryValues<D>) {
        (
            BoundaryConfig::<D>::periodic(),
            BoundaryValues::<D>::zeros(),
        )
    }

    fn identity() -> Tensor2<D> {
        let mut t = Tensor2::<D>::zero();
        for i in 0..D {
            t[(i, i)] = 1.0;
        }
        t
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // a stationary boosted slice: alpha = 1, beta^x = 0.3, gamma = delta,
    // K = 0. On a periodic grid the ADM right-hand side vanishes exactly
    // only for uniform data: the K equation carries
    // -nabla_i nabla_j alpha + alpha R_ij, and both the wrapped second
    // derivative of a spatial ramp at the periodic seam and the Ricci
    // tensor of a spatially curved gamma make K evolve, which then makes
    // every later ADM stage differ from the first. Uniform
    // lapse/shift/metric keeps rhs identically zero through all stages,
    // so the stage bundles must reproduce the statically built Christoffel
    // field exactly (the metric is boosted, not Minkowski: g_00 = -0.91,
    // g_0x = 0.3). Nonzero-Gamma stage discrimination is covered by the
    // marker and dust tests below.
    fn static_data(grid: Arc<Grid<D>>) -> (AdmState<D>, ChristoffelField<N, D>) {
        let (config, values) = periodic();

        let lapse: LapseField<D> = Field::new(grid.clone(), 1.0);
        let shift: ShiftField<D> = Field::new(grid.clone(), Vector::new([0.3, 0.0, 0.0]));
        let gamma: SliceMetric<D> = Field::new(grid.clone(), identity());
        let k: ExtrinsicCurvature<D> = Field::new(grid.clone(), Tensor2::zero());

        let adm = AdmState {
            grid: grid.clone(),
            gamma,
            k,
            lapse,
            shift,
        };

        let frozen =
            christoffel_symbols::<D, N>(&adm.lapse, &adm.shift, &adm.gamma, &config, &values);

        (adm, frozen)
    }

    // With every stage bundle pointing at the same static field, the
    // stage stepper and the particles API must be exactly the frozen
    // path: identical symbols, identical single-step result, identical
    // committed particle histories. Also, the ADM stage machinery on a
    // stationary background must hand over the static field unchanged,
    // bundle by bundle.
    #[test]
    fn stage_bundles_reduce_exactly_to_frozen_path() {
        let grid = Arc::new(Grid::new([7, 7, 7], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
        let (adm, frozen) = static_data(grid.clone());
        let (config, values) = periodic();
        let dt = 0.05;

        // field level: all four bundles equal the frozen field
        let stages = adm.step_rk4_stages(dt, &config, &values);
        let connections = stage_connections::<D, N>(&stages, &config, &values);
        for (i, bundle) in connections.connections.iter().enumerate() {
            grid.for_each_index(|idx| {
                assert_eq!(
                    *bundle.get(idx),
                    *frozen.get(idx),
                    "bundle {} diverges from the frozen field at {:?}",
                    i,
                    idx
                );
            });
        }
        assert_eq!(connections.dt, dt);

        // stepper level: the array-of-refs call equals the frozen call
        let state = GeodesicState {
            pos: [1.25, 1.25, 1.25],
            vel: [0.3, 0.1, -0.2],
        };
        let direct = geodesic_step_rk4::<D, N>(&state, dt, &frozen).unwrap();
        let staged =
            geodesic_stages_step_rk4::<D, N>(&state, dt, [&frozen, &frozen, &frozen, &frozen])
                .unwrap();
        assert_eq!(direct, staged);

        // particle level: histories identical over two coupled steps
        let mut frozen_particles: Particles<D> = Particles::new();
        let mut coupled_particles: Particles<D> = Particles::new();
        frozen_particles
            .spawn(
                0.0,
                GeodesicState {
                    pos: [1.25, 1.25, 1.25],
                    vel: [0.3, 0.1, -0.2],
                },
            )
            .unwrap();
        coupled_particles
            .spawn(
                0.0,
                GeodesicState {
                    pos: [1.25, 1.25, 1.25],
                    vel: [0.3, 0.1, -0.2],
                },
            )
            .unwrap();

        let mut coupling_t = 0.0;
        for _ in 0..2 {
            frozen_particles.step_all(dt, &frozen).unwrap();

            let stages = adm.step_rk4_stages(dt, &config, &values);
            let connections = stage_connections::<D, N>(&stages, &config, &values);
            coupled_particles
                .step_all_with_connections(coupling_t, &connections)
                .unwrap();
            coupling_t += dt;
        }

        assert_eq!(
            frozen_particles.iter().count(),
            coupled_particles.iter().count()
        );
        let (a, b) = (
            frozen_particles.get(0).unwrap(),
            coupled_particles.get(0).unwrap(),
        );
        assert_eq!(a.history, b.history);
        assert_eq!(a.time, b.time);
    }

    // marker bundles: gamma = delta, alpha = 1, beta = 0, and a
    // stage-indexed time derivative A_xx = c_i + e x at stage i. The
    // resulting Gamma^x_0x = (c_i + e x) / 2 varies linearly in x, so
    // trilinear interpolation reproduces it exactly and the only
    // difference between stages is the marker level. A hand-rolled RK4
    // that samples bundle i at substage i must agree with the stage
    // stepper, and bundling one marker four times must give a measurably
    // different trajectory.
    #[test]
    fn stage_marker_connections_drive_per_stage_evolution() {
        let grid = Arc::new(Grid::new([9, 9, 9], [0.25, 0.25, 0.25], [0.0, 0.0, 0.0]));
        let (config, values) = periodic();

        let lapse: LapseField<D> = Field::new(grid.clone(), 1.0);
        let shift: ShiftField<D> = Field::new(grid.clone(), Vector::zero());
        let gamma: SliceMetric<D> = Field::new(grid.clone(), identity());
        let dshift_dt: ShiftField<D> = Field::new(grid.clone(), Vector::zero());
        let dlapse_dt: LapseField<D> = Field::new(grid.clone(), 0.0);

        let markers = [0.2, 0.4, 0.6, 0.8];
        let e = 0.1;
        let mut bundles: Vec<ChristoffelField<N, D>> = Vec::with_capacity(4);
        for c in markers {
            let mut dgamma_dt: SliceMetric<D> = Field::new(grid.clone(), Tensor2::zero());
            grid.for_each_index(|idx| {
                let x = idx[0] as f64 * grid.spacing[0];
                let mut a = Tensor2::<D>::zero();
                a[(0, 0)] = c + e * x;
                dgamma_dt.set(idx, a);
            });
            bundles.push(christoffel_symbols_with_time_derivatives::<D, N>(
                &lapse, &shift, &gamma, &dgamma_dt, &dshift_dt, &dlapse_dt, &config, &values,
            ));
        }

        let state = GeodesicState {
            pos: [0.75, 0.5, 0.5],
            vel: [0.3, 0.05, -0.07],
        };

        // hand-rolled RK4 with per-stage sampling, built only from public
        // primitives: geodesic_rhs on stage-interpolated tables
        let dt = 0.1;
        let interp = |i: usize, pos: [f64; D]| interpolate_christoffel(&bundles[i], pos).unwrap();
        let axpy = |s: &GeodesicState<D>, a: f64, d: &GeodesicState<D>| -> GeodesicState<D> {
            let mut out = *s;
            for i in 0..D {
                out.pos[i] += a * d.pos[i];
                out.vel[i] += a * d.vel[i];
            }
            out
        };

        let k1 = geodesic_rhs::<D, N>(&state, &interp(0, state.pos));
        let y2 = axpy(&state, 0.5 * dt, &k1);
        let k2 = geodesic_rhs::<D, N>(&y2, &interp(1, y2.pos));
        let y3 = axpy(&state, 0.5 * dt, &k2);
        let k3 = geodesic_rhs::<D, N>(&y3, &interp(2, y3.pos));
        let y4 = axpy(&state, dt, &k3);
        let k4 = geodesic_rhs::<D, N>(&y4, &interp(3, y4.pos));

        // the engine's RK4 combination: one fused sum of the four
        // weighted stage rates added to the start state in exactly that
        // grouping (float addition is not associative, so a chained axpy
        // accumulation can differ in the last ulp)
        let mut expected = state;
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

        let staged = geodesic_stages_step_rk4::<D, N>(
            &state,
            dt,
            [&bundles[0], &bundles[1], &bundles[2], &bundles[3]],
        )
        .unwrap();

        assert_eq!(staged.pos, expected.pos, "staged pos vs hand RK4");
        assert_eq!(staged.vel, expected.vel, "staged vel vs hand RK4");

        let frozen = geodesic_stages_step_rk4::<D, N>(
            &state,
            dt,
            [&bundles[0], &bundles[0], &bundles[0], &bundles[0]],
        )
        .unwrap();
        assert!(
            (staged.vel[0] - frozen.vel[0]).abs() > 1e-9,
            "distinct stage markers must move the trajectory off the frozen one"
        );
    }

    // the clock contract: refusing an out-of-sync batch without moving
    // anything, accepting the batch at the exact coupling time (ballistic
    // advance by dt), rejecting a zero dt, rejecting NaN coupling times,
    // and accepting within the stated tolerance.
    #[test]
    fn synchronized_clock_check_mismatch_aborts() {
        let grid = Arc::new(Grid::new([8, 4, 4], [0.25, 0.25, 0.25], [0.0, 0.0, 0.0]));
        let (config, values) = periodic();

        let lapse: LapseField<D> = Field::new(grid.clone(), 1.0);
        let shift: ShiftField<D> = Field::new(grid.clone(), Vector::zero());
        let gamma: SliceMetric<D> = Field::new(grid.clone(), identity());
        let frozen = christoffel_symbols::<D, N>(&lapse, &shift, &gamma, &config, &values);

        let connections = StageConnections {
            dt: 0.1,
            connections: [
                frozen.clone(),
                frozen.clone(),
                frozen.clone(),
                frozen.clone(),
            ],
        };

        let initial = GeodesicState {
            pos: [0.75, 0.5, 0.5],
            vel: [0.2, 0.0, 0.0],
        };

        let mut particles: Particles<D> = Particles::new();
        particles.spawn(0.0, initial).unwrap();

        // NaN coupling time is rejected before anything happens
        let err = particles
            .step_all_with_connections(f64::NAN, &connections)
            .unwrap_err();
        assert!(matches!(err, ParticleError::NonFiniteTime));
        assert_eq!(particles.get(0).unwrap().time, 0.0);

        // a zero-length synchronized step is an invalid time step
        let zero = StageConnections {
            dt: 0.0,
            connections: [
                frozen.clone(),
                frozen.clone(),
                frozen.clone(),
                frozen.clone(),
            ],
        };
        let err = particles.step_all_with_connections(0.0, &zero).unwrap_err();
        assert!(matches!(err, ParticleError::InvalidTimeStep));

        // one full dt ahead: refused with the particle id and coupling
        // time reported, and nothing moved
        let err = particles
            .step_all_with_connections(0.1, &connections)
            .unwrap_err();
        let (id, coupling) = match &err {
            ParticleError::TimeMismatch {
                particle_id,
                coupling_time,
            } => (*particle_id, *coupling_time),
            other => panic!("expected TimeMismatch, got {:?}", other),
        };
        assert_eq!(id, 0);
        assert_eq!(coupling, 0.1);

        let p = particles.get(0).unwrap();
        assert_eq!(p.time, 0.0);
        assert_eq!(p.state, initial);
        assert_eq!(p.history.len(), 1);
        assert_eq!(p.status, ParticleStatus::Active);

        // exact coupling time: accepted, ballistic advance by dt
        particles
            .step_all_with_connections(0.0, &connections)
            .unwrap();
        let p = particles.get(0).unwrap();
        assert_eq!(p.time, 0.1);
        assert_eq!(p.state.pos[0], 0.75 + 0.1 * 0.2);
        assert_eq!(p.history.len(), 2);

        // within the stated tolerance: accepted as well; the particle
        // starts within 1e-10 of the coupling time, tolerance at t = 1e-10
        // is ~1e-9
        let mut tolerant: Particles<D> = Particles::new();
        tolerant.spawn(1e-10, initial).unwrap();
        tolerant
            .step_all_with_connections(0.0, &connections)
            .unwrap();
        assert!((tolerant.get(0).unwrap().time - (1e-10 + 0.1)).abs() < 1e-15);
    }

    // stopped particles keep their last valid timestamp and must not be
    // re-synchronized: later synchronized steps advance the survivors
    // while the stopped one stays untouched, even though its clock no
    // longer matches the coupling time.
    #[test]
    fn stopped_particles_are_exempt_from_clock_sync() {
        let grid = Arc::new(Grid::new([8, 4, 4], [0.25, 0.25, 0.25], [0.0, 0.0, 0.0]));
        let (config, values) = periodic();
        let lapse: LapseField<D> = Field::new(grid.clone(), 1.0);
        let shift: ShiftField<D> = Field::new(grid.clone(), Vector::zero());
        let gamma: SliceMetric<D> = Field::new(grid.clone(), identity());
        let frozen = christoffel_symbols::<D, N>(&lapse, &shift, &gamma, &config, &values);

        let connections = StageConnections {
            dt: 0.1,
            connections: [
                frozen.clone(),
                frozen.clone(),
                frozen.clone(),
                frozen.clone(),
            ],
        };

        let mut particles: Particles<D> = Particles::new();
        particles
            .spawn(
                37.0,
                GeodesicState {
                    pos: [1.65, 0.5, 0.5],
                    vel: [5.0, 0.0, 0.0],
                },
            )
            .unwrap();
        let runner_id = particles
            .spawn(
                37.0,
                GeodesicState {
                    pos: [0.9, 0.5, 0.5],
                    vel: [0.1, 0.0, 0.0],
                },
            )
            .unwrap();

        // ballistic zero-symbols field: the runner leaves the domain
        // within the first substage and stops, keeping its last valid
        // state and timestamp (37.0, never advanced, history untouched);
        // the other one advances normally with the frozen fast path
        particles.step_all(0.1, &frozen).unwrap();

        let stopped = particles.get(0).unwrap();
        assert_eq!(stopped.status, ParticleStatus::StoppedAtBoundary);
        let stopped_time = stopped.time;
        assert_eq!(stopped_time, 37.0);
        assert_eq!(stopped.history.len(), 1);

        // the survivor drives the coupling time from here on; after two
        // more steps its clock has moved 0.2 ahead of the stopped
        // particle, which would fail any clock check but for the
        // exemption
        let mut t = particles.get(runner_id).unwrap().time;
        for _ in 0..2 {
            particles
                .step_all_with_connections(t, &connections)
                .unwrap();
            t = particles.get(runner_id).unwrap().time;
        }

        let stopped = particles.get(0).unwrap();
        assert_eq!(stopped.time, stopped_time);
        assert_eq!(stopped.status, ParticleStatus::StoppedAtBoundary);
        assert!(
            (stopped.time - t).abs() > 0.19,
            "clocks must have drifted apart"
        );

        let runner = particles.get(runner_id).unwrap();
        assert_eq!(runner.status, ParticleStatus::Active);
        assert!((runner.time - (37.0 + 0.1 + 0.2)).abs() < 1e-9);
    }

    // prescribed exponential expansion a(t) = a0 e^{H t} with
    // alpha = 1, beta = 0: build stage bundles with the exact scale
    // factor at each stage time (0, dt/2, dt/2, dt) and integrate a
    // timelike geodesic. For this metric form the momentum
    //   p = a^2 v / sqrt(1 - a^2 v^2)
    // is conserved exactly, and v(t) = p / (a sqrt(a^2 + p^2)) is known
    // exactly, so both the trajectory and the conservation are checked
    // against closed expressions and only RK4 truncation remains.
    #[test]
    fn flrw_prescribed_scale_factor_trajectory() {
        let grid = Arc::new(Grid::new([3, 3, 3], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]));
        let (config, values) = periodic();

        let lapse: LapseField<D> = Field::new(grid.clone(), 1.0);
        let shift: ShiftField<D> = Field::new(grid.clone(), Vector::zero());
        let dshift_dt: ShiftField<D> = Field::new(grid.clone(), Vector::zero());
        let dlapse_dt: LapseField<D> = Field::new(grid.clone(), 0.0);
        let h = 0.5;
        let a0 = 1.0;
        let v0 = 0.28;

        let scale = |t: f64| a0 * (h * t).exp();

        let bundle_at = |t: f64| {
            let a = scale(t);
            let a2 = a * a;
            let mut gamma: SliceMetric<D> = Field::new(grid.clone(), Tensor2::zero());
            let mut dgamma_dt: SliceMetric<D> = Field::new(grid.clone(), Tensor2::zero());
            grid.for_each_index(|idx| {
                let mut g = Tensor2::<D>::zero();
                let mut da = Tensor2::<D>::zero();
                for i in 0..D {
                    g[(i, i)] = a2;
                    da[(i, i)] = 2.0 * h * a2;
                }
                gamma.set(idx, g);
                dgamma_dt.set(idx, da);
            });
            christoffel_symbols_with_time_derivatives::<D, N>(
                &lapse, &shift, &gamma, &dgamma_dt, &dshift_dt, &dlapse_dt, &config, &values,
            )
        };

        let p = a0 * a0 * v0 / (1.0 - a0 * a0 * v0 * v0).sqrt();
        let v_exact = |t: f64| {
            let a = scale(t);
            p / (a * (a * a + p * p).sqrt())
        };

        let mut particles: Particles<D> = Particles::new();
        particles
            .spawn(
                0.0,
                GeodesicState {
                    pos: [0.5, 0.5, 0.5],
                    vel: [v0, 0.0, 0.0],
                },
            )
            .unwrap();

        let dt = 0.02;
        let steps = 20;
        let mut t = 0.0;
        for _ in 0..steps {
            let bundles = [
                bundle_at(t),
                bundle_at(t + dt / 2.0),
                bundle_at(t + dt / 2.0),
                bundle_at(t + dt),
            ];
            let connections = StageConnections {
                dt,
                connections: bundles,
            };
            particles
                .step_all_with_connections(t, &connections)
                .unwrap();
            let next_t = particles.get(0).unwrap().time;
            t = next_t;
        }

        assert!((t - dt * steps as f64).abs() < 1e-9);

        let end = particles.get(0).unwrap().state;
        let t_end = dt * steps as f64;
        let expected_v = v_exact(t_end);
        let measured_error = (end.vel[0] - expected_v).abs();

        // momentum conservation read off the same trajectory
        let a_end = scale(t_end);
        let p_end =
            a_end * a_end * end.vel[0] / (1.0 - a_end * a_end * end.vel[0] * end.vel[0]).sqrt();
        let momentum_drift = (p_end - p).abs();

        assert!(
            measured_error < 1e-5,
            "v: measured {e} expected {x} error {m:e}",
            e = end.vel[0],
            x = expected_v,
            m = measured_error
        );
        assert!(
            momentum_drift < 1e-5,
            "p: measured {} expected {} drift {}",
            p_end,
            p,
            momentum_drift
        );
    }

    // the real coupling: uniform expanding dust slice evolved by the ADM
    // integrator with 1+log lapse and matter sources, stage bundles built
    // from the actual ADM stage pairs each step. Uniformity and isotropy
    // survive the ADM system exactly (uniform sources, diagonal uniform
    // K, no shift), so the slice stays of FLRW form gamma = a(t)^2 delta.
    // Assertions:
    //   1. every stage bundle encodes its own stage pair, pointwise:
    //      Gamma^x_0x = A_xx / (2 gamma_xx) and Gamma^0_00 = alpha_dot / alpha
    //   2. the committed particle state equals a hand-rolled RK4 sampling
    //      the same bundles through the public primitives, on the real
    //      ADM stage data
    //   3. the particle clock tracks the ADM steps exactly
    #[test]
    fn coupled_dust_adm_run_matches_hand_and_form() {
        let grid = Arc::new(Grid::new([4, 4, 4], [0.25, 0.25, 0.25], [0.0, 0.0, 0.0]));
        let (config, values) = periodic();

        let mut k0 = Tensor2::<D>::zero();
        for i in 0..D {
            k0[(i, i)] = -0.04;
        }
        let mut adm = AdmState {
            grid: grid.clone(),
            gamma: Field::new(grid.clone(), identity()),
            k: Field::new(grid.clone(), k0),
            lapse: Field::new(grid.clone(), 1.0),
            shift: Field::new(grid.clone(), Vector::zero()),
        };

        let matter = MatterSources {
            rho: Field::new(grid.clone(), 0.05),
            momentum: Field::new(grid.clone(), Vector::zero()),
            stress: Field::new(grid.clone(), Tensor2::zero()),
            stress_trace: Field::new(grid.clone(), 0.0),
        };

        let idx = [1usize, 1, 1];

        // one coupled step: real dust ADM stage data
        let stages = adm.step_rk4_with_matter_stages(0.01, &matter, &config, &values);
        let connections = stage_connections::<D, N>(&stages, &config, &values);

        // 1. every bundle encodes its own (state, rate) pair
        let pairs = [
            (&stages.y0, &stages.k1),
            (&stages.y2, &stages.k2),
            (&stages.y3, &stages.k3),
            (&stages.y4, &stages.k4),
        ];
        for (i, (y, k)) in pairs.iter().enumerate() {
            let c = connections.connections[i].get(idx);
            let a_xx = k.gamma.get(idx)[(0, 0)];
            let gamma_xx = y.gamma.get(idx)[(0, 0)];
            assert!(
                close(c.data[1][0][1], 0.5 * a_xx / gamma_xx, 1e-12),
                "bundle {}: Gamma^x_0x = {} vs A_xx/(2 gamma_xx) = {}",
                i,
                c.data[1][0][1],
                0.5 * a_xx / gamma_xx
            );
            let alpha_dot = *k.lapse.get(idx);
            assert!(
                close(c.data[0][0][0], alpha_dot / *y.lapse.get(idx), 1e-12),
                "bundle {}: Gamma^0_00 = {} vs alpha_dot/alpha = {}",
                i,
                c.data[0][0][0],
                alpha_dot / *y.lapse.get(idx)
            );
        }

        // 2. hand-rolled RK4 over the same bundles vs the committed state
        let mut particles: Particles<D> = Particles::new();
        particles
            .spawn(
                0.0,
                GeodesicState {
                    pos: [0.5, 0.5, 0.5],
                    vel: [0.1, 0.0, 0.01],
                },
            )
            .unwrap();
        particles
            .step_all_with_connections(0.0, &connections)
            .unwrap();

        let start = GeodesicState {
            pos: [0.5, 0.5, 0.5],
            vel: [0.1, 0.0, 0.01],
        };
        let interp = |i: usize, pos: [f64; D]| {
            interpolate_christoffel(&connections.connections[i], pos).unwrap()
        };
        let axpy = |s: &GeodesicState<D>, a: f64, d: &GeodesicState<D>| -> GeodesicState<D> {
            let mut out = *s;
            for i in 0..D {
                out.pos[i] += a * d.pos[i];
                out.vel[i] += a * d.vel[i];
            }
            out
        };
        let dt = connections.dt;
        let k1 = geodesic_rhs::<D, N>(&start, &interp(0, start.pos));
        let y2 = axpy(&start, 0.5 * dt, &k1);
        let k2 = geodesic_rhs::<D, N>(&y2, &interp(1, y2.pos));
        let y3 = axpy(&start, 0.5 * dt, &k2);
        let k3 = geodesic_rhs::<D, N>(&y3, &interp(2, y3.pos));
        let y4 = axpy(&start, dt, &k3);
        let k4 = geodesic_rhs::<D, N>(&y4, &interp(3, y4.pos));
        let expected = axpy(
            &axpy(
                &axpy(&axpy(&start, dt / 6.0, &k1), dt / 3.0, &k2),
                dt / 3.0,
                &k3,
            ),
            dt / 6.0,
            &k4,
        );

        let committed = particles.get(0).unwrap();
        assert_eq!(
            committed.state.pos, expected.pos,
            "committed pos vs hand RK4"
        );
        assert_eq!(
            committed.state.vel, expected.vel,
            "committed vel vs hand RK4"
        );

        // 3. the clock tracks the ADM step exactly
        assert_eq!(committed.time, 0.01);

        // recompute the ADM state advance for completeness: the next
        // coupling step must see exactly the collapsed state
        adm = stages.collapse(&config, &values);
        let stages2 = adm.step_rk4_with_matter_stages(0.01, &matter, &config, &values);
        let connections2 = stage_connections::<D, N>(&stages2, &config, &values);
        particles
            .step_all_with_connections(committed.time, &connections2)
            .unwrap();
        assert!((particles.get(0).unwrap().time - 0.02).abs() < 1e-15);
    }
}
