#[cfg(test)]
mod tests {
    use einstein_engine::geometry::metric::{
        ExtrinsicCurvature, LapseField, ShiftField, SliceMetric,
    };
    use einstein_engine::geometry::tensor::{Tensor2, Vector};
    use einstein_engine::grid::boundary::{BoundaryConfig, BoundaryValues};
    use einstein_engine::grid::{Field, Grid};
    use einstein_engine::numerics::rk4::AdmState;
    use einstein_engine::relativity::matter::MatterSources;
    use std::sync::Arc;

    const D: usize = 3;

    fn small_grid() -> Arc<Grid<D>> {
        Arc::new(Grid::new([8, 4, 4], [0.25, 0.25, 0.25], [0.0, 0.0, 0.0]))
    }

    fn periodic() -> (BoundaryConfig<D>, BoundaryValues<D>) {
        (
            BoundaryConfig::<D>::periodic(),
            BoundaryValues::<D>::zeros(),
        )
    }

    // a slice that is not kinematically stationary: constant K with zero
    // trace (K_yy = 0.1, K_zz = -0.1) and a small position-dependent
    // lapse, so every stage of an RK4 step carries data of its own.
    fn perturbed_state(grid: Arc<Grid<D>>) -> AdmState<D> {
        let mut gamma: SliceMetric<D> = Field::new(grid.clone(), Tensor2::<D>::zero());
        let mut k: ExtrinsicCurvature<D> = Field::new(grid.clone(), Tensor2::<D>::zero());
        let mut lapse: LapseField<D> = Field::zeros(grid.clone());
        let shift: ShiftField<D> = Field::new(grid.clone(), Vector::<D>::zero());

        grid.for_each_index(|idx| {
            let mut gij = Tensor2::<D>::zero();
            let mut kij = Tensor2::<D>::zero();
            for i in 0..D {
                gij[(i, i)] = 1.0;
            }
            kij[(1, 1)] = 0.1;
            kij[(2, 2)] = -0.1;
            gamma.set(idx, gij);
            k.set(idx, kij);
        });

        for i in 0..8 {
            let x = i as f64 * 0.25;
            for j in 0..4 {
                for l in 0..4 {
                    lapse.set([i, j, l], 1.0 + 0.1 * x);
                }
            }
        }

        AdmState {
            grid: grid.clone(),
            gamma,
            k,
            lapse,
            shift,
        }
    }

    fn states_equal(a: &AdmState<D>, b: &AdmState<D>) -> bool {
        if a.grid.shape != b.grid.shape {
            return false;
        }
        let mut ok = true;
        a.grid.for_each_index(|idx| {
            ok = ok && a.gamma.get(idx) == b.gamma.get(idx);
            ok = ok && a.k.get(idx) == b.k.get(idx);
            ok = ok && a.lapse.get(idx) == b.lapse.get(idx);
            ok = ok && a.shift.get(idx) == b.shift.get(idx);
        });
        ok
    }

    // collapsing the exposed stages must reproduce the direct stepper
    // exactly: same operations, same order, same boundary enforcement.
    #[test]
    fn collapse_matches_direct_stepper() {
        let grid = small_grid();
        let state = perturbed_state(grid.clone());
        let (config, values) = periodic();
        let dt = 0.01;

        let direct = state.step_rk4(dt, &config, &values);
        let collapsed = state
            .step_rk4_stages(dt, &config, &values)
            .collapse(&config, &values);

        assert!(
            states_equal(&direct, &collapsed),
            "collapse of stage data diverged from the direct RK4 step"
        );
    }

    // the stored stage data must pair ki with the state it was evaluated
    // on and hold the states themselves, both reconstructible from the
    // public building blocks (rhs, axpy). With an all-periodic boundary
    // the enforcement step is a no-op, so the stage states equal their
    // public analogues exactly.
    #[test]
    fn stages_match_public_building_blocks() {
        let grid = small_grid();
        let state = perturbed_state(grid.clone());
        let (config, values) = periodic();
        let dt = 0.01;

        let vacuum = MatterSources::vacuum(&grid);
        let stages = state.step_rk4_with_matter_stages(dt, &vacuum, &config, &values);

        assert!(
            states_equal(&stages.y0, &state),
            "y0 must be the enforced incoming state"
        );
        assert!(
            states_equal(
                &stages.k1,
                &state.rhs_with_matter(&vacuum, &config, &values)
            ),
            "k1 must be the rhs of y0"
        );
        assert!(
            states_equal(&stages.y2, &state.axpy(0.5 * dt, &stages.k1)),
            "y2 must be y0 + dt/2 k1"
        );
        assert!(
            states_equal(
                &stages.k2,
                &stages.y2.rhs_with_matter(&vacuum, &config, &values)
            ),
            "k2 must be the rhs of y2"
        );
        assert!(
            states_equal(&stages.y3, &state.axpy(0.5 * dt, &stages.k2)),
            "y3 must be y0 + dt/2 k2"
        );
        assert!(
            states_equal(
                &stages.k3,
                &stages.y3.rhs_with_matter(&vacuum, &config, &values)
            ),
            "k3 must be the rhs of y3"
        );
        assert!(
            states_equal(&stages.y4, &state.axpy(dt, &stages.k3)),
            "y4 must be y0 + dt k3"
        );
        assert!(
            states_equal(
                &stages.k4,
                &stages.y4.rhs_with_matter(&vacuum, &config, &values)
            ),
            "k4 must be the rhs of y4"
        );
        assert_eq!(stages.dt, dt);
    }

    // the vacuum-friendly entry point must be exactly its explicit
    // vacuum-sources twin.
    #[test]
    fn stage_vacuum_wrapper_matches_explicit_vacuum() {
        let grid = small_grid();
        let state = perturbed_state(grid.clone());
        let (config, values) = periodic();
        let dt = 0.01;

        let vacuum = MatterSources::vacuum(&grid);
        let explicit = state.step_rk4_with_matter_stages(dt, &vacuum, &config, &values);
        let vacuum_wrapped = state.step_rk4_stages(dt, &config, &values);

        assert!(
            states_equal(&explicit.y0, &vacuum_wrapped.y0)
                && states_equal(&explicit.y2, &vacuum_wrapped.y2)
                && states_equal(&explicit.y3, &vacuum_wrapped.y3)
                && states_equal(&explicit.y4, &vacuum_wrapped.y4)
                && states_equal(&explicit.k1, &vacuum_wrapped.k1)
                && states_equal(&explicit.k2, &vacuum_wrapped.k2)
                && states_equal(&explicit.k3, &vacuum_wrapped.k3)
                && states_equal(&explicit.k4, &vacuum_wrapped.k4),
            "vacuum stage wrapper diverged from explicit vacuum sources"
        );
    }

    // the stages must carry real evolution data: on this slice the
    // metric equation gives gamma_yy' = -2 alpha K_yy = -0.2 alpha with
    // beta = 0 and uniform gamma (no Lie-derivative terms), so k1's
    // gamma is not zero and y2 has already moved off y0.
    #[test]
    fn stages_hold_evolution_data() {
        let grid = small_grid();
        let state = perturbed_state(grid.clone());
        let (config, values) = periodic();
        let dt = 0.01;

        let stages = state.step_rk4_stages(dt, &config, &values);

        grid.for_each_index(|idx| {
            let expected = -2.0 * *stages.y0.lapse.get(idx) * 0.1;
            assert_eq!(
                stages.k1.gamma.get(idx)[(1, 1)],
                expected,
                "k1 gamma_yy at {:?}",
                idx
            );
        });

        assert!(
            !states_equal(&stages.y0, &stages.y2),
            "y2 must have moved off y0 when the rhs is nonzero"
        );
    }
}
