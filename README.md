# Einstein Engine

A Rust-based numerical relativity engine for exploring and simulating Einstein's equations on a numerical spacetime grid.

The project aims to build a modern, modular, and readable bridge between the mathematics of General Relativity and high-performance numerical simulation.

## Vision

The long-term goal is to simulate dynamical spacetime systems from Einstein's field equations rather than using a predefined gravitational potential.

The core idea is:

$$
G_{\mu\nu} = 8\pi G\,T_{\mu\nu}
$$

with the spacetime represented numerically on a 3D spatial grid whose state evolves in time.

Conceptually:

```text
Matter / Tμν
      │
      ▼
Einstein Equations
      │
      ▼
Spacetime Geometry
      │
      ▼
Christoffel Symbols / Curvature
      │
      ▼
Geodesics / Matter Evolution
      │
      └──────────────► Tμν
                       │
                       ▼
                  Next timestep
```

The engine is intended to support both physical simulation and mathematical experimentation.

## Main Approach

The engine uses a **3+1 decomposition** of spacetime.

Instead of evolving a full 4D spacetime grid directly, the simulation maintains a 3D spatial slice and evolves it through coordinate time.

The main geometric variables are:

$$
\gamma_{ij},\quad K_{ij},\quad \alpha,\quad \beta^i
$$

where:

* \(\gamma_{ij}\) is the spatial metric.
* \(K_{ij}\) is the extrinsic curvature.
* \(\alpha\) is the lapse.
* \(\beta^i\) is the shift vector.

These variables define the spacetime metric through

$$
ds^2 =
-\alpha^2dt^2
+
\gamma_{ij}(dx^i+\beta^i dt)(dx^j+\beta^j dt)
$$

The numerical state is therefore stored on a 3D grid and advanced through time.

## Project Goals

The project is being designed around several goals:

* Implement real numerical solutions of Einstein's equations.
* Keep the mathematics visible and understandable in the source code.
* Use Rust for performance, memory safety, and modular systems design.
* Separate physics, numerical methods, geometry, and visualization.
* Validate every major component against known solutions.
* Support small simulations locally and larger simulations on remote compute environments.

## Planned Numerical Pipeline

```text
3D Grid
   ↓
3+1 Spacetime State
   ↓
γij, Kij, α, βi
   ↓
Spatial Derivatives
   ↓
Curvature
   ↓
Einstein Constraints
   ↓
Einstein Evolution
   ↓
Time Integration
   ↓
Updated Spacetime
```

Matter and geodesic solvers will be integrated into this pipeline as the engine matures.

## Architecture

Files marked **stub** are skeleton placeholders, empty and not yet wired in.

```text
einstein-engine/
│
├── Cargo.toml                # workspace root
├── engine/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── grid/                              # implemented + tested
│       │   ├── mod.rs
│       │   ├── grid.rs                        # const-generic Grid
│       │   ├── field.rs                       # Field<Arc<Grid>, values>
│       │   └── boundary.rs                    # BoundaryKind/Config + apply
│       │
│       ├── geometry/                          # implemented + tested
│       │   ├── mod.rs
│       │   ├── tensor.rs                      # Vector, Tensor2, matrix invert
│       │   ├── metric.rs                      # 3+1 split + Metric + 4-metric
│       │   ├── christoffel.rs                 # Gamma^a_bc from 3+1 fields (+ spatial)
│       │   ├── riemann.rs                     # R^mu_nu rho sigma from Christoffel
│       │   ├── ricci.rs                       # R_mu_nu by contracting Riemann
│       │   ├── curvature.rs                   # Ricci scalar R; spatial R^(3)
│       │   ├── covariant_derivative.rs        # D_k V^i, D_k W_i, div_j T^i_j
│       │
│       ├── numerics/
│       │   ├── mod.rs
│       │   ├── derivatives.rs                 # forward/central + edges
│       │   ├── interpolation.rs               # stub
│       │   ├── rk4.rs                         # EvolutionState + RK4 step
│       │   └── solver.rs                      # stub
│       │
│       ├── relativity/                        # einstein, constraints, evolution
│       │   ├── einstein.rs                    # G_mu_nu from Ricci and metric
│       │   ├── constraints.rs                 # Hamiltonian + momentum constraints
│       │   ├── evolution.rs                   # d_t gamma_ij and d_t K_ij (ADM)
│       │   ├── gauge.rs                        # 1+log lapse, frozen shift
│       │   └── matter.rs                      # stub
│       │
│       ├── initial_data/                      # stubs
│       │   ├── minkowski.rs
│       │   ├── schwarzschild.rs
│       │   └── custom.rs
│       │
│       ├── physics/                           # stress-energy implemented
│       │   ├── body.rs                        # stub
│       │   ├── particles.rs                   # stub
│       │   └── stress_energy.rs               # T_mu_nu from a density field
│       │
│       └── validation/
│           ├── mod.rs
│           ├── constraints.rs                 # max |H| and |M| on a slice
│           ├── conservation.rs                # stub
│           └── schwarzschild.rs               # stub
│
├── tests/
│   ├── grid/                                  # grid, field, boundary
│   ├── numerics/                              # derivatives, rk4
│   ├── geometry/                              # christoffel, riemann, ricci,
│   │                                           #  scalar, spatial R^(3), cov. deriv
│   ├── physics/                               # stress_energy
│   ├── relativity/                            # einstein, constraints, evolution, gauge
│   └── validation/                            # constraint monitor
│
├── examples/                                  # empty skeletons
│   ├── flat_spacetime.rs
│   ├── schwarzschild.rs
│   └── binary_system.rs
│
├── visualization/                             # planned
├── simulations/                               # planned
├── data/                                      # planned
├── docs/                                      # planned
│
└── README.md
```

## Development Roadmap

### Phase 1 — Numerical Foundation

* [x] 3D grid
* [x] Field storage
* [x] Indexing and memory layout
* [x] Boundary handling
* [x] Spatial derivatives (forward + central, edge-aware)
* [ ] Interpolation
* [x] Time integration (RK4)

### Phase 2 — 3+1 Geometry

* [x] Spatial metric \(\gamma_{ij}\) (`SliceMetric`)
* [x] Extrinsic curvature \(K_{ij}\) (`ExtrinsicCurvature`)
* [x] Lapse \(\alpha\) (`LapseField`)
* [x] Shift \(\beta^i\) (`ShiftField`)
* [x] Reconstruction of the 4D metric \(g_{\mu\nu}\) (`SpacetimeMetric`)

### Phase 3 — Curvature

* [x] Christoffel symbols (validated against Minkowski + static lapse)
* [x] Riemann tensor (spacetime and pure spatial slice)
* [x] Ricci tensor
* [x] Ricci scalar
* [x] Spatial slice curvature R^(3)
* [x] Covariant derivative (vector, covector, mixed divergence)
* [x] Einstein tensor

### Phase 4 — Numerical Relativity

* [x] Einstein constraints
* [x] Evolution equations
* [ ] BSSN-based evolution
* [x] Gauge conditions (1+log lapse, frozen shift)
* [x] Constraint monitoring
* Stable numerical evolution

### Phase 5 — Matter

* Stress-energy tensor
* Matter fields
* Matter evolution
* Coupling between matter and spacetime

### Phase 6 — Particle Dynamics

* Metric interpolation
* Christoffel interpolation
* Geodesic integration
* Test particles
* Matter tracers

### Phase 7 — Validation

* Minkowski spacetime
* Known analytic solutions
* Schwarzschild spacetime
* Constraint preservation
* Conservation tests
* Convergence tests

### Phase 8 — Larger Simulations

* Single compact objects
* Binary systems
* Multi-body systems
* Gravitational-wave experiments
* Large 3D simulations

## Visualization

The numerical core is intentionally separated from visualization.

Rust is responsible for:

```text
Physics
Numerics
Geometry
Evolution
Data output
```

Python and Manim are intended to handle:

```text
Analysis
Plots
Trajectories
Curvature visualization
Spacetime visualization
Animations
```

This separation allows the simulation engine to run locally or on remote compute resources without depending on the visualization layer.

## Validation Philosophy

Correctness is more important than visual results.

A simulation is not considered successful merely because it produces an interesting animation.

Each numerical component should be tested against:

1. Analytical results where available.
2. Constraint equations.
3. Conservation laws.
4. Numerical convergence.
5. Known spacetime solutions.

The first geometric target is flat Minkowski spacetime:

$$
\alpha = 1,
\qquad
\beta^i = 0,
\qquad
\gamma_{ij} = \delta_{ij},
\qquad
K_{ij}=0
$$

which must produce zero spacetime curvature. This identity case is exercised by the flat-space tests in every module that touches curvature or the constraints.

## Performance

The project is designed for scalable execution.

Development and debugging can be performed on a local machine, while larger simulations can be executed on more powerful remote systems.

Possible future optimization targets include:

* Multithreading
* SIMD
* Cache-aware field layouts
* Parallel grid operations
* GPU acceleration
* Distributed simulations

Performance optimizations will be introduced only after the numerical implementation is validated.

## Status

**Early development**

The numerical foundation, the 3+1 geometry, and the full curvature → Einstein → constraint chain are in place:

* `Grid`, `Field`, and boundary conditions are implemented and tested.
* Spatial derivative operators work on arbitrary grid dimensions.
* The 3+1 variables (lapse, shift, extrinsic curvature, slice metric) live together in `geometry/metric.rs`, with the `Metric` pointwise struct and the 4-metric rebuild.
* The full curvature chain (Christoffel → Riemann → Ricci → scalar) works for the spacetime metric, and the same machinery computes the spatial slice curvature R^(3).
* Covariant derivatives (vector, covector, mixed divergence) live in `geometry/covariant_derivative.rs` — the momentum constraint already reuses the mixed divergence.
* `physics/stress_energy.rs` provides the matter side (dust), and `relativity/einstein.rs` the left side, so `G_mu_nu = 8 pi T_mu_nu` can be assembled.
* Both Einstein constraints are implemented and tested: the Hamiltonian constraint (`H = R^(3) + K^2 - K_ij K^ij - 16 pi rho`) and the momentum constraint (`M^i = div_j(K^i_j - delta^i_j K) - 8 pi S^i`).
* The ADM evolution right-hand sides are implemented: `d_t gamma_ij = -2 alpha K_ij + L_beta gamma_ij` and `d_t K_ij = -alpha nabla_i nabla_j alpha + alpha (R_ij - 2 K_ik K^k_j + K K_ij) + L_beta K_ij`, both in partial-derivative form where possible and with the upper triangle mirrored so symmetry is structural.
* `numerics::derivatives::shift_derivative` supplies the `beta^k d_k` advection term that the Lie derivatives need.
* `engine/examples/evolution.rs` checks that flat spacetime stays fixed, then transports a plane wave with RK4 and reports the speed it actually travels at.
* `numerics::rk4` adds `EvolutionState` (the four ADM variables) and a Runge-Kutta 4 step; the state also carries `lapse_rhs` and `shift_rhs`, which `rhs` composes into the gauge part of `d(state)/dt`.
* `step_rk4` re-imposes the boundary conditions on the evolved state (and on the state it starts from), otherwise RK4 would leave the faces wherever the stencils pushed them. Enforcement is scoped to the lapse: `BoundaryValues` holds one scalar per face, which is a complete description of a scalar condition and nothing more. Holding a vector or a rank-2 field to a single value across all of its components is not a statement about the physics but a way to silently overwrite the state, so the shift, metric and extrinsic curvature are left alone until a per-component boundary type exists. Periodic faces are skipped everywhere, since the wrapping stencils already carry that coupling and `boundary::apply`'s endpoint averaging would clip the field. **Known limitation:** the derivative stencils still read the same single scalar per face, so a Dirichlet run does feed that value into the metric's face derivatives; periodic grids, where we actually run, are unaffected.
* `relativity::gauge` gives those variables a time derivative: 1+log slicing for the lapse, `d_t alpha = beta^i d_i alpha - 2 alpha K` with `K = gamma^ij K_ij`, and a shift that stays frozen because a Gamma-driver would need `d_t Gamma^i`, i.e. third derivatives of the metric.
* `validation::constraints::residuals` reduces both Einstein constraints on a slice to two numbers, so drift can be measured instead of assumed.
* 92 unit tests pass (`cargo test` from inside `engine/`).

Two of the RK4 tests are the ones worth trusting. A plane wave built to
travel at light speed is measured travelling at 0.9977, against the 0.9984
ceiling that second-order centred differences impose at that `k * h`, so the
shortfall is discretisation rather than a bug in the equations. And the
temporal error ratio measures 9038 where fourth order predicts 10000,
confirming the integrator rather than just confirming that it runs.

Getting the wave to travel *forwards* needed the right sign for `K_ij`. The
ADM metric equation gives `K_ij = -1/2 d_t h_ij` at first order, so the
opposite sign quietly builds a left-moving wave. The amplitude matters just
as much: `A * k` has to stay well below 1, or the `K_ij K^ij` term in the `K`
equation stops being a correction and the slice no longer solves the
constraints.

Constraint monitoring is now in place, and it found something. The plane wave
used to validate RK4 is **not an exact vacuum solution**: `|H|max = 1.235e-3`
and `|M|max = 6.155e-4`. Hand-estimated from `H = 3R + K^2 - K_ij K^ij` with
`3R = 0` and traceless `K`, the Hamiltonian residual should be `3.16e-4`, so
the nonlinear `3R` terms contribute roughly three times that much and do not
cancel the `K_ij K^ij` term the way the linearised argument assumes. The
wave-speed test still passes because the violation is `O((A k)^2)` and the
linear dynamics dominate over eight steps, but the data is off-shell and
getting it right means solving the constraints for the initial slice rather
than imposing TT data directly.

Constraint drift is real but slow, and it is now measured rather than
asserted: over 8 RK4 steps `|H|max` oscillates between `1.218e-3` and
`1.243e-3` and grows by 0.6%, with the envelope rising monotonically. That is
too small to assert on this horizon, so the test bounds growth at 2x and
catches instability instead. A longer run is what would make the drift claim
itself testable. The gauge running underneath is 1+log, but the wave is
transversely traceless so its `K` trace only survives through
`gamma^ij != delta^ij`: the lapse moves by `2.0e-7` over the run and the
slice is indistinguishable from geodesic at this horizon. Tracing `K` with
`delta^ij` instead would give exactly zero, so a test pins that the
inverse metric is really used.

Traceless data is a weak test of a slicing condition, so there is a second
one where it actually bites: a flat slice with `trace K = 1 + x^2`, so
`d_t alpha = -2 alpha (1 + x^2)` is large and curved. After one step the
lapse has a spread of `0.33`, and feeding that gauge-made lapse back into
`extrinsic_curvature_rhs` changes `d_t K` by `1.28` -- the `-alpha nabla_i
nabla_j alpha` term the gauge created. That closes the loop from the gauge
to the geometry, which no other test does.

Next up is constraint-solving initial data, then BSSN.
Plain ADM is weakly hyperbolic, so it is useful for wave tests but not for
stable long runs.

Remote compute support, visualization, and large simulations remain future work.

## License

License: TBD.
