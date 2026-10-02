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
│       ├── geometry/                          # partly implemented
│       │   ├── mod.rs
│       │   ├── tensor.rs                      # Vector, Tensor2
│       │   ├── christoffel.rs                 # Gamma^a_bc from 3+1 fields
│       │   ├── curvature.rs                   # stub
│       │   ├── ricci.rs                       # stub
│       │   └── riemann.rs                     # stub
│       │
│       ├── spacetime/                         # split 3+1 variables
│       │   ├── mod.rs
│       │   ├── lapse.rs                       # LapseField = Field<f64>
│       │   ├── shift.rs                       # ShiftField = Field<Vector>
│       │   ├── extrinsic.rs                   # ExtrinsicCurvature
│       │   └── slice.rs                       # SliceMetric + 4-metric rebuild
│       │
│       ├── numerics/
│       │   ├── mod.rs
│       │   ├── derivatives.rs                 # forward/central + edges
│       │   ├── interpolation.rs               # stub
│       │   ├── rk4.rs                         # stub
│       │   └── solver.rs                      # stub
│       │
│       ├── relativity/                        # stubs
│       │   ├── einstein.rs
│       │   ├── constraints.rs
│       │   ├── evolution.rs
│       │   └── matter.rs
│       │
│       ├── initial_data/                      # stubs
│       │   ├── minkowski.rs
│       │   ├── schwarzschild.rs
│       │   └── custom.rs
│       │
│       ├── physics/                           # stubs
│       │   ├── body.rs
│       │   ├── particles.rs
│       │   └── stress_energy.rs
│       │
│       └── validation/                        # stubs
│           ├── constraints.rs
│           ├── conservation.rs
│           └── schwarzschild.rs
│
├── tests/
│   └── unit/                                  # one target per suite
│       ├── grid.rs                            # 10 tests
│       ├── field.rs                           # 7 tests
│       ├── boundary.rs                        # 8 tests
│       ├── derivatives.rs                     # 5 tests
│       ├── christoffel.rs                     # 2 tests
│       └── slice.rs                           # 4 tests
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
* [ ] Time integration

### Phase 2 — 3+1 Geometry

* [x] Spatial metric \(\gamma_{ij}\) (`SliceMetric`)
* [x] Extrinsic curvature \(K_{ij}\) (`ExtrinsicCurvature`)
* [x] Lapse \(\alpha\) (`LapseField`)
* [x] Shift \(\beta^i\) (`ShiftField`)
* [x] Reconstruction of the 4D metric \(g_{\mu\nu}\) (`SpacetimeMetric`)

### Phase 3 — Curvature

* [x] Christoffel symbols (validated against Minkowski + static lapse)
* [ ] Riemann tensor
* [ ] Ricci tensor
* [ ] Ricci scalar
* [ ] Einstein tensor

### Phase 4 — Numerical Relativity

* Einstein constraints
* Evolution equations
* BSSN-based evolution
* Gauge conditions
* Constraint monitoring
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

which must produce zero spacetime curvature.

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

The numerical foundation and the 3+1 geometric representation are taking shape:

* `Grid`, `Field`, and boundary conditions are implemented and tested.
* Spatial derivative operators work on arbitrary grid dimensions.
* The 3+1 variables are split into dedicated modules (`lapse`, `shift`, `extrinsic`, `slice`).
* Christoffel symbols are computed from the metric fields and match analytic solutions.
* 36 unit tests pass (`cargo test` from inside `engine/`).

Next up is curvature (`riemann.rs`, `ricci.rs`) and then the Einstein constraints/evolution.

Remote compute support, visualization, and large simulations remain future work.

## License

License: TBD.
