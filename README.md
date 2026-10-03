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
│       │   ├── rk4.rs                         # stub
│       │   └── solver.rs                      # stub
│       │
│       ├── relativity/                        # einstein + constraints implemented
│       │   ├── einstein.rs                    # G_mu_nu from Ricci and metric
│       │   ├── constraints.rs                 # Hamiltonian + momentum constraints
│       │   ├── evolution.rs                   # stub
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
│       └── validation/                        # stubs
│           ├── constraints.rs
│           ├── conservation.rs
│           └── schwarzschild.rs
│
├── tests/
│   ├── grid/                                  # grid, field, boundary
│   ├── numerics/                              # derivatives
│   ├── geometry/                              # christoffel, riemann, ricci,
│   │                                           #  scalar, spatial R^(3), cov. deriv
│   ├── physics/                               # stress_energy
│   └── relativity/                            # einstein, constraints
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
* [x] Riemann tensor (spacetime and pure spatial slice)
* [x] Ricci tensor
* [x] Ricci scalar
* [x] Spatial slice curvature R^(3)
* [x] Covariant derivative (vector, covector, mixed divergence)
* [x] Einstein tensor

### Phase 4 — Numerical Relativity

* [x] Einstein constraints
* [ ] Evolution equations
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
* 61 unit tests pass (`cargo test` from inside `engine/`).

Next up are the Einstein evolution equations and time integration (ADM or BSSN), followed by gauge conditions.

Remote compute support, visualization, and large simulations remain future work.

## License

License: TBD.
