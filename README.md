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
│   ├── examples/                        # demo + plane-wave evolution demo
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
│       │   ├── interpolation.rs               # trilinear interp + Christoffel sampling
│       │   ├── rk4.rs                         # AdmState + RK4 step
│       │   └── solver.rs                      # stub
│       │
│       ├── relativity/                        # einstein, constraints, evolution, gauge, bssn
 │       │   ├── bssn.rs                        # BSSN state, conversions, RHS + RK4
│       │   ├── einstein.rs                    # G_mu_nu from Ricci and metric
│       │   ├── constraints.rs                 # Hamiltonian + momentum constraints
│       │   ├── evolution.rs                   # d_t gamma_ij and d_t K_ij (ADM)
│       │   ├── gauge.rs                        # 1+log lapse, frozen shift
│       │   └── matter.rs                      # stub
│       │
│       ├── initial_data/                      # schwarzschild implemented
│       │   ├── mod.rs
│       │   ├── minkowski.rs                   # stub
│       │   ├── schwarzschild.rs               # isotropic, guard-railed
│       │   └── custom.rs                      # stub
│       │
│       ├── physics/                           # dust, geodesics, particles implemented
│       │   ├── body.rs                        # stub
│       │   ├── dust.rs                        # DustState, primitives, sources
│       │   ├── dust_evolution.rs              # dust rhs + RK4 stepping
│       │   ├── geodesics.rs                   # geodesic ODE + RK4 stepping
│       │   ├── particles.rs                   # particle collection + history
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
│   ├── numerics/                              # derivatives, rk4, interpolation
│   ├── geometry/                              # christoffel, riemann, ricci,
│   │                                           #  scalar, spatial R^(3), cov. deriv
│   ├── physics/                               # stress_energy, dust, geodesics
│   ├── initial_data/                          # schwarzschild
│   ├── relativity/                            # einstein, constraints, evolution, gauge, bssn
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
* [x] Interpolation
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
* [x] BSSN-based evolution (RHS + RK4; long-run bound still to un-ignore)
* [x] Gauge conditions (1+log lapse, frozen shift)
* [x] Constraint monitoring
* Stable numerical evolution

### Phase 5 — Matter

* [x] Stress-energy tensor
* [x] Matter fields
* [x] Matter evolution
* [x] Coupling between matter and spacetime (stage-consistent ADM + BSSN)

### Phase 6 — Particle Dynamics

* [x] Metric interpolation
* [x] Christoffel interpolation
* [x] Geodesic integration
* [x] Test particles
* [x] Stage-consistent coupling to the ADM integrator (time-dependent Christoffels, RK4 stage bundles, synchronized particle clocks)
* [ ] Matter tracers

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
* The ADM evolution right-hand sides are implemented: `d_t gamma_ij = -2 alpha K_ij + L_beta gamma_ij` and `d_t K_ij = -nabla_i nabla_j alpha + alpha (R_ij - 2 K_ik K^k_j + K K_ij) + L_beta K_ij`, both in partial-derivative form where possible and with the upper triangle mirrored so symmetry is structural.
* `numerics::derivatives::shift_derivative` supplies the `beta^k d_k` advection term that the Lie derivatives need.
* `engine/examples/evolution.rs` checks that flat spacetime stays fixed, then transports a plane wave with RK4 and reports the speed it actually travels at.
* `numerics::rk4` adds `AdmState` (the four ADM variables) and a Runge-Kutta 4 step; the state also carries `lapse_rhs` and `shift_rhs`, which `rhs` composes into the gauge part of `d(state)/dt`.
* `step_rk4` re-imposes the boundary conditions on the evolved state (and on the state it starts from), otherwise RK4 would leave the faces wherever the stencils pushed them. Enforcement is scoped to the lapse: `BoundaryValues` holds one scalar per face, which is a complete description of a scalar condition and nothing more. Holding a vector or a rank-2 field to a single value across all of its components is not a statement about the physics but a way to silently overwrite the state, so the shift, metric and extrinsic curvature are left alone until a per-component boundary type exists. Periodic faces are skipped everywhere, since the wrapping stencils already carry that coupling and `boundary::apply`'s endpoint averaging would clip the field. **Known limitation:** the derivative stencils still read the same single scalar per face, so a Dirichlet run does feed that value into the metric's face derivatives; periodic grids, where we actually run, are unaffected.
* `relativity::gauge` gives those variables a time derivative: 1+log slicing for the lapse, `d_t alpha = beta^i d_i alpha - 2 alpha K` with `K = gamma^ij K_ij`, and a shift that stays frozen because a Gamma-driver would need `d_t Gamma^i`, i.e. third derivatives of the metric.
* `validation::constraints::residuals` reduces both Einstein constraints on a slice to two numbers, so drift can be measured instead of assumed.
* `engine/src/initial_data/schwarzschild.rs` builds the isotropic Schwarzschild slice (numbers below) and refuses input that reaches the puncture; `hamiltonian_residual_converges_second_order` and `slice_is_a_discrete_fixed_point` pin the on-shell and fixed-point properties.
* `relativity::bssn` introduces the BSSN conformal variables (`phi`, `gtilde_ij`, `K`, `A tilde_ij`, `Gamma^i`) and exact conversions to and from the ADM `AdmState`: `phi = ln(det gamma)/12`, `gtilde_ij = det^{-1/3} gamma_ij`, `K = gamma^{ij} K_ij`, `A tilde_ij = e^{-4 phi} (K_ij - (1/3) gamma_ij K)`, `Gamma^i = gtilde^{jk} Gamma^i_jk(gtilde)`. The round trip is the identity, and the algebraic constraints `det(gtilde) = 1` and `gtilde^ij A tilde_ij = 0` hold to machine precision. `BssnState::rhs` implements the vacuum standard BSSN equations (Rosswog & Diener 2404.15952, eqs 66-70) with 1+log lapse and frozen shift; `Rtilde_ij` is the direct conformal Ricci plus the `gtilde_k(i d_j) (Gamma^k - Gamma(n)^k)` hyperbolic correction, `Rphi_ij` carries the conformal factor, and the `d_t K` equation matches ADM up to `-alpha * H`. Schwarzschild BSSN data evolved 17 RK4 steps stays bounded (`max|K| ~ 0.06`, `max|A| ~ 0.05`) where direct-`Rtilde` blows up to `1e31`. Algebraic constraints (`det gtilde = 1`, `tr A tilde = 0`) are re-imposed at every RK4 substep. `bssn_schwarzschild_stays_bounded_over_a_long_run` evolves the M=1 static slice 40 steps and stays finite (`max|K|, max|A| < 1`, `|H|` ratio < 1000) where ADM goes non-finite; truncation-level is not achievable with periodic faces wrapping non-periodic data, so it asserts boundedness, not truncation.
* Matter now gravitates: `relativity::matter` provides the 3+1 projections (`rho`, `S^i`, `S_ij`, `S`) plus `MatterSources::vacuum`, and both evolution RHS take `_with_matter` variants (vacuum wrappers, so no existing caller changed) -- ADM `+4 pi alpha ((S - rho) gamma_ij - 2 S_ij)`, BSSN `+4 pi alpha (rho + s)`, `-e^{-4 phi} alpha 8 pi (S_ij - gamma_ij s/3)`, `-16 pi alpha gtilde^ij s_j`. Zero matter reproduces vacuum bit-for-bit; increments match analytics pointwise; the BSSN/ADM `d_t K` identity holds with matter.
* Dust is dynamical: `physics::dust_evolution` evolves, on the `DustState` in `physics::dust`, the conserved density `Dtilde = sqrt(det gamma) rho W` by `d_t Dtilde + d_i [Dtilde (alpha v^i - beta^i)] = 0` and the conserved momentum `Ptilde_j` by the geodesic source `(alpha sqrt(gamma)/2) T^{mu nu} d_j g_{munu}`, with exact primitive recovery (`v^2 = R/(1+R)`, never superluminal, vacuum floor at `Dtilde <= 0`) — momentum evolution was the missing piece and is now implemented, with caustics documented as out of scope. `BssnState::step_rk4_with_dust` rebuilds sources at every RK4 stage. `engine/examples/coupled_dust.rs` evolves FLRW-patch dust 16 steps (exact on-shell start, conserved mass bit-identical, `|H|` at 1e-11 vs 1.5e-2 fixed-source) plus a flat transport leg at v = 0.2 (profile 1.3e-5, velocity preserved to 2e-14); timestep refinement shows fourth order.
* Geodesics integrate the coordinate-time equation `dv^i/dt = -Gamma^i_{mu nu} w^mu w^nu + v^i Gamma^0_{mu nu} w^mu w^nu` with per-stage interpolated Christoffel symbols (`numerics::interpolation::interpolate_christoffel`, valid on static backgrounds where the existing `christoffel_symbols` zero time derivatives hold exactly); Minkowski geodesics are exactly straight, the connection term is verified against hand-set symbols, RK4 converges at fourth order, and an `r = 10M` Schwarzschild circular orbit holds (a_r matches `-v^2/r`, radius drift ~9e-3, `|E|` drift ~2e-4 over an eighth orbit, `|g(u,u)+1| ~ 4e-16`).
* Particles are bookkeeping around the geodesic integrator: `physics::particles` owns id allocation, a per-step `history`, and a transactional `step_all` -- candidates for every active particle are computed before any particle is mutated, a particle whose step leaves the interpolation domain is marked `StoppedAtBoundary` (keeping its last valid state and timestamp) while the rest keep stepping, and anything non-finite aborts the call with the collection untouched. On the isotropic Schwarzschild slice, two particles half an orbit apart on the r = 10M circular orbit stay circular (radius drift 9e-3 over an eighth orbit, `dE = -u_t` at 2e-4, mirror symmetry exact to 1e-12), and a particle released from rest falls monotonically inward while its energy drifts only 2e-5.
* The Christoffel constructor got a time-aware sibling: `christoffel_symbols_with_time_derivatives` assembles `d_t g_ab` pointwise from the time derivatives of the 3+1 variables (`d_t gamma_ij`, `d_t beta^i`, `d_t alpha`, with the lapse/shift cross terms in the `g_00` and `g_0i` rows) instead of hardcoding the index-0 derivative rows to zero. The old static constructor delegates with zero derivative fields and reproduces the new path bit-exactly -- and the time-derivative helper is verified against a central difference of the full reconstruction, analytically against linear FLRW, against the real ADM right-hand sides, and with a NaN-poisoning test (a non-finite metric-rate component poisons exactly the rows a boost couples it into, because `NaN * 0 = NaN`).
* One ADM RK4 step can now hand over its stage data: `step_rk4_with_matter_stages` returns `AdmRk4Stages` (the four stage states y0, y2, y3, y4 plus the four rates k1..k4 that built them), and `collapse` reproduces the return value of the classic `step_rk4_with_matter` bit-for-bit, so advection-style couplings can see the geometry the integrator stepped through at stage times 0, dt/2, dt/2, dt instead of reconstructing it after the fact.
* Geodesics coupled to evolution are synchronized stage-wise: `physics::geodesics::stage_connections` builds four time-aware connection bundles from one ADM RK4 step (pairing (y0, k1), (y2, k2), (y3, k3), (y4, k4)), `geodesic_stages_step_rk4` samples bundle i at substage i (freezing all four is exactly the old static path), and `Particles::step_all_with_connections` couples the clocks: every active particle must sit at the coupling time within `1e-9 * (1 + |t|)` before anything moves (`TimeMismatch` fires transactionally otherwise; NaN coupling time and non-positive dt are separate rejections), and a stopped particle carries its last timestamp and is exempt. On a prescribed exponential FLRW `a(t)` the momentum `p = a^2 v / sqrt(1 - a^2 v^2)` is conserved across a 20-step coupled run (both checkpoint and drift bounded at < 1e-5, i.e. RK4 truncation size); a real uniform-dust ADM run with 1+log lapse reproduces a hand-rolled stage-RK4 through the public primitives bit-for-bit, every bundle encodes its own stage pair through `Gamma^x_0x = A_xx / (2 gamma_xx)` and `Gamma^0_00 = alpha_dot / alpha` (one nuance this verified: under 1+log the naive `p` is no longer conserved, since `Gamma^0_00 = alpha_dot/alpha != 0` breaks the static-gauge derivation), and a stationary regime reduces identically to the frozen path on bundles, stepper and committed histories. (Known constraint found while testing: on a periodic grid the ADM right-hand side vanishes exactly only for spatially uniform data -- the K equation carries `-nabla_i nabla_j alpha + alpha R_ij`, whose wrapped seam derivatives and slice Ricci terms light up otherwise.)
* The same stage-honest coupling now exists for BSSN, formulation-agnostically: `BssnState::step_rk4_with_matter_stages` exposes the real BSSN RK4 stages (same enforce-and-relax discipline as the classic stepper, whose `collapse` reproduces it bit-for-bit, and which the classic `step_rk4_with_matter` now delegates to verbatim), `relativity::bssn::stage_physical_adm` derives each stage's physical geometry with the product rule -- `gamma_ij = e^{4 phi} gtilde_ij`, so `d_t gamma_ij = e^{4 phi}(4 d_t phi gtilde_ij + d_t gtilde_ij)` with the rates coming from the BSSN right-hand side of the very same stage pair, never ADM substitutes -- and `physics::geodesics::bssn_stage_connections` feeds those into the very same `StageConnections` bundle type, so `geodesic_stages_step_rk4` and `Particles::step_all_with_connections` need zero changes (the only duplicated logic is this one adapter; there is no separate BSSN particle physics). The tests verify the ADM-to-BSSN split round-trips gamma and K at 1e-12 with `det(gtilde) = 1` and `trace(gtilde^-1 A tilde) = 0` exactly, the derived metric rate against a finite difference along the stage direction (~1e-8), bundle 0 against the static Christoffel constructor in the K = A tilde = 0 limit while later stages measurably leave it (slice Ricci drives A tilde), per-stage pair fidelity through `Gamma^x_0x = 2 d_t phi` and `Gamma^0_00 = alpha_dot/alpha` on a slice whose stage rates differ at O(dt) (cross-wired pairing would fail), and a two-step coupled geodesic run that matches a hand-rolled RK4 bit-for-bit (a measurable note the tests document: in 1+log with frozen shift, `Gamma^0_00 = -2K` dominates `-Gamma^x_0x v`, so coordinate velocities grow along the twist-free gauge instead of redshifting).
* Schwarzschild circular orbits are now validated at the photon sphere and the ISCO on a genuine fine grid: `initial_data::isotropic_fields` is the same isotropic slice without the factory's every-node-outside-the-horizon rule (the photon sphere at areal r = 3M sits at isotropic rho = 1.87M, and resolving it needs a centered h = 0.25 grid whose innermost nodes sit at rho = 0.22M < M/2; the data there is the smooth continuation of the same closed-form functions, node-free at the puncture itself, and bit-identical to `initial_data::schwarzschild` wherever the factory still runs). `physics::orbits` derives the circular initial conditions algebraically and chart-independently: the radial balance `Gamma^rho_00 = Gamma^rho_phiphi omega^2` with the isotropic symbols gives `omega^2 = M/r^3` exactly in coordinate time (Kepler's third law in both charts, since `t`, `phi` and the areal radius `r = psi^2 rho` are shared), so rho(3M) = M(2+sqrt(3))/2 and rho(6M) = M(5+2sqrt(6))/2 with analytic endpoints, and the characterization line `g_mn w^m w^n = -(1 - 3M/r)` is exactly null at 3M and -1/2 at 6M. Measured on the h = 0.25 photon grid: the grid balance `a_x` matches `-rho omega^2` at the 1-1.5% level (the O(h^2) symbol error), the photon arc holds an exact null normalization drift 3.9e-3 and radius drift 1.6e-2 over a quarter revolution -- the photon sphere is validated on a quarter-revolution arc because it is an unstable null orbit whose epicyclic exponent lambda = omega = 1/(3 sqrt(3) M) would amplify a 1e-3 grid seed by x533 over a full period -- and the ISCO closes two full revolutions with radius drift 6.1e-3, phase drift 8.4e-3 rad and norm drift 5.4e-4 around the exact -1/2. The two error channels are separated and measured: with the exact analytic connection (`physics::orbits::analytic_christoffel`), the RK4 integrator converges at ratios 16.53/16.40 (fourth order); with the grid pipeline spatial error converges at ratio 3.94 (central differences); and what the full pipeline delivers in dt is pinned at effective order 2.13 (grid-spacing independent) because the trilinear interpolation of the nodal connection is only C^0 across cell faces -- that test is the documented boundary of the pipeline's temporal accuracy, not a bug. `engine/examples/geodesic_orbits.rs` runs both orbits on the frozen connection, writes `geodesic_orbits_static.csv` (particle id, label, revolutions, t, x, y, z, vx, vy, vz, r_areal = rho(1 + M/(2 rho))^2, g_ww), a summary with the measured drifts, and an expanding-BSSN-vacuum leg (`geodesic_orbits_dynamic.csv`) where the stage-consistent integration gains |v| 0.304 -> 0.354 while the frozen path stands still (matching the analytic gauge acceleration e^{(alpha_dot/alpha - 4 d_t phi + 2 d_t phi v^2) T} at 2e-3) -- the frozen-vs-staged distinction in one binary; the example re-runs byte-identical and its CSVs are schema-checked by the suite.
* M5 made the matter coupling stage-consistent on both formulations and closed the ADM gap: `AdmState::step_rk4_with_dust` (with `step_rk4_with_dust_stages`) rebuilds `MatterSources` from the intermediate dust *and* geometry at every RK4 stage, whereas the older `step_rk4_with_matter` holds its `MatterSources` argument fixed at Y_n (kept unchanged, deliberately: callers who pre-build sources get exactly what they always got). `BssnState::step_rk4_with_dust` was refactored into `step_rk4_with_dust_stages` plus a collapse and reproduces its old numbers bit for bit; the dust half of every coupled step is exposed as a shared `DustRk4Stages` (with `collapse`) mirroring the geometry stage structs, so `stage_connections`/`bssn_stage_connections` and the particle pipeline consume one consistent step with zero changes. The `unit_coupled_matter` suite proves per category: every returned (y_i, m_i, k_g i, k_m i) pair is bitwise the pure kernel at that pair (a cross-wired steppering fails), the stage path collapses bitwise onto the classic steppers for one step and ten steps on both formulations, the ADM stage bundles carry the true metric rate per stage (`Gamma^x_0x = -alpha K_xx/gamma_xx` at uniform metric -- the Hubble wiring, the counterpart of BSSN's `Gamma^x_0x = 2 d_t phi`), staged differs from frozen-source measurably on a strain patch (geometry 1.6e-6, dust 3.2e-7 over one step, with the dust rates themselves changing across stages because the gradient term `T^{mu nu} d_j g_{mu nu}` switches on as soon as the metric is inhomogeneous) while the FLRW control conserves the mass bitwise and holds |H| at 3.5e-11 over ten steps, zero dust reduces both coupled steppers to their vacuum steppers exactly, and the particles pipeline keeps its clock-sync, atomic-refusal and stop-at-boundary semantics when driven by dust-coupled BSSN bundles.
* 198 unit tests pass (`cargo test` from inside `engine/`), with one more ignored (`static_slice_stays_bounded_over_a_long_run`) because plain ADM cannot yet hold a long curved run -- it is the receipt for the BSSN phase.

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
`extrinsic_curvature_rhs` changes `d_t K` by `1.28` -- the `-nabla_i
nabla_j alpha` term the gauge created. That closes the loop from the gauge
to the geometry, which no other test does.

The K equation itself then had to be corrected, and Schwarzschild is what
caught it. The code and its unit test both carried
`d_t K_ij = -alpha nabla_i nabla_j alpha + ...`, but ADM requires the
unrepeated `-nabla_i nabla_j alpha` (with `alpha R_ij` next to it). Every
slice tested before had `alpha = 1` or constant `alpha`, where a factor of
`alpha` is invisible; on the Schwarzschild slice, with `alpha` between
0.27 and 0.68 in the domain, it is not. With the factor out, the static
fixed point `D_i D_j alpha = alpha R_ij` holds, and the slice is verified
as one: pointwise against `psi = 1 + M/2r`, `gamma_ij = psi^4 delta_ij`,
zero `K_ij`, the Schwarzschild lapse and zero shift; `|M| = 0` exactly;
`|H|` over a window converging at second order (8.626e-3 at `h = 1/16`
against 2.165e-3 at `h = 1/32`, ratio 3.98); and as a discrete fixed point
where `d_t gamma = d_t alpha = d_t beta = 0` while `d_t K` sits at the
truncation level, also second-order.

That fixed point is static but not stable in this scheme. Evolved, `|H|`
grows as `exp(lambda t)` at a rate set by the grid: shrinking `dt` from
`h/4` to `h/100` changes nothing, a thin-density run (`M = 0.05`) blows up
on the same schedule, and freezing the lapse makes it worse, not better.
Plain ADM carries these weakly hyperbolic modes -- the textbook reason
long-lived evolutions use BSSN. So the long-run bound is recorded as an
ignored test and the roadmap's BSSN phase is what unlocks
`Stable numerical evolution`.

Next up is constraint-solving initial data, then un-ignoring the long-run bound (`Stable numerical evolution`).
Plain ADM is weakly hyperbolic, so it is useful for wave tests but not for
stable long runs.

Remote compute support, visualization, and large simulations remain future work.

## License

License: TBD.
