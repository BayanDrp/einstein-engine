# Einstein Engine

[![CI](https://github.com/BayanDrp/einstein-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/BayanDrp/einstein-engine/actions/workflows/ci.yml)

A high performance numerical relativity engine written in Rust for simulating Einstein field equations on discrete spacetime grids.

The engine evolves dynamical spacetime geometry and relativistic matter without assuming background gravitational potentials. It implements both classic ADM and conformal BSSN formulations, coupled relativistic dust hydrodynamics, time dependent geodesic integration, and a WebGL visualization renderer.

**Version 1.0 scope.** This release is a validated 3+1 core. The 1+D generalization remains a research branch. See [SCOPE.md](SCOPE.md) for the supported scope and the promotion criteria for future dimensions.

```
Matter Stress Energy (T_μν)
         │
         ▼
 Einstein Field Equations (G_μν = 8π T_μν)
         │
         ▼
  3+1 Spacetime Geometry (γ_ij, K_ij, α, β^i)
         │
         ▼
  Curvature & Connection (Γ^μ_αβ, R_μν, R^(3))
         │
         ▼
 Geodesic Orbits & Matter Hydrodynamics
         │
         └──────────────► Updated T_μν
                               │
                               ▼
                         Next RK4 Substage
```

---

## Table of Contents

* [Theoretical Foundation](#theoretical-foundation)
* [3+1 Decomposition and ADM Formalism](#31-decomposition-and-adm-formalism)
* [Curvature and Differential Geometry](#curvature-and-differential-geometry)
* [Einstein Constraint Equations](#einstein-constraint-equations)
* [Evolution Systems: ADM vs BSSN](#evolution-systems-adm-vs-bssn)
* [Gauge Conditions](#gauge-conditions)
* [Relativistic Matter and Dust Hydrodynamics](#relativistic-matter-and-dust-hydrodynamics)
* [Geodesic Motion and Particle Dynamics](#geodesic-motion-and-particle-dynamics)
* [Discrete Numerics and Solvers](#discrete-numerics-and-solvers)
* [Validation Benchmarks](#validation-benchmarks)
* [Visualization Pipeline](#visualization-pipeline)
* [Repository Architecture](#repository-architecture)
* [Getting Started](#getting-started)

---

## Theoretical Foundation

General Relativity describes gravitation as the curvature of four dimensional spacetime dictated by the Einstein field equations:

$$
G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2}g_{\mu\nu}R = 8\pi G T_{\mu\nu}
$$

Because these ten coupled nonlinear partial differential equations cannot be solved analytically for generic dynamical configurations (such as binary black hole mergers, gravitational collapse, or inhomogeneous cosmologies), numerical relativity provides the necessary framework to integrate them forward in time.

---

## 3+1 Decomposition and ADM Formalism

Einstein Engine formulates spacetime dynamics through a 3+1 foliation (Arnowitt, Deser, and Misner), slicing four dimensional spacetime into spacelike hypersurfaces parameterized by coordinate time $t$.

The four dimensional line element is expressed as:

$$
ds^2 = g_{\mu\nu} dx^\mu dx^\nu = -\alpha^2 dt^2 + \gamma_{ij}(dx^i + \beta^i dt)(dx^j + \beta^j dt)
$$

The fundamental geometric variables stored on the grid are:

* **Spatial Metric ($\gamma_{ij}$)**: Symmetric rank 2 tensor describing proper distances inside the spatial hypersurface.
* **Extrinsic Curvature ($K_{ij}$)**: Measures the embedding curvature and time rate of change of the spatial metric along hypersurface normals:
  $$K_{ij} = -\frac{1}{2}\mathcal{L}_n \gamma_{ij} = -\frac{1}{2\alpha}(\partial_t - \mathcal{L}_\beta)\gamma_{ij}$$
* **Lapse Scalar ($\alpha$)**: Measures the rate of proper time elapsed for normal observers per unit coordinate time ($d\tau = \alpha dt$).
* **Shift Vector ($\beta^i$)**: Specifies coordinate dragging across neighboring spatial slices.

The 4D Lorentzian metric $g_{\mu\nu}$ with signature $(-+++)$ is reconstructed pointwise:

$$
g_{00} = -\alpha^2 + \beta_k \beta^k,\quad g_{0i} = \beta_i,\quad g_{ij} = \gamma_{ij}
$$

---

## Curvature and Differential Geometry

The engine constructs the full curvature hierarchy from the metric fields:

### Spatial Connection
The Christoffel symbols of the second kind on the spatial slice are:

$$
\Gamma^k_{ij} = \frac{1}{2}\gamma^{kl}\left(\partial_i \gamma_{jl} + \partial_j \gamma_{il} - \partial_l \gamma_{ij}\right)
$$

### Spacetime Connection with Time Derivatives
To correctly integrate geodesics in non stationary spacetimes, the engine provides `christoffel_symbols_with_time_derivatives`. It evaluates the time rate of change of the four metric $\partial_t g_{\mu\nu}$ directly from intermediate evolution rates:

$$
\Gamma^\mu_{\alpha\beta} = \frac{1}{2}g^{\mu\nu}\left(\partial_\alpha g_{\beta\nu} + \partial_\beta g_{\alpha\nu} - \partial_\nu g_{\alpha\beta}\right)
$$

### Riemann and Ricci Tensors
Curvature is computed by evaluating derivatives of the connection:

$$
R^\mu_{\ \nu\rho\sigma} = \partial_\rho \Gamma^\mu_{\nu\sigma} - \partial_\sigma \Gamma^\mu_{\nu\rho} + \Gamma^\mu_{\lambda\rho}\Gamma^\lambda_{\nu\sigma} - \Gamma^\mu_{\lambda\sigma}\Gamma^\lambda_{\nu\rho}
$$

Contracting indices produces the Ricci tensor $R_{\mu\nu} = R^\lambda_{\ \mu\lambda\nu}$ and the Ricci scalar $R = g^{\mu\nu}R_{\mu\nu}$. The spatial 3-curvature scalar $R^{(3)} = \gamma^{ij}R_{ij}$ is computed analogously on each hypersurface.

### Covariant Derivatives
The engine implements covariant differentiation for vectors, covectors, and mixed tensors:

$$
D_k V^i = \partial_k V^i + \Gamma^i_{kj}V^j,\quad D_k W_i = \partial_k W_i - \Gamma^j_{ki}W_j,\quad D_j T^i_{\ j} = \partial_j T^i_{\ j} + \Gamma^i_{jk}T^k_{\ j} - \Gamma^k_{jk}T^i_{\ k}
$$

---

## Einstein Constraint Equations

In the 3+1 split, the field equations divide into evolution equations and kinematic constraint equations that must hold on every spatial slice:

### Hamiltonian Constraint
Relates the spatial intrinsic curvature to the extrinsic curvature and matter energy density $\rho$:

$$
H = R^{(3)} + K^2 - K_{ij}K^{ij} - 16\pi\rho = 0
$$

where $K = \gamma^{ij}K_{ij}$ is the trace of extrinsic curvature.

### Momentum Constraint
Relates spatial divergences of extrinsic curvature to the matter momentum density $S^i$:

$$
M^i = D_j\left(K^{ij} - \gamma^{ij}K\right) - 8\pi S^i = 0
$$

The engine includes constraint monitors (`validation::constraints::residuals`) that measure maximum residuals $|H|_{\max}$ and $|M^i|_{\max}$ across the grid, diagnosing numerical drift and verifying convergence.

---

## Evolution Systems: ADM vs BSSN

### Classic ADM Equations
The standard ADM evolution equations govern the time derivative of the spatial metric and extrinsic curvature:

$$
\partial_t \gamma_{ij} = -2\alpha K_{ij} + \mathcal{L}_\beta \gamma_{ij}
$$

$$
\partial_t K_{ij} = -D_i D_j \alpha + \alpha\left(R_{ij} - 2 K_{ik}K^k_{\ j} + K K_{ij}\right) + \mathcal{L}_\beta K_{ij}
$$

Here $\mathcal{L}_\beta$ denotes the Lie derivative along the shift vector:

$$
\mathcal{L}_\beta \gamma_{ij} = \beta^k \partial_k \gamma_{ij} + \gamma_{kj}\partial_i \beta^k + \gamma_{ik}\partial_j \beta^k
$$

While ADM is mathematically clean, it is weakly hyperbolic. Numerical roundoff and constraint violations grow exponentially over extended simulations, making long term stability impossible on curved backgrounds.

### The BSSN Formalism
To guarantee strong hyperbolicity and long term numerical stability, the engine implements the Baumgarte, Shapiro, Shibata, and Nakamura (BSSN) formulation.

BSSN transforms the ADM variables into conformal quantities:

1. **Conformal Factor ($\phi$)**:
   $$\phi = \frac{1}{12}\ln(\det\gamma)$$
2. **Conformal Metric ($\tilde{\gamma}_{ij}$)**:
   $$\tilde{\gamma}_{ij} = e^{-4\phi}\gamma_{ij} = (\det\gamma)^{-1/3}\gamma_{ij},\quad \det\tilde{\gamma} = 1$$
3. **Trace of Extrinsic Curvature ($K$)**:
   $$K = \gamma^{ij}K_{ij}$$
4. **Conformal Trace Free Extrinsic Curvature ($\tilde{A}_{ij}$)**:
   $$\tilde{A}_{ij} = e^{-4\phi}\left(K_{ij} - \frac{1}{3}\gamma_{ij}K\right),\quad \operatorname{tr}\tilde{A} = \tilde{\gamma}^{ij}\tilde{A}_{ij} = 0$$
5. **Conformal Connection Functions ($\tilde{\Gamma}^i$)**:
   $$\tilde{\Gamma}^i = \tilde{\gamma}^{jk}\tilde{\Gamma}^i_{jk}$$

### BSSN Evolution Equations with Hyperbolic Corrections
The BSSN state evolves via coupled PDEs:

$$
\partial_t \phi = -\frac{1}{6}\alpha K + \beta^k \partial_k \phi + \frac{1}{6}\partial_k \beta^k
$$

$$
\partial_t \tilde{\gamma}_{ij} = -2\alpha \tilde{A}_{ij} + \beta^k \partial_k \tilde{\gamma}_{ij} + \tilde{\gamma}_{ik}\partial_j \beta^k + \tilde{\gamma}_{jk}\partial_i \beta^k - \frac{2}{3}\tilde{\gamma}_{ij}\partial_k \beta^k
$$

$$
\partial_t K = -D^i D_i \alpha + \alpha\left(\tilde{A}_{ij}\tilde{A}^{ij} + \frac{1}{3}K^2\right) + 4\pi\alpha(\rho + S) + \beta^k \partial_k K
$$

$$
\partial_t \tilde{A}_{ij} = e^{-4\phi}\left[-D_i D_j \alpha + \alpha(R_{ij} - 8\pi S_{ij})\right]^{\mathrm{TF}} + \alpha\left(K \tilde{A}_{ij} - 2\tilde{A}_{ik}\tilde{A}^k_{\ j}\right) + \mathcal{L}_\beta \tilde{A}_{ij}
$$

The physical Ricci tensor splits conformally into $R_{ij} = \tilde{R}_{ij} + R^\phi_{ij}$. In the engine, $\tilde{R}_{ij}$ incorporates the hyperbolic connection replacement:

$$
\tilde{R}_{ij} = -\frac{1}{2}\tilde{\gamma}^{kl}\partial_k \partial_l \tilde{\gamma}_{ij} + \tilde{\gamma}_{k(i}\partial_{j)}\tilde{\Gamma}^k + \tilde{\Gamma}^k \tilde{\Gamma}_{(ij)k} + \tilde{\gamma}^{lm}\left(2\tilde{\Gamma}^k_{l(i}\tilde{\Gamma}_{j)km} + \tilde{\Gamma}^k_{im}\tilde{\Gamma}_{klj}\right)
$$

### Algebraic Enforcement
At every Runge Kutta substage, the algebraic constraints $\det\tilde{\gamma} = 1$ and $\operatorname{tr}\tilde{A} = 0$ are explicitly reprojected:

$$
\tilde{\gamma}_{ij} \leftarrow (\det\tilde{\gamma})^{-1/3}\tilde{\gamma}_{ij},\quad \tilde{A}_{ij} \leftarrow \tilde{A}_{ij} - \frac{1}{3}\tilde{\gamma}_{ij}(\tilde{\gamma}^{kl}\tilde{A}_{kl})
$$

---

## Gauge Conditions

Coordinate gauges determine how spatial coordinates and time slices evolve:

* **1+log Slicing (Lapse Gauge)**:
  $$\partial_t \alpha = \beta^i \partial_i \alpha - 2\alpha K$$
  This condition slows down coordinate time flow near strong gravitational fields (singularity avoidance), preventing spatial grid collapse near black hole horizons.
* **Frozen Shift**:
  $$\partial_t \beta^i = 0$$
  Maintains stationary spatial coordinates in isotropic configurations.

---

## Relativistic Matter and Dust Hydrodynamics

The stress energy tensor $T_{\mu\nu}$ for a perfect fluid is:

$$
T_{\mu\nu} = (\rho_0 + \rho_0 \epsilon + P)u_\mu u_\nu + P g_{\mu\nu}
$$

For pressureless dust ($P = 0$), $T_{\mu\nu} = \rho_0 u_\mu u_\nu$.

### 3+1 Matter Projections
The source terms entering Einstein's equations are:

* Energy density: $\rho = n^\mu n^\nu T_{\mu\nu}$
* Momentum density: $S^i = -\gamma^{ij}n^\mu T_{\mu j}$
* Spatial stress tensor: $S_{ij} = \gamma^\mu_{\ i}\gamma^\nu_{\ j}T_{\mu\nu}$
* Trace of stress: $S = \gamma^{ij}S_{ij}$

### Relativistic Dust Evolution
The engine models dynamic dust via conserved variables:

* Conserved density: $\tilde{D} = \sqrt{\det\gamma}\,\rho W$
* Conserved momentum: $\tilde{P}_j = \tilde{D} u_j$

where $W = (1 - v^2)^{-1/2}$ is the Lorentz factor.

The evolution equations are:

$$
\partial_t \tilde{D} + \partial_i \left[\tilde{D}\left(\alpha v^i - \beta^i\right)\right] = 0
$$

$$
\partial_t \tilde{P}_j + \partial_i \left[\tilde{P}_j\left(\alpha v^i - \beta^i\right)\right] = \frac{1}{2}\alpha \sqrt{\det\gamma}\,T^{\mu\nu}\partial_j g_{\mu\nu}
$$

### Primitive Inversion
Physical 3-velocity $v^i$ is recovered algebraically from conserved momentum $\tilde{P}_j$:

$$
\mathcal{R} = \frac{\gamma^{ij}\tilde{P}_i \tilde{P}_j}{\tilde{D}^2},\quad v^2 = \frac{\mathcal{R}}{1 + \mathcal{R}} < 1,\quad W = \sqrt{1 + \mathcal{R}}
$$

This inversion is unconditionally subluminal and protected by a numerical vacuum floor.

### Stage Consistent RK4 Matter Coupling
Intermediate geometry and matter states are synchronized at every Runge Kutta substage ($y_0, y_2, y_3, y_4$). Matter sources are rebuilt dynamically rather than held frozen across time steps.

---

## Geodesic Motion and Particle Dynamics

Test particles follow timelike or null geodesics through the curved spacetime.

### Coordinate Time Geodesic Equation
Rather than parameterizing by proper time $\tau$ (which diverges for photons), trajectories are integrated directly in coordinate time $t$:

$$
\frac{dx^i}{dt} = v^i
$$

$$
\frac{dv^i}{dt} = -\Gamma^i_{\mu\nu}w^\mu w^\nu + v^i \Gamma^0_{\mu\nu}w^\mu w^\nu
$$

where $w^\mu = (1, v^x, v^y, v^z)$ is the coordinate 4-velocity vector.

### Stage Consistent Time Aware Christoffel Symbols
When evolving spacetime alongside moving particles, spatial interpolation of static Christoffel symbols introduces phase lag errors. The engine couples particles to BSSN evolution stages by pairing intermediate rates ($y_i, k_i$) and assembling time dependent connection bundles $\Gamma^\mu_{\alpha\beta}(t_k)$.

### Schwarzschild Black Hole Orbits
The engine verifies circular orbits on isotropic Schwarzschild slices:

$$
ds^2 = -\left(\frac{1 - M/(2\rho)}{1 + M/(2\rho)}\right)^2 dt^2 + \left(1 + \frac{M}{2\rho}\right)^4 (dx^2 + dy^2 + dz^2)
$$

* **Photon Sphere ($r_{\mathrm{areal}} = 3M$)**: Null circular orbit at isotropic radius $\rho = \frac{M}{2}(2 + \sqrt{3})$. Unstable orbit used to test epicyclic amplification and null normalization preservation.
* **Innermost Stable Circular Orbit ($r_{\mathrm{areal}} = 6M$)**: Timelike circular orbit at isotropic radius $\rho = \frac{M}{2}(5 + 2\sqrt{6})$. Keplerian coordinate angular velocity: $\omega^2 = M / r^3$.

---

## Discrete Numerics and Solvers

* **Const Generic Grid Layout**: `Grid<const NDIM: usize>` manages contiguous row major memory layouts, multi dimensional indices, coordinate mapping, and boundary conditions.
* **Spatial Derivatives**: Second order central differences for interior nodes, with edge handling for periodic, Dirichlet, and Neumann boundaries.
* **Trilinear Spatial Interpolation**: Computes continuous off lattice metric tensors and Christoffel connections for particle tracking.
* **Runge Kutta 4 (RK4)**: Classical fourth order time integration advancing state bundles $(y_n \to y_{n+1})$ with stage caching.

---

## Validation Benchmarks

The suite validates every component against analytic benchmarks:

* **Minkowski Space Invariance**: Flat initial data preserves $\Gamma^\mu_{\alpha\beta} = 0$, $R_{\mu\nu} = 0$, $H = 0$, and $M^i = 0$ bit for bit over hundreds of RK4 steps.
* **Gravitational Wave Speed**: A transverse traceless wave packet evolved with ADM moves at measured speed $v = 0.9977c$ (matching the theoretical second order difference limit of $0.9984c$).
* **Temporal RK4 Order**: Refinement tests confirm fourth order temporal convergence with error ratios scaling as $10^4$ for $10\times$ step size reductions.
* **Schwarzschild Static Fixed Point**: The isotropic slice confirms zero shift, zero momentum constraints, and spatial Hamiltonian convergence: $|H|_{h=1/16} = 8.626\times 10^{-3}$ vs $|H|_{h=1/32} = 2.165\times 10^{-3}$ (ratio 3.98, second order).
* **ISCO Orbit Closure**: Particles complete two full revolutions closing within radius drift $6.1\times 10^{-3}M$ and phase error $8.4\times 10^{-3}\text{ rad}$.
* **BSSN Long Run Boundedness**: While plain ADM develops hyperbolic runaway modes on curved slices, BSSN evolves the static Schwarzschild slice stably across extended horizons.

---

## Visualization Pipeline

Simulation outputs are visualized using a standalone web renderer located in `visualization/`:

* **Three.js Scene**: Renders luminous particle halos, matter spheres, and black hole horizons with accretion ring shaders.
* **Trajectory Playback**: Loads normalized CSV tracks with adaptive camera sweeps and coordinate time HUD overlays.
* **Headless Capture**: `bin/render.mjs` orchestrates headless Chrome via Puppeteer, stepping scenes deterministically frame by frame and piping PNG buffers into `ffmpeg` for 60fps MP4 video generation.

---

## Repository Architecture

```
einstein-engine/
├── Cargo.toml                       # Virtual workspace manifest
├── .github/workflows/ci.yml         # CI pipeline (check, format, clippy, tests, docs)
│
├── engine/                          # Numerical relativity core crate
│   ├── Cargo.toml                   # Library configuration & test harnesses
│   ├── src/
│   │   ├── lib.rs                   # Public crate module exports
│   │   ├── grid/                    # Grid definitions, memory layout, boundaries
│   │   ├── geometry/                # Tensors, metric, Christoffel, Riemann, Ricci
│   │   ├── numerics/                # Derivatives, trilinear interpolation, RK4
│   │   ├── relativity/              # ADM, BSSN, constraints, gauge, matter sources
│   │   ├── physics/                 # Dust hydrodynamics, geodesics, particles, orbits
│   │   ├── initial_data/            # Analytic initial slices (Schwarzschild)
│   │   └── validation/              # Constraint monitors and residual diagnostics
│   └── examples/
│       ├── demo.rs                  # Pointwise metric & Christoffel inspection
│       ├── evolution.rs             # ADM gravitational plane wave transport
│       ├── matter_response.rs       # FLRW dust geometry response
│       ├── coupled_dust.rs          # Coupled dust & BSSN spacetime evolution
│       └── geodesic_orbits.rs       # Photon sphere and ISCO orbit simulations
│
├── tests/                           # 28 dedicated integration test suites
│   ├── grid/                        # Boundary conditions and fields
│   ├── numerics/                    # Spatial stencils, interpolation, RK4 order
│   ├── geometry/                    # Christoffel, Riemann, Ricci, covariant derivatives
│   ├── relativity/                  # ADM, BSSN, gauge slicing, matter coupling
│   ├── physics/                     # Geodesics, particles, dust, orbits
│   ├── initial_data/                # Schwarzschild slice properties
│   └── validation/                  # Constraint preservation tests
│
└── visualization/                   # Display layer for trajectory simulation
    ├── package.json                 # Node dependencies (Three.js, Puppeteer)
    ├── bin/
    │   ├── server.mjs               # Local playback web server
    │   └── render.mjs               # Headless frame capture and MP4 encoder
    ├── web/
    │   ├── index.html               # WebGL canvas entry page
    │   └── src/                     # Scene staging, ingest, layout, shaders
    └── test/                        # Node test runner suite for visualizer
```

---

## Getting Started

### Prerequisites

* **Rust Toolchain**: Stable release (2021 edition compatible).
* **Node.js**: Version 20 or higher (for the visualization package).
* **FFmpeg**: Required only for headless MP4 video rendering.

### Build and Compilation

Verify that all crates and test targets compile cleanly:

```bash
cargo check --workspace --all-targets
```

Build optimized release artifacts:

```bash
cargo build --release
```

### Running Test Suites

Run the full numerical relativity integration test suite:

```bash
cargo test
```

Run linter checks and generate documentation:

```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo doc --workspace --no-deps
```

### Running Simulation Examples

Execute the coupled BSSN dust evolution simulation:

```bash
cargo run --example coupled_dust
```

Execute the Schwarzschild geodesic orbit simulation:

```bash
cargo run --example geodesic_orbits
```

### Running the Visualizer

Navigate into the visualization directory:

```bash
cd visualization
npm install
npm test
```

Start the interactive development server:

```bash
node bin/server.mjs
```

Open `http://localhost:8765` in your browser to inspect orbital trajectories.

To render an MP4 simulation video via headless Chrome:

```bash
node bin/render.mjs data/schwarzschild -q h --seconds 14 --out schwarzschild_orbit
```
