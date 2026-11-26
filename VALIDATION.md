# Validation

This document is the release contract for Einstein Engine 1.0. It defines the ten validation scenarios, the quantity each scenario measures, and the tolerance that must hold before the release can be tagged.

The suite is a public API test. It does not replace the integration tests. It is a small end to end gate that proves the engine still reproduces the benchmarks that define the 3+1 release.

## Scope

This validation covers the 3+1 core and its visualization boundary:

1. ADM and BSSN evolution.
2. Relativistic dust and matter coupling.
3. Geodesic and particle integration.
4. Schwarzschild initial data and constraint monitoring.
5. CSV output used by the visualization pipeline.
6. WebGL rendering and frame playback.

The 1+D extension is a research branch and is not validated here. See [SCOPE.md](SCOPE.md).

Passing this suite is evidence against known failure modes. It is not a proof of correctness for arbitrary initial data.

## How to run

From the repository root:

```bash
npm --prefix visualization ci
cargo run --release --example validate
```

The release profile is required for a meaningful runtime. The curvature chain is roughly 50 times slower in the development profile.

V10 is launched by the validation runner and uses the installed visualization dependencies. The visualization unit tests can also be run separately:

```bash
cd visualization
npm test
```

The visualization CI job must provide headless Chromium with software WebGL enabled.

The validation runner currently uses one full configuration:

```sh
cargo run --release --example validate
```

The runner exits with code 0 only when every scenario passes. A skipped scenario is a failure unless the skip is explicitly allowed by this document. The release configuration target runtime is under 3 minutes.

## Scenario summary

| ID | Scenario | What it proves | Existing coverage |
|---|---|---|---|
| V1 | Minkowski | Flat data stays flat in ADM and BSSN | `tests/relativity/bssn.rs`, `tests/relativity/evolution.rs` |
| V2 | Linear wave | The wave operator and RK4 transport a gravitational wave at the expected speed | `engine/examples/evolution.rs`, README benchmark |
| V3 | Schwarzschild static slice | Isotropic data is on shell and the BSSN right hand side vanishes | `tests/initial_data/schwarzschild.rs`, `tests/relativity/bssn.rs` |
| V4 | ISCO and photon sphere | Geodesic integration tracks analytic circular orbits | `tests/physics/orbits.rs` |
| V5 | FLRW dust | Uniform dust and geometry remain on shell and dilute with volume | `tests/physics/coupled_matter.rs`, `engine/examples/coupled_dust.rs` |
| V6 | Dust conservation | The dust update conserves rest mass on a closed domain | `tests/physics/dust.rs`, `tests/physics/coupled_matter.rs` |
| V7 | BSSN boundedness | BSSN remains bounded on a curved slice where ADM does not | `tests/relativity/bssn.rs` |
| V8 | Stage coupled particles | Stage aware connections improve a time dependent trajectory | `tests/physics/geodesics_time.rs` |
| V9 | CSV schema and determinism | Output files match the documented schema and repeat exactly | `engine/examples/validate/csv_determinism.rs` |
| V10 | WebGL renderer smoke test | The browser loads visualization data and renders deterministic frames | `engine/examples/validate/webgl.rs`, `engine/examples/validate/webgl_test.mjs` |

## V1 Minkowski

**Purpose.** Flat spacetime is an exact solution. Any nonzero evolution of flat data indicates a sign error, a boundary leak, or a broken stencil.

**Setup.** A periodic grid with `gamma_ij = delta_ij`, `K_ij = 0`, `alpha = 1`, and `beta^i = 0`. Evolve both ADM and BSSN for at least 200 steps with `dt = 0.25 h`.

**Pass criteria.**

1. The Hamiltonian and momentum residuals are below `1e-12` at the start and at the end.
2. The metric, extrinsic curvature, lapse, and shift do not move beyond `1e-12`.
3. The BSSN fields `phi`, `gtilde_ij`, `Atilde_ij`, `Gamma^i`, and `K` stay at their initial values within `1e-12`.
4. No field becomes non finite.

**Why this tolerance.** The data are exactly zero or exactly one, and the stencils of constants cancel. Roundoff is the only expected error. The existing BSSN test already requires an exact zero right hand side.

**Runtime target.** Under 5 seconds.

## V2 Linear wave

**Purpose.** Verify the coupled ADM wave operator and the RK4 transport, not just the pointwise right hand side.

**Setup.** Use the same transverse traceless packet as `engine/examples/evolution.rs`: a plane wave along `x` with `h_yy = A cos(k (x - x0))` and `h_zz = -A cos(k (x - x0))`, `A = 0.01`, and `k = 2 pi` on a periodic grid. Use 128 cells along `x` over length 2.0, so `k h = pi/32`. Evolve 32 RK4 steps at `dt = 0.0125` to reach `t = 0.4`, and measure the crest position at the start and the end to infer the propagation speed.

**Pass criteria.**

1. The wave moves in the correct direction.
2. The measured speed is within 0.5 percent of the finite difference light speed for the chosen `k h`. This scenario measures `0.9981c` against `0.9996c` at `k h = pi/32`.
3. The wave amplitude after transport is within 1 percent of the initial amplitude.
4. No field becomes non finite.

**Why this tolerance.** The measured value is dominated by the second order spatial dispersion of the centered stencil. The 0.5 percent band is wider than the benchmark error and still catches a wrong wave operator or a reversed sign.

**Runtime target.** Under 10 seconds.

## V3 Schwarzschild static slice

**Purpose.** Prove that the isotropic Schwarzschild data are on shell and that the BSSN decomposition has a vanishing right hand side on the static slice.

**Setup.** Use the isotropic Schwarzschild constructor with mass `M = 0.5` or `M = 1.0`. Inspect a window away from the puncture and away from the periodic faces, for example `x, y, z` in `[0.75, 1.25]`. Use the existing coarse and fine grids at `h = 1/16` and `h = 1/32` for the convergence check.

**Pass criteria.**

1. The momentum constraint is exactly zero because `K_ij = 0`.
2. The Hamiltonian residual in the window decreases by a factor between 3 and 5 when the spacing is halved. This is the second order signature.
3. The BSSN right hand side in the window satisfies `|d_t phi| < 1e-6`, `|d_t K| < 1e-3`, `|d_t Atilde_ij| < 1e-6`, and `|d_t Gamma^i| < 1e-6`.
4. The numerical metric and lapse match the analytic isotropic form within `1e-12`.

**Why this tolerance.** The convergence ratio is the strong check. A single absolute residual can be made small by a wrong operator on a coarse grid. The BSSN thresholds are the existing fixed point tolerances and must not be loosened without a measured justification.

**Runtime target.** Under 15 seconds.

## V4 ISCO and photon sphere

**Purpose.** Verify the analytic connection, the geodesic right hand side, and the interpolation used by particle tracking on known circular orbits.

**Setup.** Use the analytic isotropic Schwarzschild connection, not the stage coupled connection. Set `M = 1`. For the ISCO use areal radius `6M`, `dt = 0.05`, `h = 0.25`, and two full revolutions. For the photon sphere use areal radius `3M`, `dt = 0.01`, `h = 0.25`, and a quarter revolution.

**Pass criteria.**

1. ISCO areal radius drift is below `0.012 M`.
2. ISCO phase error is below `0.02 rad`.
3. ISCO normalization drift around `g_ww = -1/2` is below `1.2e-3`.
4. ISCO closure error is below `0.07`.
5. The late quarter radius error is below eight times the early quarter error, so there is no secular growth.
6. Photon sphere drift is below `0.04 M`, phase error is below `0.02 rad`, `|g_ww|` is below `0.008`, and the endpoint error is below `0.04`.
7. The particle stays in the orbital plane within `1e-12`.

**Why this tolerance.** The photon sphere is an unstable orbit, so its drift is amplified by grid error and is allowed to be larger. The ISCO is stable and should close much more tightly. These are the measured bounds of the existing tests.

**Runtime target.** Under 30 seconds.

## V5 FLRW dust

**Purpose.** Verify genuinely coupled dust and geometry on a homogeneous patch, not the fixed source response diagnostic.

**Setup.** Use the flat FLRW patch from `engine/examples/coupled_dust.rs` Phase A: `gamma_ij = delta_ij`, `K_ij = C delta_ij`, uniform dust with `6 C^2 = 16 pi rho_0`, `C = 0.1`, periodic boundaries, and at least ten coupled BSSN steps.

**Pass criteria.**

1. The initial Hamiltonian residual is below `1e-12` and the initial momentum residual is below `1e-14`.
2. The metric, extrinsic curvature, and lapse remain uniform to exact equality across the grid.
3. After ten steps the Hamiltonian residual is below `1e-9` and the momentum residual is below `1e-12`.
4. The total conserved dust density is unchanged to exact equality or to roundoff.
5. The stage driven BSSN path and the classic BSSN path agree to exact equality.

**Why this tolerance.** The patch is constructed to satisfy the constraints exactly, and uniformity removes spatial truncation from the evolution. A coupled run that loses uniformity or drifts the constraints has a real coupling bug. The fixed source `matter_response` example is not used here because its own documentation says it is not a cosmology run.

**Runtime target.** Under 15 seconds.

## V6 Dust conservation

**Purpose.** Verify the dust update conserves rest mass on a closed domain.

**Setup.** Use a periodic domain with either a homogeneous dust state or a smooth bump. Evolve the dust with the frozen-geometry stepper that the coupled steppers reuse underneath, for enough steps to move the profile across several cells. Include one run with zero velocity, where the momentum should stay exactly zero.

**Pass criteria.**

1. The total conserved density is constant to `1e-14` relative error, or bitwise equal where the arithmetic permits.
2. The total momentum stays zero for the zero velocity run.
3. The continuity residual is below `1e-10` in the smooth region.
4. No density becomes negative or non finite.

**Why this tolerance.** On a periodic domain there is no boundary flux, so the only loss should be roundoff. This is rest mass conservation, not a claim of global energy conservation in general relativity.

**Runtime target.** Under 15 seconds.

## V7 BSSN boundedness

**Purpose.** Show that BSSN remains bounded on the static Schwarzschild slice over a long run.

**Setup.** Use mass `M = 1`, a `16^3` grid with `h = 1/16`, periodic boundaries, and 40 steps with `dt = 0.5 h`. Inspect the window `[0.8, 1.2]`.

**Pass criteria.**

1. Every evolved field remains finite at every step.
2. The final trace `K` and the final conformal `Atilde` maximum are each below `1.0`.
3. The worst Hamiltonian residual over the run is less than 1000 times the initial truncation residual.
4. The runner reports the actual growth ratio, not just pass or fail.

**Why this tolerance.** The initial residual is nonzero because the slice is only discretely on shell. The bound is a stability check, not an accuracy check. Accuracy is covered by V3. The known ADM instability may be run as a negative control, but it is not a pass criterion.

**Runtime target.** Under 60 seconds.

## V8 Stage coupled particles

**Purpose.** Show that stage aware connections improve a trajectory when the geometry changes inside a time step.

**Setup.** Use the exponential scale factor background from `tests/physics/geodesics_time.rs`, or the coupled FLRW stage data. Run the same particle with two connection policies: the stage aware bundles and a frozen bundle taken at the start of the step. Compare both against the analytic velocity.

**Pass criteria.**

1. The stage aware velocity error against the analytic solution is below `1e-5`.
2. The stage aware momentum drift is below `1e-5`.
3. The stage aware error is strictly smaller than the frozen connection error at the same step size.
4. The particle clock matches the accumulated step time within `1e-9`.

**Why this tolerance.** The improvement over the frozen baseline is the point of the scenario. Without that comparison the test only repeats V4.

**Runtime target.** Under 20 seconds.

## V9 CSV schema and determinism

**Purpose.** Protect the output contract used by the visualization and make repeated runs reproducible.

**Setup.** Run each CSV producing example twice into separate temporary directories. Cover `geodesic_orbits`, `coupled_dust`, and `matter_response`. Compare the files byte for byte on the same build and platform.

**Pass criteria.**

1. Each example exits with code 0.
2. Every expected file exists and its header matches the documented schema exactly.
3. Every numeric field parses and is finite.
4. Time columns are monotonic and row counts meet the documented minimum.
5. The two runs are byte identical on the same build and platform.
6. Output contains no absolute paths, timestamps, or locale dependent text.

**CSV files covered.**

1. `geodesic_orbits_static.csv`: `particle_id,label,revolutions,t,x,y,z,vx,vy,vz,r_areal,g_ww`, at least 4000 data rows
2. `geodesic_orbits_static_summary.csv`: `label,revolutions,r_areal_target,omega,drift_areal_max,phase_err_max_rad,g_ww_drift_max,end_err,closure_full_rev,orbit_period_coord`, at least 2 data rows
3. `geodesic_orbits_dynamic.csv`: `particle_id,mode,t,x,y,z,vx,vy,vz,speed`, at least 2 data rows
4. `coupled_dust.csv`: `step,t,dt,min_alpha,max_abs_K,rho_mean,rho_min,max_abs_v,Hmax,Mmax,mass_rel_err`, at least 17 data rows
5. `coupled_dust_transport.csv`: `t,profile_err,mass_rel_err,Hmax,max_abs_v`, at least 1 data row
6. `matter_response.csv`: `step,t,min_alpha,max_abs_K,Hmax,Mmax`, at least 17 data rows

The `coupled_dust` and `matter_response` examples write to the current directory, so the runner must execute them in isolated temporary directories.

**Determinism policy.** Byte equality is required on the same build and platform. Cross platform byte equality is not required because libm implementations of `ln`, `powf`, and `sqrt` can differ. On another platform, compare numbers within a documented tolerance instead.

**Why this tolerance.** The existing `example_runs_and_writes_reproducible_csv` test already compares two real runs of `geodesic_orbits` byte for byte. V9 extends that contract to every producer used by the visualization.

**Runtime target.** Under 30 seconds.

## V10 WebGL renderer smoke test

**Purpose.** Verify that the visualization loads simulation data in a browser and renders it through the real WebGL pipeline. This scenario checks the display path, not the physical correctness of the trajectories.

**Setup.** The V10 runner starts the visualization server and loads `data/schwarzschild` in headless Chromium controlled by Puppeteer. Enable SwiftShader so the test uses software WebGL and does not depend on a GPU. Load the page in capture mode at 1280 by 720 pixels. Render the first, middle, and final playback frames with bloom enabled, then repeat with bloom disabled. Do not encode an MP4 or require ffmpeg.

**Pass criteria.**

1. The page becomes ready within 30 seconds, and no page error or failed application resource request occurs.
2. The browser creates a WebGL context, the context is not lost during rendering, and the scene contains the five visible tracks from `data/schwarzschild`. The summary CSV is skipped because it is not a trajectory.
3. Each selected frame produces at least one WebGL draw call. Camera and visible body positions remain finite.
4. Re-rendering the same frame after another frame produces identical body transforms, camera position, and time display.
5. The bloom and no-bloom paths both render successfully.
6. The test does not compare pixels against a golden image. Browser and software renderer differences make cross-platform pixel equality unreliable.

**Why this tolerance.** V9 checks the CSV contract and repeatable output. V10 checks the next boundary, from CSV loading through scene construction and WebGL drawing. Structural frame checks catch broken rendering without making the release gate depend on GPU-specific pixels.

**Runtime target.** Under 30 seconds.

## Current implementation status

The runner is located in `engine/examples/validate/`, with one module per scenario. V1 through V10 are implemented and wired. The entry point runs every scenario and exits with code 1 if any scenario fails.

## Release checklist

Tag `v1.0.0` only when all of the following hold on the exact commit being tagged:

1. `cargo fmt --all -- --check` passes.
2. `cargo clippy --workspace --all-targets -- -D warnings` passes.
3. `cargo test --workspace` passes.
4. `cargo run --release --example validate` passes all ten validation scenarios.
5. `cd visualization && npm test` passes the visualization unit tests.
6. `cargo doc --workspace --no-deps` passes with warnings denied.
7. The working tree is clean and CI is green.
8. `engine/Cargo.toml` contains `version = "1.0.0"`.
9. README, SCOPE, VALIDATION, and both license files are present.

## Changing a tolerance

A tolerance may change only with a measured justification. Record the old value, the new value, the reason, and the commit in the same change. Do not relax a tolerance to make a failing scenario pass. If a scenario becomes flaky, fix the setup or reduce the runtime, not the physics.

## Known limitations

1. The suite validates the 3+1 core only.
2. Most scenarios use periodic boundaries. Dirichlet and Neumann boundaries are covered by the integration tests, not by this gate.
3. There is no dedicated gauge scenario. V3 and V7 exercise the 1+log lapse indirectly.
4. Cross platform bitwise reproducibility of CSV output is not claimed.
5. Passing the suite does not establish convergence for arbitrary initial data.
