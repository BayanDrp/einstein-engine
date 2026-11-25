# Scope

This document defines what Einstein Engine supports in version 1.0 and what remains a research branch.

## Version 1.0

Version 1.0 is a validated 3+1 numerical relativity core. The supported scope is:

1. Classic ADM and conformal BSSN evolution on a fixed spatial grid.
2. Relativistic dust coupled to the spacetime geometry at every Runge Kutta substage.
3. Time dependent geodesic integration for test particles and photons.
4. Constraint monitoring, analytic Schwarzschild initial data, and benchmark tests.
5. A WebGL visualization pipeline for simulation output.

The engine is intended for research and education on known analytic and weakly dynamical configurations. It is not a production numerical relativity code for binary black hole mergers or other highly dynamical vacuum spacetimes.

## The 1+D Research Branch

The code base uses const generics for the spatial dimension D and the spacetime dimension N. That design makes a 1+D extension plausible. A throwaway probe has shown that the BSSN equations can be written for general D while preserving the 3+1 results. The probe is not part of version 1.0.

A supported 1+D evolution still needs all of the following:

1. Generalized initial data, starting with Schwarzschild Tangherlini slices for D = 4.
2. Matter and evolution call sites that use N = D + 1 instead of N = 4.
3. Dimension aware gauge, boundary, and wave extraction treatment.
4. Convergence, constraint preservation, and long term stability tests in the target dimension.
5. Published comparisons against established higher dimensional numerical relativity results.

Until those pieces exist and are validated, 1+D work is a research branch. It should not be presented as a supported feature, and it should not be merged into the stable line without the tests above.

## Out of Scope for 1.0

1. Any claim of production 1+D evolution.
2. Binary black hole mergers, gravitational wave catalogs, or parameter estimation.
3. Exotic matter, electromagnetic fields, or cosmology beyond the existing dust model.

## Promotion Criteria

A 1+D branch can become a supported feature only when it passes the full 3+1 regression suite, reproduces a known higher dimensional static solution, demonstrates the expected convergence order, and survives a long term evolution without constraint growth. The branch must also document its coordinate choices and boundary treatment.
