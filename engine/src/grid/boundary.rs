use super::Field;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryKind {
    Dirichlet,
    Neumann,
    Periodic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundaryConfig<const NDIM: usize> {
    pub lower: [BoundaryKind; NDIM],
    pub upper: [BoundaryKind; NDIM],
}

impl<const NDIM: usize> BoundaryConfig<NDIM> {
    /// All faces periodic.
    pub fn periodic() -> Self {
        Self {
            lower: [BoundaryKind::Periodic; NDIM],
            upper: [BoundaryKind::Periodic; NDIM],
        }
    }
}

/// Prescribed data for value-type faces.
///
/// Interpretation depends on the face kind in [`BoundaryConfig`]:
/// - `Dirichlet`: the face value itself.
/// - `Neumann`: the outward normal derivative at the face.
/// - `Periodic`: ignored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundaryValues<const NDIM: usize> {
    pub lower: [f64; NDIM],
    pub upper: [f64; NDIM],
}

impl<const NDIM: usize> BoundaryValues<NDIM> {
    /// Same value on every face.
    pub fn uniform(value: f64) -> Self {
        Self {
            lower: [value; NDIM],
            upper: [value; NDIM],
        }
    }

    /// Zero on every face.
    pub fn zeros() -> Self {
        Self::uniform(0.0)
    }
}

/// Enforces `config` on a scalar `field` in place.
///
/// Semantics per face:
/// - `Dirichlet`: face points are set to the prescribed value.
/// - `Neumann`: face points are extrapolated from the adjacent interior
///   point, `f_edge = f_inner + g * dx`, where `g` is the prescribed
///   outward normal derivative. On a degenerate axis of length 1 there is
///   no interior point, so the face is set to `g * dx`.
/// - `Periodic`: both faces are set to their pointwise average, identifying
///   the endpoints.
pub fn apply<const NDIM: usize>(
    field: &mut Field<f64, NDIM>,
    config: &BoundaryConfig<NDIM>,
    values: &BoundaryValues<NDIM>,
) {
    for a in 0..NDIM {
        for side in 0..2 {
            let (kind, value) = match side {
                0 => (config.lower[a], values.lower[a]),
                _ => (config.upper[a], values.upper[a]),
            };
            apply_face(field, a, side, kind, value, true);
        }
    }
}

/// Like [`apply`], but leaves periodic faces untouched.
///
/// The periodic averaging in `apply` identifies the two end faces as a single
/// physical point, which is right for a grid that stores a duplicated
/// endpoint. An evolving field is sampled at distinct points across the wrap,
/// though, so averaging the ends every step would clip it. Periodic coupling
/// is already carried by the wrapping derivative stencils, so this variant
/// imposes only the Dirichlet and Neumann faces.
///
/// Note that this is still a *scalar* condition: one value per face is shared
/// by every call. Vector and tensor fields need a value per component, which
/// `BoundaryValues` cannot express, so callers must apply this only to fields
/// whose condition is genuinely scalar (the lapse, for now). See
/// `EvolutionState::enforce_boundaries`.
pub fn apply_nonperiodic<const NDIM: usize>(
    field: &mut Field<f64, NDIM>,
    config: &BoundaryConfig<NDIM>,
    values: &BoundaryValues<NDIM>,
) {
    for a in 0..NDIM {
        for side in 0..2 {
            let (kind, value) = match side {
                0 => (config.lower[a], values.lower[a]),
                _ => (config.upper[a], values.upper[a]),
            };
            apply_face(field, a, side, kind, value, false);
        }
    }
}

fn apply_face<const NDIM: usize>(
    field: &mut Field<f64, NDIM>,
    axis: usize,
    side: usize,
    kind: BoundaryKind,
    value: f64,
    include_periodic: bool,
) {
    let shape = field.grid.shape;
    let spacing = field.grid.spacing;
    let grid = field.grid.clone();

    let n = shape[axis];
    if n == 0 {
        return;
    }
    let edge = n - 1;
    let face = if side == 0 { 0 } else { edge };

    match kind {
        BoundaryKind::Dirichlet => {
            grid.for_each_index(|idx| {
                if idx[axis] == face {
                    field.set(idx, value);
                }
            });
        }
        BoundaryKind::Neumann => {
            let dx = spacing[axis];
            grid.for_each_index(|idx| {
                if idx[axis] == face {
                    let f_inner = if n >= 2 {
                        let mut inner = idx;
                        inner[axis] = if side == 0 { 1 } else { edge - 1 };
                        *field.get(inner)
                    } else {
                        0.0
                    };
                    field.set(idx, f_inner + value * dx);
                }
            });
        }
        BoundaryKind::Periodic => {
            if !include_periodic {
                return;
            }
            let mut lo = Vec::new();
            let mut hi = Vec::new();
            grid.for_each_index(|idx| {
                if idx[axis] == 0 {
                    lo.push(*field.get(idx));
                }
                if idx[axis] == edge {
                    hi.push(*field.get(idx));
                }
            });
            let mut li = 0;
            let mut hi_i = 0;
            grid.for_each_index(|idx| {
                if idx[axis] == 0 {
                    field.set(idx, 0.5 * (lo[li] + hi[li]));
                    li += 1;
                }
                if idx[axis] == edge {
                    field.set(idx, 0.5 * (lo[hi_i] + hi[hi_i]));
                    hi_i += 1;
                }
            });
        }
    }
}

impl<const NDIM: usize> BoundaryConfig<NDIM> {
    pub fn dirichlet() -> Self {
        Self {
            lower: [BoundaryKind::Dirichlet; NDIM],
            upper: [BoundaryKind::Dirichlet; NDIM],
        }
    }
    pub fn neumann() -> Self {
        Self {
            lower: [BoundaryKind::Neumann; NDIM],
            upper: [BoundaryKind::Neumann; NDIM],
        }
    }
}