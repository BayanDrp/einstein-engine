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
    let shape = field.grid.shape;
    let spacing = field.grid.spacing;
    let grid = field.grid.clone();

    for a in 0..NDIM {
        let n = shape[a];
        if n == 0 {
            continue;
        }
        let edge = n - 1;

        for (side, kinds, vals) in [
            (0usize, &config.lower, &values.lower),
            (1usize, &config.upper, &values.upper),
        ] {
            let face = if side == 0 { 0 } else { edge };
            match kinds[a] {
                BoundaryKind::Dirichlet => {
                    let v = vals[a];
                    grid.for_each_index(|idx| {
                        if idx[a] == face {
                            field.set(idx, v);
                        }
                    });
                }
                BoundaryKind::Neumann => {
                    let dx = spacing[a];
                    let g = vals[a];
                    grid.for_each_index(|idx| {
                        if idx[a] == face {
                            let f_inner = if n >= 2 {
                                let mut inner = idx;
                                inner[a] = if side == 0 { 1 } else { edge - 1 };
                                *field.get(inner)
                            } else {
                                0.0
                            };
                            field.set(idx, f_inner + g * dx);
                        }
                    });
                }
                BoundaryKind::Periodic => {
                    let mut lo = Vec::new();
                    let mut hi = Vec::new();
                    grid.for_each_index(|idx| {
                        if idx[a] == 0 {
                            lo.push(*field.get(idx));
                        }
                        if idx[a] == edge {
                            hi.push(*field.get(idx));
                        }
                    });
                    let mut li = 0;
                    let mut hi_i = 0;
                    grid.for_each_index(|idx| {
                        if idx[a] == 0 {
                            field.set(idx, 0.5 * (lo[li] + hi[li]));
                            li += 1;
                        }
                        if idx[a] == edge {
                            field.set(idx, 0.5 * (lo[hi_i] + hi[hi_i]));
                            hi_i += 1;
                        }
                    });
                }
            }
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