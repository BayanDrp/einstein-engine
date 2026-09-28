use super::Field;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryKind {
    Dirichlet,
    Neumann,
    Periodic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundaryConfig<const D: usize> {
    pub lower: [BoundaryKind; D],
    pub upper: [BoundaryKind; D],
}

impl<const D: usize> BoundaryConfig<D> {
    /// All faces periodic.
    pub fn periodic() -> Self {
        Self {
            lower: [BoundaryKind::Periodic; D],
            upper: [BoundaryKind::Periodic; D],
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
pub struct BoundaryValues<const D: usize> {
    pub lower: [f64; D],
    pub upper: [f64; D],
}

impl<const D: usize> BoundaryValues<D> {
    /// Same value on every face.
    pub fn uniform(value: f64) -> Self {
        Self {
            lower: [value; D],
            upper: [value; D],
        }
    }

    /// Zero on every face.
    pub fn zeros() -> Self {
        Self::uniform(0.0)
    }
}

/// Calls `f` once per multi-index of a grid with the given shape (row-major).
fn each_index<const D: usize>(shape: [usize; D], mut f: impl FnMut([usize; D])) {
    let mut idx = [0usize; D];
    loop {
        f(idx);
        let mut carry = true;
        for d in (0..D).rev() {
            if carry {
                idx[d] += 1;
                if idx[d] < shape[d] {
                    carry = false;
                } else {
                    idx[d] = 0;
                }
            }
        }
        if carry {
            break;
        }
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
pub fn apply<const D: usize>(
    field: &mut Field<f64, D>,
    config: &BoundaryConfig<D>,
    values: &BoundaryValues<D>,
) {
    let shape = field.grid.shape;
    let spacing = field.grid.spacing;

    for a in 0..D {
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
                    each_index(shape, |idx| {
                        if idx[a] == face {
                            field.set(idx, v);
                        }
                    });
                }
                BoundaryKind::Neumann => {
                    let dx = spacing[a];
                    let g = vals[a];
                    each_index(shape, |idx| {
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
                    each_index(shape, |idx| {
                        if idx[a] == 0 {
                            lo.push(*field.get(idx));
                        }
                        if idx[a] == edge {
                            hi.push(*field.get(idx));
                        }
                    });
                    let mut li = 0;
                    let mut hi_i = 0;
                    each_index(shape, |idx| {
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

impl<const D: usize> BoundaryConfig<D> {
    pub fn dirichlet() -> Self {
        Self {
            lower: [BoundaryKind::Dirichlet; D],
            upper: [BoundaryKind::Dirichlet; D],
        }
    }
    pub fn neumann() -> Self {
        Self {
            lower: [BoundaryKind::Neumann; D],
            upper: [BoundaryKind::Neumann; D],
        }
    }
}