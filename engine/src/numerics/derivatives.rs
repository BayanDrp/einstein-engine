use crate::grid::boundary::{BoundaryConfig, BoundaryKind, BoundaryValues};
use crate::grid::Field;

// forward difference: derivative = (f[i+1] - f[i]) / dx
// looks at the next point only, so it's one-sided.
pub fn forward_partial<const NDIM: usize>(
    field: &Field<f64, NDIM>,
    axis: usize,
    boundary: &BoundaryConfig<NDIM>,
    boundary_values: &BoundaryValues<NDIM>,
) -> Field<f64, NDIM> {
    assert!(axis < NDIM);

    let shape = field.grid.shape;
    let spacing = field.grid.spacing;

    let mut result = Field::zeros(field.grid.clone());

    let mut idx = [0usize; NDIM];

    loop {
        if idx[axis] + 1 < shape[axis] {
            let mut next = idx;
            next[axis] += 1;
            
            
            let current_value = *field.get(idx);
            // boundary conditions override the normal stencil
            if boundary.lower[axis] == BoundaryKind::Dirichlet
                && idx[axis] == 0
            {
                let mut next = idx;
                next[axis] += 1;

                let next_value = *field.get(next);
                let boundary_value = boundary_values.lower[axis];

                let derivative =
                    (next_value - boundary_value)
                    / spacing[axis];

                result.set(idx, derivative);
                continue;
            }

            if boundary.upper[axis] == BoundaryKind::Dirichlet
                && idx[axis] == shape[axis] - 1
            {
                let mut prev = idx;
                prev[axis] -= 1;

                let prev_value = *field.get(prev);
                let boundary_value = boundary_values.upper[axis];

                let derivative =
                    (boundary_value - prev_value)
                    / spacing[axis];

                result.set(idx, derivative);
                continue;
            }
            
            let next_value = *field.get(next);

            

            let derivative =
                (next_value - current_value) / spacing[axis];

            result.set(idx, derivative);
        }

        let mut carry = true;

        for d in (0..NDIM).rev() {
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

    result
}

// central difference: derivative = (f[i+1] - f[i-1]) / (2 * dx)
// at the edges we fall back to one-sided stencils.
pub fn central_partial<const NDIM: usize>(
    field: &Field<f64, NDIM>,
    axis: usize,
    boundary: &BoundaryConfig<NDIM>,
    boundary_values: &BoundaryValues<NDIM>,
) -> Field<f64, NDIM> {
    assert!(axis < NDIM);

    let shape = field.grid.shape;
    let spacing = field.grid.spacing;
    let n = shape[axis];
    let dx = spacing[axis];

    let mut result = Field::zeros(field.grid.clone());
    let mut idx = [0usize; NDIM];

    // same point as idx but with `axis` set to `pos`
    let at = |field: &Field<f64, NDIM>, idx: [usize; NDIM], pos: usize| -> f64 {
        let mut nb = idx;
        nb[axis] = pos;
        *field.get(nb)
    };

    loop {
        let i = idx[axis];
        let derivative = if n <= 1 {
            // one point along this axis: no derivative
            0.0
        } else if i > 0 && i + 1 < n {
            // interior point: central stencil
            (at(field, idx, i + 1) - at(field, idx, i - 1)) / (2.0 * dx)
        } else if i == 0 {
            // lower edge
            match boundary.lower[axis] {
                // use the boundary value as the missing neighbor
                BoundaryKind::Dirichlet => (at(field, idx, 1) - boundary_values.lower[axis]) / dx,
                // wrap to the other side
                BoundaryKind::Periodic => (at(field, idx, 1) - at(field, idx, n - 1)) / (2.0 * dx),
                // fall back to a one-sided stencil
                BoundaryKind::Neumann => (at(field, idx, 1) - at(field, idx, 0)) / dx,
            }
        } else {
            // upper edge
            match boundary.upper[axis] {
                BoundaryKind::Dirichlet => (boundary_values.upper[axis] - at(field, idx, n - 2)) / dx,
                BoundaryKind::Periodic => (at(field, idx, 0) - at(field, idx, n - 2)) / (2.0 * dx),
                BoundaryKind::Neumann => (at(field, idx, n - 1) - at(field, idx, n - 2)) / dx,
            }
        };
        result.set(idx, derivative);

        let mut carry = true;

        for d in (0..NDIM).rev() {
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

    result
}