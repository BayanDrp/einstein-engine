use crate::geometry::tensor::Vector;
use crate::grid::Field;

/// the shift vector beta^i: one vector per grid point
pub type ShiftField<const D: usize> = Field<Vector<D>, D>;