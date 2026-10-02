use crate::grid::Field;

/// the lapse alpha: one positive scalar per grid point
pub type LapseField<const D: usize> = Field<f64, D>;