use crate::geometry::tensor::Tensor2;
use crate::grid::Field;

/// the extrinsic curvature K_ij: one symmetric tensor per grid point
pub type ExtrinsicCurvature<const D: usize> = Field<Tensor2<D>, D>;