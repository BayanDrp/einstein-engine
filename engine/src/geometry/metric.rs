use crate::grid::Field;
use super::tensor::{Tensor2, Vector};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metric<const D: usize> {
    pub spatial: Tensor2<D>,
    pub lapse: f64,
    pub shift: Vector<D>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpacetimeMetric<const D: usize> {
    pub tt: f64,
    pub ti: Vector<D>,
    pub spatial: Tensor2<D>,
}

pub type MetricField<const D: usize> = Field<Metric<D>, D>;

impl<const D: usize> Metric<D> {
    pub fn new(spatial: Tensor2<D>, lapse: f64, shift: Vector<D>) -> Self {
        assert!(D > 0);
        assert!(lapse.is_finite() && lapse > 0.0);

        Self {
            spatial,
            lapse,
            shift,
        }
    }

    pub fn lower_shift(&self) -> Vector<D> {
        let mut result = Vector::zero();

        for i in 0..D {
            for j in 0..D {
                result[i] += self.spatial[(i, j)] * self.shift[j];
            }
        }

        result
    }

    pub fn shift_squared(&self) -> f64 {
        let lower = self.lower_shift();

        let mut result = 0.0;

        for i in 0..D {
            result += lower[i] * self.shift[i];
        }

        result
    }

    pub fn spacetime(&self) -> SpacetimeMetric<D> {
        let shift_lower = self.lower_shift();

        SpacetimeMetric {
            tt: -self.lapse * self.lapse + self.shift_squared(),
            ti: shift_lower,
            spatial: self.spatial,
        }
    }
}
