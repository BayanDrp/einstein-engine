use crate::geometry::tensor::{Tensor2, Vector};
use crate::grid::Field;

// the 3+1 split of the spacetime metric, all in one place.
// lapse, shift, extrinsic curvature and the spatial slice metric
// are fields; the metric structs assemble them per point.

/// the lapse alpha: one positive scalar per grid point
pub type LapseField<const D: usize> = Field<f64, D>;

/// the shift vector beta^i: one vector per grid point
pub type ShiftField<const D: usize> = Field<Vector<D>, D>;

/// the extrinsic curvature K_ij: one symmetric tensor per grid point
pub type ExtrinsicCurvature<const D: usize> = Field<Tensor2<D>, D>;

/// the spatial 3-metric gamma_ij on the slice: one tensor per grid point
pub type SliceMetric<const D: usize> = Field<Tensor2<D>, D>;

/// lower the shift with the spatial metric: beta_i = gamma_ij beta^j
pub fn lower_shift<const D: usize>(spatial: &Tensor2<D>, shift: &Vector<D>) -> Vector<D> {
    let mut out = Vector::zero();
    for i in 0..D {
        for j in 0..D {
            out[i] += spatial[(i, j)] * shift[j];
        }
    }
    out
}

/// the shift norm: beta^2 = beta_i beta^i
pub fn shift_squared<const D: usize>(spatial: &Tensor2<D>, shift: &Vector<D>) -> f64 {
    let lower = lower_shift(spatial, shift);
    let mut out = 0.0;
    for i in 0..D {
        out += lower[i] * shift[i];
    }
    out
}

/// the one-point 3+1 data: metric of the spatial slice
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metric<const D: usize> {
    pub spatial: Tensor2<D>,
    pub lapse: f64,
    pub shift: Vector<D>,
}

impl<const D: usize> Metric<D> {
    pub fn new(spatial: Tensor2<D>, lapse: f64, shift: Vector<D>) -> Self {
        assert!(lapse.is_finite() && lapse > 0.0);

        Self {
            spatial,
            lapse,
            shift,
        }
    }

    pub fn lower_shift(&self) -> Vector<D> {
        lower_shift(&self.spatial, &self.shift)
    }

    pub fn shift_squared(&self) -> f64 {
        shift_squared(&self.spatial, &self.shift)
    }

    pub fn spacetime(&self) -> SpacetimeMetric<D> {
        SpacetimeMetric::from_3plus1(self.lapse, &self.shift, &self.spatial)
    }
}

pub type MetricField<const D: usize> = Field<Metric<D>, D>;

/// the 4-metric components g_ab from the 3+1 data at one point
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpacetimeMetric<const D: usize> {
    pub tt: f64,
    pub ti: Vector<D>,
    pub spatial: Tensor2<D>,
}

impl<const D: usize> SpacetimeMetric<D> {
    /// build from lapse alpha, shift beta^j and the spatial metric gamma_ij
    pub fn from_3plus1(lapse: f64, shift: &Vector<D>, spatial: &Tensor2<D>) -> Self {
        Self {
            tt: -lapse * lapse + shift_squared(spatial, shift),
            ti: lower_shift(spatial, shift),
            spatial: *spatial,
        }
    }

    /// the symmetric 4-metric as a 4x4 array; index 0 is time,
    /// indices 1..=D are spatial
    pub fn as_4x4(&self) -> [[f64; 4]; 4] {
        let mut g = [[0.0f64; 4]; 4];
        g[0][0] = self.tt;
        for i in 1..=D {
            g[0][i] = self.ti[i - 1];
            g[i][0] = self.ti[i - 1];
        }
        for i in 1..=D {
            for j in 1..=D {
                g[i][j] = self.spatial[(i - 1, j - 1)];
            }
        }
        g
    }
}