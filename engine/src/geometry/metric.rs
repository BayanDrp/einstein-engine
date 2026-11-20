use crate::geometry::tensor::{invert, Tensor2, Vector};
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

    /// the symmetric spacetime metric as an array; index 0 is time,
    /// indices 1..=D are spatial. N is the spacetime dimension,
    /// so pass N = D + 1 (3 for 2+1, 4 for 3+1).
    pub fn as_matrix<const N: usize>(&self) -> [[f64; N]; N] {
        assert!(
            N == D + 1,
            "spacetime dim N must be D + 1, got N = {N}, D = {D}"
        );
        let mut g = [[0.0f64; N]; N];
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

/// partial time derivative of the spacetime metric components d_t g_ab
/// at one point, assembled from the time derivatives of the 3+1
/// variables. At a fixed spatial point x^i the components
///
///   g_00 = -alpha^2 + gamma_ij beta^i beta^j
///   g_0i = gamma_ij beta^j
///   g_ij = gamma_ij
///
/// depend on t only through alpha(t), beta^i(t) and gamma_ij(t), so the
/// chain rule gives
///
///   d_t g_ij = A_ij
///   d_t g_0i = A_ij beta^j + gamma_ij B^j
///   d_t g_00 = -2 alpha alpha_dot + A_ij beta^i beta^j + 2 gamma_ij beta^i B^j
///
/// with A_ij = d_t gamma_ij, B^i = d_t beta^i and alpha_dot = d_t alpha
/// (`spatial_metric_rhs` and the gauge right-hand sides produce exactly
/// these quantities). N is the spacetime dimension, pass N = D + 1. The
/// result has the same index layout as [`SpacetimeMetric::as_matrix`],
/// index 0 is time, 1..=D are spatial, and it is symmetric provided A_ij
/// is symmetric.
pub fn temporal_metric_derivative<const D: usize, const N: usize>(
    lapse: f64,
    shift: &Vector<D>,
    spatial: &Tensor2<D>,
    dgamma_dt: &Tensor2<D>,
    dshift_dt: &Vector<D>,
    dlapse_dt: f64,
) -> [[f64; N]; N] {
    assert!(
        N == D + 1,
        "spacetime dim N must be D + 1, got N = {N}, D = {D}"
    );

    let mut dg = [[0.0f64; N]; N];

    // spatial block: d_t g_ij = A_ij
    for i in 0..D {
        for j in 0..D {
            dg[1 + i][1 + j] = dgamma_dt[(i, j)];
        }
    }

    // mixed block: d_t g_0i = A_ij beta^j + gamma_ij B^j
    for i in 0..D {
        let mut s = 0.0;
        for j in 0..D {
            s += dgamma_dt[(i, j)] * shift[j] + spatial[(i, j)] * dshift_dt[j];
        }
        dg[1 + i][0] = s;
        dg[0][1 + i] = s;
    }

    // time-time block: d_t g_00 = -2 alpha alpha_dot + A_ij beta^i beta^j
    // + 2 gamma_ij beta^i B^j
    let mut s = -2.0 * lapse * dlapse_dt;
    for i in 0..D {
        for j in 0..D {
            s += dgamma_dt[(i, j)] * shift[i] * shift[j]
                + spatial[(i, j)] * (dshift_dt[i] * shift[j] + shift[i] * dshift_dt[j]);
        }
    }
    dg[0][0] = s;

    dg
}

/// inverse spacetime metric g^{ab} as an array, same index ordering as
/// [`SpacetimeMetric::as_matrix`]. N is the spacetime dimension, so pass
/// N = D + 1. Panics on a singular metric, like [`invert`].
pub fn inverse_spacetime_metric<const D: usize, const N: usize>(
    metric: &SpacetimeMetric<D>,
) -> [[f64; N]; N] {
    invert(&metric.as_matrix::<N>())
}
