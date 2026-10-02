#[cfg(test)]
mod tests {
    use einstein_engine::geometry::metric::{
        lower_shift, shift_squared, Metric, SpacetimeMetric,
    };
    use einstein_engine::geometry::tensor::{Tensor2, Vector};

    // gamma = [[2,1,0],[1,3,0],[0,0,1]], beta^j = (1,2,3)
    // beta_i = gamma_ij beta^j = (4, 7, 3), beta^2 = 27
    fn gamma() -> Tensor2<3> {
        Tensor2::new([[2.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 1.0]])
    }

    #[test]
    fn lower_shift_lowers_with_spatial_metric() {
        let beta = Vector::new([1.0, 2.0, 3.0]);
        let lower = lower_shift(&gamma(), &beta);
        assert_eq!(lower[0], 4.0);
        assert_eq!(lower[1], 7.0);
        assert_eq!(lower[2], 3.0);
    }

    #[test]
    fn shift_squared_is_dot_of_lower_and_upper() {
        // beta_i beta^i = 4*1 + 7*2 + 3*3 = 27
        assert_eq!(shift_squared(&gamma(), &Vector::new([1.0, 2.0, 3.0])), 27.0);
    }

    #[test]
    fn metric_lowers_and_squares() {
        let m = Metric::new(gamma(), 2.0, Vector::new([1.0, 2.0, 3.0]));
        let lower = m.lower_shift();
        assert_eq!(lower[0], 4.0);
        assert_eq!(lower[1], 7.0);
        assert_eq!(m.shift_squared(), 27.0);
    }

    #[test]
    fn metric_rejects_nonpositive_lapse() {
        let spatial = Tensor2::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let shift = Vector::zero();
        assert!(std::panic::catch_unwind(|| {
            let _ = Metric::new(spatial, 0.0, shift);
        })
        .is_err());
    }

    #[test]
    fn spacetime_assembles_4_metric() {
        let g = SpacetimeMetric::from_3plus1(2.0, &Vector::new([1.0, 2.0, 3.0]), &gamma());
        // g_tt = -alpha^2 + beta^2 = -4 + 27 = 23
        assert_eq!(g.tt, 23.0);
        assert_eq!(g.ti[0], 4.0);
        assert_eq!(g.ti[1], 7.0);
        assert_eq!(g.ti[2], 3.0);
        assert_eq!(g.spatial, gamma());
    }

    #[test]
    fn metric_spacetime_matches_free_function() {
        let m = Metric::new(gamma(), 2.0, Vector::new([1.0, 2.0, 3.0]));
        let f = SpacetimeMetric::from_3plus1(2.0, &Vector::new([1.0, 2.0, 3.0]), &gamma());
        assert_eq!(m.spacetime(), f);
    }

    #[test]
    fn minkowski_is_lorentzian() {
        let flat = Tensor2::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let g = SpacetimeMetric::from_3plus1(1.0, &Vector::zero(), &flat);
        assert_eq!(g.tt, -1.0);
        assert_eq!(g.ti[0], 0.0);
        assert_eq!(g.spatial, flat);
    }
}