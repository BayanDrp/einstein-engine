import unittest

import numpy as np

from visualization.mapping import Frame, fit_to_screen, sample_at
from visualization.model import Track


def circle_track(name: str, radius: float, samples: int = 8, visible: bool = True):
    angle = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    positions = np.stack([radius * np.cos(angle), radius * np.sin(angle), np.zeros(samples)], axis=1)
    times = np.linspace(0.0, float(samples - 1), samples)
    return Track(name=name, times=times, positions=positions, visible=visible)


class TestFrame(unittest.TestCase):
    def test_symmetric_data_centers_and_fits_widest_axis(self):
        a = Track("a", [0.0, 1.0], [[-5.0, 0, 0], [5.0, 0, 0]])
        frame = fit_to_screen([a], target_width=7.6)
        self.assertTrue(np.allclose(frame.center, [0.0, 0.0, 0.0]))
        self.assertAlmostEqual(frame.scale, 7.6 / 10.0)
        self.assertTrue(np.allclose(frame.to_scene(np.array([5.0, 0.0, 0.0])), [3.8, 0.0, 0.0]))

    def test_uniform_scale_preserves_the_photon_inside_isco_ratio(self):
        photon = circle_track("photon", 1.87)
        isco = circle_track("isco", 4.95)
        frame = fit_to_screen([photon, isco])
        r_photon = np.abs(frame.to_scene(photon.positions)).max()
        r_isco = np.abs(frame.to_scene(isco.positions)).max()
        self.assertAlmostEqual(r_isco, 3.8, places=6)
        self.assertAlmostEqual(r_photon / r_isco, 1.87 / 4.95, places=6)

    def test_to_scene_shapes(self):
        frame = Frame(center=np.zeros(3), scale=2.0)
        one = frame.to_scene(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(one.shape, (3,))
        many = frame.to_scene(np.zeros((4, 3)))
        self.assertEqual(many.shape, (4, 3))
        with self.assertRaises(ValueError):
            frame.to_scene(np.zeros((2, 2)))

    def test_stationary_only_folder_degrades_to_unit_frame(self):
        observer = Track("observer", [0.0], [[-1.5, 1.0, 0.3]])
        frame = fit_to_screen([observer])
        self.assertEqual(frame.scale, 1.0)
        self.assertTrue(np.allclose(frame.center, [-1.5, 1.0, 0.3]))
        self.assertTrue(np.allclose(frame.to_scene(observer.positions), [[0.0, 0.0, 0.0]]))

    def test_invisible_tracks_do_not_steer_the_fit(self):
        seen = Track("seen", [0.0, 1.0], [[-1.0, 0, 0], [1.0, 0, 0]])
        hidden = Track("ghost", [0.0, 1.0], [[-100.0, 0, 0], [100.0, 0, 0]], visible=False)
        frame = fit_to_screen([seen, hidden])
        self.assertAlmostEqual(frame.scale, 7.6 / 2.0)
        with self.assertRaises(ValueError):
            fit_to_screen([hidden])


class TestSampleAt(unittest.TestCase):
    def setUp(self):
        self.track = Track("a", [0.0, 1.0, 2.0], [[0.0, 0, 0], [1.0, 0, 0], [2.0, 2.0, 0]])
        self.frame = Frame(center=np.zeros(3), scale=1.0)

    def test_exact_at_recorded_times(self):
        at_samples = sample_at(self.track, self.track.times, self.frame)
        self.assertTrue(np.allclose(at_samples, self.track.positions))

    def test_midpoints_take_display_chords(self):
        mid = sample_at(self.track, np.array([0.5, 1.5]), self.frame)
        self.assertTrue(np.allclose(mid, [[0.5, 0, 0], [1.5, 1.0, 0]]))

    def test_outside_range_holds_nearest_sample(self):
        ends = sample_at(self.track, np.array([-2.0, 99.0]), self.frame)
        self.assertTrue(np.allclose(ends, [[0.0, 0, 0], [2.0, 2.0, 0]]))

    def test_empty_query_gives_empty_positions(self):
        empty = sample_at(self.track, np.array([]), self.frame)
        self.assertEqual(empty.shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
