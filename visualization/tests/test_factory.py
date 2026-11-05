import unittest

import numpy as np
from manim import ORIGIN

from visualization.factory import PALETTE, track_mobject
from visualization.io import load_folder
from visualization.model import ObjectKind, Track


class TestFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tracks, _ = load_folder("data/test_values")
        cls.tracks = {track.name: track for track in tracks}

    def test_fixture_radios_and_kinds_flow_through(self):
        mob = track_mobject(self.tracks["photon"], 0)
        self.assertEqual(type(mob).__name__, "VGroup")
        halo, core = mob[0], mob[1]
        self.assertAlmostEqual(core.radius, 0.08)
        self.assertAlmostEqual(halo.radius, 0.2)  # glow shell: radius × 2.5
        self.assertAlmostEqual(core.get_fill_opacity(), 1.0)
        self.assertAlmostEqual(halo.get_fill_opacity(), 0.2)
        mob = track_mobject(self.tracks["observer"], 1)
        self.assertAlmostEqual(mob[1].radius, 0.12)
        mob = track_mobject(self.tracks["dust_cloud"], 2)
        halo, core = mob[0], mob[1]
        self.assertEqual(type(core).__name__, "Sphere")
        self.assertAlmostEqual(core.radius, 0.35)
        self.assertAlmostEqual(halo.radius, 0.595)  # glow shell: radius × 1.7
        self.assertAlmostEqual(core.get_fill_opacity(), 0.85)
        self.assertAlmostEqual(halo.get_fill_opacity(), 0.12)

    def test_csv_colors_are_honored(self):
        from manim import Dot3D, ORIGIN, Sphere

        photon = track_mobject(self.tracks["photon"], 0)
        reference = Dot3D(point=ORIGIN, radius=0.08, color="yellow")
        self.assertEqual(photon[1].color, reference.color)
        cloud = track_mobject(self.tracks["dust_cloud"], 0)
        cloud_reference = Sphere(radius=0.35, color="orange")
        self.assertEqual(cloud[1].color, cloud_reference.color)

    def test_palette_fallback_is_deterministic_by_index(self):
        plain = Track("plain", times=[0.0, 1.0], positions=[[0, 0, 0], [1, 0, 0]])
        first = str(track_mobject(plain, 0)[1].color)
        again = str(track_mobject(plain, 0)[1].color)
        other = str(track_mobject(plain, 1)[1].color)
        self.assertEqual(first, again)
        self.assertEqual(track_mobject(plain, 0)[1].color, PALETTE[0])
        self.assertNotEqual(first, other)

    def test_unknown_kind_falls_back_to_particle(self):
        from types import SimpleNamespace

        phantom = SimpleNamespace(
            kind=SimpleNamespace(value="photon"),  # a future kind BUILDERS doesn't know
            display_size=0.08,
            color=None,
        )
        mob = track_mobject(phantom, 0)
        self.assertEqual(type(mob).__name__, "VGroup")
        self.assertEqual(type(mob[1]).__name__, "Dot3D")

    def test_bodies_are_built_at_the_scene_origin(self):
        for index, track in enumerate(self.tracks.values()):
            if not track.visible:
                continue
            self.assertTrue(np.allclose(track_mobject(track, index).get_center(), ORIGIN),
                            f"{track.name} was not built at ORIGIN")

    def test_real_enum_kind_values_reach_the_dispatch(self):
        mob = track_mobject(self.tracks["dust_cloud"], 0)
        self.assertEqual(self.tracks["dust_cloud"].kind, ObjectKind.MATTER)
        self.assertEqual(type(mob[1]).__name__, "Sphere")


if __name__ == "__main__":
    unittest.main()
