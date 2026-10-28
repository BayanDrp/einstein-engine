import tempfile
import unittest
from pathlib import Path

from visualization import io
from visualization.io import NotATrajectory, TrackIO, load_folder
from visualization.model import ObjectKind


class TestIo(unittest.TestCase):
    def _csv(self, name: str, text: str) -> Path:
        path = self._dir / name
        path.write_text(text)
        return path

    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self._dir = Path(self._ctx.name)

    def tearDown(self):
        self._ctx.cleanup()

    def test_contract_columns_read_via_alias(self):
        path = self._csv("photon.csv", "t,x,y,z,type\n0,0,0,0,particle\n1,1,0,0,particle\n")
        tracks = TrackIO.read_tracks_from_csv(path)
        self.assertEqual([t.name for t in tracks], ["photon"])  # stem fallback
        self.assertEqual(tracks[0].kind, ObjectKind.PARTICLE)

    def test_time_alias_and_identity_join(self):
        path = self._csv(
            "run.csv",
            "particle_id,label,time,x,y,z\n0,orbit,0,0,0,0\n0,orbit,1,1,0,0\n1,orbit,0,5,5,5\n1,orbit,1,6,5,5\n",
        )
        tracks = TrackIO.read_tracks_from_csv(path)
        self.assertEqual([t.name for t in tracks], ["orbit-0", "orbit-1"])

    def test_unknown_kind_defaults_to_particle(self):
        path = self._csv("odd.csv", "t,x,y,z,type\n0,0,0,0,photon\n1,1,0,0,photon\n")
        tracks = TrackIO.read_tracks_from_csv(path)
        self.assertEqual(tracks[0].kind, ObjectKind.PARTICLE)

    def test_invalid_display_size_falls_back_to_model_default(self):
        path = self._csv("bad_size.csv", "t,x,y,z,type,display_size\n0,0,0,0,matter,-2\n1,1,0,0,matter,-2\n")
        tracks = TrackIO.read_tracks_from_csv(path)
        self.assertEqual(tracks[0].display_size, 0.08)

    def test_visible_string_parsing(self):
        path = self._csv("vis.csv", "t,x,y,z,type,visible\n0,0,0,0,particle,false\n1,1,0,0,particle,false\n")
        tracks = TrackIO.read_tracks_from_csv(path)
        self.assertFalse(tracks[0].visible)

    def test_diagnostics_file_is_not_a_trajectory(self):
        path = self._csv("coupled_dust.csv", "step,t,dt,min_alpha\n0,0,0.1,1.0\n")
        with self.assertRaises(NotATrajectory):
            TrackIO.read_tracks_from_csv(path)

    def test_partial_trajectory_raises_with_file_name(self):
        path = self._csv("harf.csv", "x,y,z\n0,0,0\n1,0,0\n")
        with self.assertRaises(ValueError) as caught:
            TrackIO.read_tracks_from_csv(path)
        self.assertIn("harf.csv", str(caught.exception))

    def test_non_monotonic_times_name_the_track(self):
        path = self._csv("warp.csv", "t,x,y,z,label\n2,0,0,0,a\n1,1,0,0,a\n0,2,0,0,a\n")
        with self.assertRaises(ValueError) as caught:
            TrackIO.read_tracks_from_csv(path)
        message = str(caught.exception)
        self.assertIn("warp.csv", message)
        self.assertIn("a", message)

    def test_load_folder_skips_diagnostics_and_keeps_tracks(self):
        self._csv("observer.csv", "t,x,y,z,type\n0,1,2,3,particle\n")
        self._csv("dust.csv", "t,rho\n0,1.0\n")
        self._csv("summary.csv", "label,drift\na,1e-3\n")
        tracks, skipped = load_folder(self._dir)
        self.assertEqual([t.name for t in tracks], ["observer"])
        self.assertEqual(skipped, ["dust.csv: diagnostics series (t without coordinates)",
                                   "summary.csv: neither a t/time nor coordinate columns"])

    def test_round_trip_keeps_none_color_as_none(self):
        from visualization.model import Track

        original = Track("ghost", [0.0, 1.0], [[0, 0, 0], [1, 1, 1]])
        path = self._dir / "rt.csv"
        TrackIO.write_tracks_to_csv([original], path)
        back = TrackIO.read_tracks_from_csv(path)
        self.assertEqual(back[0].name, "ghost")
        self.assertIsNone(back[0].color)
        self.assertEqual(back[0].kind, ObjectKind.PARTICLE)


if __name__ == "__main__":
    unittest.main()
