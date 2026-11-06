import tempfile
import time
import unittest
from pathlib import Path

from visualization.cli import (
    QUALITY_MAP,
    find_newest_video,
    launcher_text,
    parse_args,
)


class TestParseArgs(unittest.TestCase):
    def test_defaults_match_the_locked_spec(self):
        args = parse_args([])
        self.assertEqual(args.folder, Path("data"))
        self.assertEqual(args.quality, "m")
        self.assertFalse(args.no_trails)
        self.assertIsNone(args.seconds)
        self.assertEqual(args.out, "output_vid")
        self.assertFalse(args.dry_run)

    def test_quality_maps_to_manim_flags(self):
        self.assertEqual(
            QUALITY_MAP, {"l": "-ql", "m": "-qm", "h": "-qh", "p": "-qp"}
        )

    def test_explicit_flags_override(self):
        args = parse_args(["somewhere", "-q", "h", "--no-trails",
                           "--seconds", "12.5", "--out", "take2", "--dry-run"])
        self.assertEqual(args.folder, Path("somewhere"))
        self.assertEqual(args.quality, "h")
        self.assertTrue(args.no_trails)
        self.assertEqual(args.seconds, 12.5)
        self.assertEqual(args.out, "take2")
        self.assertTrue(args.dry_run)


class TestLauncherText(unittest.TestCase):
    def test_defines_a_discoverable_scene_subclass(self):
        text = launcher_text(Path("/x/y"), trails=False, seconds=7.0)
        self.assertIn("from visualization.scene import make_scene", text)
        self.assertIn("class Scene(make_scene('/x/y'", text)
        self.assertIn("trails=False", text)
        self.assertIn("seconds=7.0", text)
        self.assertIn("    pass", text)

    def test_no_seconds_when_auto(self):
        self.assertNotIn("seconds", launcher_text(Path("/x/y"), True, None))


class TestFindNewestVideo(unittest.TestCase):
    def test_returns_newest_finished_mp4_and_skips_partials(self):
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp)
            final_dir = media / "videos" / "einstein_launcher" / "720p30"
            partials = final_dir / "partial_movie_files"
            partials.mkdir(parents=True)
            time.sleep(0.02)  # mtime resolution
            (partials / "chunk.mp4").write_bytes(b"x")  # must be ignored
            old = final_dir / "output_vid.mp4"
            old.write_bytes(b"x")
            time.sleep(0.02)
            new = final_dir / "output_vid.mp4"  # same name, overwritten later
            new.write_bytes(b"replacement")
            time.sleep(0.02)
            other = media / "videos" / "some_other_launcher" / "keep.mp4"
            other.parent.mkdir(parents=True)
            other.write_bytes(b"x")  # different launcher stem: ignored

            self.assertEqual(find_newest_video(media), new.resolve())

    def test_returns_none_when_nothing_rendered(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(find_newest_video(Path(tmp)))


class TestSecondsOverrideReachesTheScene(unittest.TestCase):
    def test_make_scene_honors_seconds(self):
        from visualization.scene import make_scene

        scene_class = make_scene("data/test_values", trails=False, seconds=9.0)
        self.assertEqual(scene_class.playback_seconds, 9.0)
        auto_class = make_scene("data/test_values", trails=False)
        self.assertEqual(auto_class.playback_seconds, 4.0)  # span/6 clamped to the floor


if __name__ == "__main__":
    unittest.main()
