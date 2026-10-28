import unittest
from visualization import model


class TestTrack(unittest.TestCase):
    def test_track_initialization(self):
        times = [0, 1, 2]
        positions = [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
        ]

        track = model.Track(
            name="Test Track",
            times=times,
            positions=positions,
            kind=model.ObjectKind.PARTICLE,
            display_size=0.1,
            color="red",
            visible=True,
        )

        self.assertIsInstance(track, model.Track)
        self.assertEqual(track.name, "Test Track")
        self.assertEqual(track.kind, model.ObjectKind.PARTICLE)
        self.assertEqual(len(track.times), len(times))
        self.assertEqual(track.positions.shape, (3, 3))
        self.assertEqual(track.display_size, 0.1)
        self.assertEqual(track.color, "red")
        self.assertTrue(track.visible)


if __name__ == "__main__":
    unittest.main()