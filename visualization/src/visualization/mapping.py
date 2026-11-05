"""Coordinate frame: physical result space -> Manim scene space.

One uniform similarity transform per folder -- translation plus a single
scale -- decided once by `fit_to_screen` and owned by `Frame`; scene.py and
factory.py consume the frame instead of redoing the algebra.

- the bounding box is the union of every visible track's samples, so two
  objects recorded at the same physical point overlap exactly on screen
- the scale is one number applied to all three axes: shapes, angles and
  relative distances survive, only the unit changes
- a degenerate folder (every sample identical, e.g. only observer.csv in a
  result set) is not an error: the frame centers on the point with scale 1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Track


@dataclass
class Frame:
    """The physical -> scene transform decided once per folder."""

    center: np.ndarray  # the physical point that maps to scene origin
    scale: float        # scene units per physical unit (uniform, all axes)

    def to_scene(self, points: np.ndarray) -> np.ndarray:
        """Map physical (…, 3) points to scene coordinates.

        Accepts a single (3,) point or an (N, 3) array; returns the same
        shape. Component-wise translate + uniform scale, nothing else.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim not in (1, 2) or pts.shape[-1] != 3:
            raise ValueError(f"expected shape (…, 3), got {pts.shape}")
        mapped = (pts - self.center) * self.scale
        return mapped


def fit_to_screen(tracks: list[Track], target_width: float = 7.6) -> Frame:
    """Build the one `Frame` that fits every visible track in view.

    The widest axis extent maps to ``±target_width / 2`` scene units;
    `target_width = 7.6` leaves margins in the default Manim camera for
    titles and labels. Tracks with ``visible=False`` are contentless for
    playback and do not steer the fit.
    """
    visible = [track for track in tracks if track.visible]
    if not visible:
        raise ValueError("no visible tracks provided")

    all_positions = np.concatenate([track.positions for track in visible], axis=0)
    min_coords = np.min(all_positions, axis=0)
    max_coords = np.max(all_positions, axis=0)
    spans = max_coords - min_coords
    widest = float(np.max(spans))

    if widest <= 0.0:
        # every recorded sample is the same physical point: zoom-to-fit would
        # divide by zero, so degrade to a centered unit-scale frame instead
        return Frame(center=all_positions[0].copy(), scale=1.0)

    center = (min_coords + max_coords) / 2
    scale = target_width / widest
    return Frame(center=center, scale=scale)


def sample_at(track: Track, times: np.ndarray, frame: Frame) -> np.ndarray:
    """Positions of ``track`` at query ``times`` (any order), in scene coordinates.

    Between two recorded samples the object follows the straight display
    chord (`np.interp` per axis); before its first / after its last sample
    it holds the nearest recorded position. This is display smoothness up to
    30 fps -- no integration, no physics; replay only.
    """
    query = np.asarray(times, dtype=float).ravel()
    if query.size == 0:
        return np.empty((0, 3))
    sampled = np.stack(
        [np.interp(query, track.times, track.positions[:, axis]) for axis in range(3)],
        axis=1,
    )
    return frame.to_scene(sampled)
