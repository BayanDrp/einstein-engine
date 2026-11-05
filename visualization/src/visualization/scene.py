"""Generic playback scene: one result folder -> one synchronized video.

Every responsibility lands here last on purpose: io found the tracks,
mapping decided where they appear, factory decided what they look like.
This file only wires a shared playback clock to the bodies, draws their
trails, and renders frames. No physics is computed -- the engine owns
the data, this file owns the pixels.

Playback is one single play() of a ValueTracker from the folder's first
recorded time to its last: every body reads the same clock at every
rendered frame, so objects stay time-synchronized no matter how
differently sampled they are (a 3695-sample ISCO ring beside a
single-sample observer). Bodies whose recording starts later hold their
first sample until their data begins (mapping's clamped interpolation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import (
    DEGREES,
    DL,
    ThreeDAxes,
    ThreeDScene,
    Text,
    TracedPath,
    ValueTracker,
    rate_functions,
)

from .factory import track_mobject
from .io import load_folder
from .mapping import fit_to_screen, sample_at

# duration policy: one physical time unit plays at ~1/6 scene second,
# clamped so tiny spans are not invisible and huge spans do not bloat
# the render; at the lowest quality (15 fps) durations stay <= 450 frames
_MIN_SECONDS = 4.0
_MAX_SECONDS = 30.0
_SPAN_TO_SECONDS = 6.0


def playback_duration(span: float) -> float:
    """Folder time span -> scene seconds, clamped to a sane range."""
    return float(min(_MAX_SECONDS, max(_MIN_SECONDS, span / _SPAN_TO_SECONDS)))


def make_scene(folder: str | Path, trails: bool = True) -> type[ThreeDScene]:
    """Build the Scene class for ``folder``.

    Returns the class itself (manim renders it like any scene file);
    constructing it performs the io/mapping/factory work immediately, so a
    folder that yields no trajectories fails fast with a clear error.
    """
    folder = Path(folder)
    tracks, skipped = load_folder(folder)
    if not tracks:
        message = f"{folder}: no trajectory CSVs found"
        if skipped:
            message += f" (skipped: {', '.join(skipped)})"
        raise ValueError(message)

    frame = fit_to_screen(tracks)
    visible = [track for track in tracks if track.visible]
    if not visible:
        raise ValueError(f"{folder}: every track is invisible")

    all_times = np.concatenate([track.times for track in visible])
    t_start = float(np.min(all_times))
    t_end = float(np.max(all_times))
    duration = playback_duration(t_end - t_start)

    class EinsteinEngineScene(ThreeDScene):
        def construct(self):
            self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)

            # furniture, not content: keep orientation cues, drop the clutter
            axes = ThreeDAxes(
                x_range=(-4.5, 4.5, 1),
                y_range=(-4.5, 4.5, 1),
                z_range=(-4.5, 4.5, 1),
                axis_config={"include_ticks": False, "include_tip": False},
            )
            axes.set_stroke(opacity=0.35)
            self.add(axes)

            title = Text(f"Einstein Engine — {folder.name}", font_size=22)
            title.set_opacity(0.6)
            title.to_corner(DL, buff=0.3)
            self.add_fixed_in_frame_mobjects(title)

            clock = ValueTracker(t_start)
            for index, track in enumerate(visible):
                body = track_mobject(track, index)
                body.add_updater(
                    lambda mob, t=track: mob.move_to(sample_at(t, clock.get_value(), frame))
                )
                self.add(body)
                if trails:
                    # comet trail: history dissolves so motion stays the hero
                    trail = TracedPath(
                        body.get_center, stroke_width=3.0, dissipating_time=1.5
                    )
                    trail.set_stroke(color=body.get_color())
                    self.add(trail)

            self.play(
                clock.animate.set_value(t_end),
                run_time=duration,
                rate_func=rate_functions.linear,
            )
            self.wait(0.5)

    EinsteinEngineScene.__qualname__ = EinsteinEngineScene.__name__ = f"EinsteinEngineScene_{folder.name}_{len(visible)}tracks"
    return EinsteinEngineScene
