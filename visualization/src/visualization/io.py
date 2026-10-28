"""CSV ingest layer: engine result files -> Track objects.

Contract (one file holds one or more objects):

- every object needs a time column -- ``t`` or ``time`` -- and the coordinate
  columns ``x`` ``y`` ``z``; other numeric columns are extra data that travels
  with the object (unused by the model today), text columns may carry object
  identity, and a few columns act as visual hints (see below).
- rows are grouped into objects by whichever identity column the file
  declares, checked in this order: ``name``, ``label``, ``mode``, ``id``,
  ``particle_id``; all present identity columns are joined into the object
  name (``orbit-7``). A file without any identity column is ONE object named
  after the file stem.
- visual hints, read from the first row of each group:
  ``type`` (or ``kind``) = ``particle`` | ``matter``; unknown or missing
  values default to ``particle`` instead of raising; ``display_size`` is the
  visual radius in scene units (invalid/missing -> the model default);
  ``color`` and ``visible`` are optional strings/flags.

`TrackIO.read_tracks_from_csv` raises `NotATrajectory` when the file is not a
trajectory: diagnostics and summary files (few or no coordinate columns) are
of that shape and `load_folder` skips them silently. A file that has the
coordinate columns but no time column is instead a BROKEN trajectory and
raises a plain ValueError naming the file. Samples must appear in time
order (the engine writes them sorted); out-of-order rows raise, naming the
file and the track, because re-ordering would silently rewrite results.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .model import ObjectKind, Track

_TIME_ALIASES = ("t", "time")
_COORDS = ("x", "y", "z")
# first match decides which identity columns group rows, in full
_IDENTITY_ORDER = ("name", "label", "mode", "id", "particle_id")
_KIND_ALIASES = ("type", "kind")

DEFAULT_KIND = ObjectKind.PARTICLE
DEFAULT_DISPLAY_SIZE = None  # let the model default (0.08) apply


class NotATrajectory(ValueError):
    """The file is not a trajectory CSV (no \"t/time\" and no \"x,y,z\")."""


def _first_present(columns: pd.Index, aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in columns:
            return alias
    return None


def _identity_columns(columns: pd.Index) -> list[str]:
    return [alias for alias in _IDENTITY_ORDER if alias in columns]


def _kind_column(columns: pd.Index) -> str | None:
    return _first_present(columns, _KIND_ALIASES)


def _group_name(key, columns: list[str], stem: str) -> str:
    if not isinstance(key, tuple):
        key = (key,)
    parts = []
    for col, value in zip(columns, key):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    return "-".join(parts) if parts else stem


def _pythonize(raw):
    """Normalize pandas scalars (numpy bool_/ints/floats) to plain Python."""
    if isinstance(raw, np.generic):
        return raw.item()
    return raw


def _parse_kind(raw) -> ObjectKind:
    text = str(_pythonize(raw)).strip().lower()
    try:
        return ObjectKind(text)
    except ValueError:
        return DEFAULT_KIND


def _parse_display_size(raw):
    raw = _pythonize(raw)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
        if math.isfinite(value) and value > 0.0:
            return value
    return DEFAULT_DISPLAY_SIZE


def _parse_color(raw) -> str | None:
    if isinstance(raw := _pythonize(raw), str):
        text = raw.strip()
        return text if text else None  # also turns pandas NaN-in-CSV "" into None
    return None


def _parse_visible(raw) -> bool:
    raw = _pythonize(raw)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and not math.isfinite(raw):
            return True
        return bool(raw)
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in ("true", "1", "yes", "on"):
            return True
        if text in ("false", "0", "no", "off"):
            return False
    return True


class TrackIO:
    """Read/write Track objects as CSV. Reader tolerates alias columns."""

    @staticmethod
    def read_tracks_from_csv(file_path: str | Path) -> list[Track]:
        path = Path(file_path)
        df = pd.read_csv(path)
        columns = df.columns

        time_col = _first_present(columns, _TIME_ALIASES)
        missing_coords = [c for c in _COORDS if c not in columns]
        if time_col is None and missing_coords:
            raise NotATrajectory("neither a t/time nor coordinate columns")
        if time_col is None:
            raise ValueError(f"{path}: trajectory has x/y/z but no t/time column")
        if missing_coords:
            # time-series diagnostics (t, alpha, rho, progress bars) are legal
            # results; they are simply not trajectories
            raise NotATrajectory("diagnostics series (t without coordinates)")

        numerics = [time_col, *_COORDS]
        df[numerics] = df[numerics].apply(pd.to_numeric, errors="coerce")
        values = df[numerics].to_numpy()
        bad = np.argwhere(~np.isfinite(values))
        if bad.size:
            row, col = bad[0]
            raise ValueError(
                f"{path}: row {row + 2}: column '{numerics[col]}' is not numeric"
            )

        identity = _identity_columns(columns)
        kind_col = _kind_column(columns)

        tracks: list[Track] = []
        groups = (
            [("all", df)] if not identity else list(df.groupby(identity, dropna=False, sort=True))
        )
        for key, group in groups:
            name = (
                _group_name(key, identity, path.stem)
                if identity
                else _group_name(key, [], path.stem)
            )
            times = group[time_col].to_numpy()
            positions = group[list(_COORDS)].to_numpy()
            display_size = _parse_display_size(group["display_size"].iloc[0]) if "display_size" in group.columns else DEFAULT_DISPLAY_SIZE
            kind = _parse_kind(group[kind_col].iloc[0]) if kind_col else DEFAULT_KIND
            color = _parse_color(group["color"].iloc[0]) if "color" in group.columns else None
            visible = _parse_visible(group["visible"].iloc[0]) if "visible" in group.columns else True
            try:
                track = Track(
                    name=name,
                    times=times,
                    positions=positions,
                    kind=kind,
                    display_size=display_size if display_size is not None else 0.08,
                    color=color,
                    visible=visible,
                )
            except ValueError as err:
                raise ValueError(f"{path}: track '{name}': {err}") from None
            tracks.append(track)
        return tracks

    @staticmethod
    def write_tracks_to_csv(tracks: list[Track], file_path: str | Path):
        data = []
        for track in tracks:
            for time, position in zip(track.times, track.positions):
                data.append(
                    {
                        "name": track.name,
                        "time": time,
                        "x": position[0],
                        "y": position[1],
                        "z": position[2],
                        "kind": track.kind.value,
                        "display_size": track.display_size,
                        "color": track.color,
                        "visible": track.visible,
                    }
                )
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)


def load_folder(folder: str | Path) -> tuple[list[Track], list[str]]:
    """Read every trajectory CSV under ``folder`` (top level, sorted).

    Returns ``(tracks, skipped)``; skipped carries one line per non-trajectory
    CSV so mixed result folders (diagnostics + summary + tracks) just work.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a folder")
    tracks: list[Track] = []
    skipped: list[str] = []
    for path in sorted(folder.glob("*.csv")):
        try:
            tracks.extend(TrackIO.read_tracks_from_csv(path))
        except NotATrajectory as reason:
            skipped.append(f"{path.name}: {reason}")
    return tracks, skipped
