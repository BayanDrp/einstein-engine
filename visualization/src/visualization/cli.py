"""CLI - scan a folder or render it through manim.

This module stays stdlib-only at import time: main.py must run under any
plain python, so pandas/manim imports happen lazily inside the work
functions, and the render itself executes inside the uv venv via a
subprocess (where manim lives).
"""

from __future__ import annotations

import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

QUALITY_MAP = {
    "l": "-ql",
    "m": "-qm",
    "h": "-qh",
    "p": "-qp",
}

# fixed stem so media/videos/ holds one directory per install, not per run
_LAUNCHER_STEM = "einstein_launcher"


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(description="Render a folder of trajectory CSVs.")
    parser.add_argument(
        "folder", nargs="?", type=Path, default=Path("data"),
        help="Folder containing trajectory CSVs (default: data)",
    )
    parser.add_argument(
        "-q", "--quality", choices=list(QUALITY_MAP), default="m",
        help="Render quality (l=480p15, m=720p30, h=1080p60, p=4K)",
    )
    parser.add_argument(
        "--no-trails", action="store_true",
        help="Disable trails (traced paths) for all tracks",
    )
    parser.add_argument(
        "--seconds", type=float, default=None,
        help="Override the playback duration in seconds (default: auto from the data span)",
    )
    parser.add_argument(
        "--out", default="output_vid",
        help="Output video file name (default: output_vid)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the tracks the folder yields, then exit without rendering",
    )
    return parser.parse_args(argv)


def launcher_text(folder: Path, trails: bool, seconds: float | None) -> str:
    """The scene-discovery contract: a class defined in the rendered module."""
    options = f", trails={'True' if trails else 'False'}"
    if seconds is not None:
        options += f", seconds={seconds}"
    return (
        "from visualization.scene import make_scene\n"
        f"class Scene(make_scene({str(folder)!r}{options})):\n"
        "    pass\n"
    )


def find_newest_video(media_dir: Path) -> Path | None:
    """Newest finished mp4 under our launcher's part of media/, partials excluded."""
    candidates = [
        path
        for path in (media_dir / "videos" / _LAUNCHER_STEM).rglob("*.mp4")
        if "partial_movie_files" not in path.parts
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _project_root() -> Path:
    """visualization/ project directory (src/visualization/cli.py -> parents[2])."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return Path.cwd()


def _resolve_folder(raw: Path) -> Path | None:
    folder = raw.resolve()
    if not folder.is_dir():
        print(f"error: folder not found: {raw}", file=sys.stderr)
        return None
    return folder


def run_dry_run(args: Namespace) -> int:
    folder = _resolve_folder(args.folder)
    if folder is None:
        return 1
    from .io import load_folder  # lazy: pandas is a heavy import

    tracks, skipped = load_folder(folder)
    if not tracks:
        print(f"no trajectory CSVs found in {folder}")
        if skipped:
            print(f"skipped: {', '.join(skipped)}")
        return 1
    print(f"{folder}: {len(tracks)} track(s), {len(skipped)} skipped file(s)")
    for track in tracks:
        start, end = track.times[0], track.times[-1]
        print(
            f"  {track.name:<14} {track.kind.value:<8} {len(track.times):>6} samples"
            f"  t {start:>8}..{end:<8} display={track.display_size:.2f}  visible={track.visible}"
        )
    if skipped:
        print(f"skipped: {', '.join(skipped)}")
    return 0


def run_render(args: Namespace) -> int:
    folder = _resolve_folder(args.folder)
    if folder is None:
        return 1

    root = _project_root()
    media_dir = root / "media"
    launcher = media_dir / "launcher" / f"{_LAUNCHER_STEM}.py"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text(
        launcher_text(folder, trails=not args.no_trails, seconds=args.seconds),
        encoding="utf-8",
    )

    command = [
        "uv", "run", "manim",
        QUALITY_MAP[args.quality],
        "--media_dir", str(media_dir),
        str(launcher),
        "Scene",
        "-o", args.out,
    ]
    try:
        completed = subprocess.run(command, cwd=root)
    except FileNotFoundError:
        print("error: uv not found - run from a machine with uv (or the project venv)", file=sys.stderr)
        return 1
    if completed.returncode != 0:
        print("error: manim render failed (output above)", file=sys.stderr)
        return 1

    video = find_newest_video(media_dir)
    if video is None:
        print("error: manim reported success but produced no mp4", file=sys.stderr)
        return 1
    print(video.resolve())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        return run_dry_run(args)
    return run_render(args)
