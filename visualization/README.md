# Einstein Engine: visualization layer

The display layer for the einstein-engine simulations. It replays the engine's
CSV output as a synchronized 3D video. It computes nothing: if a number moves
on screen, the engine put it there. Interpolation exists only to make playback
smooth between engine samples.

## Quickstart

The bounce `python main.py data/` works from inside this directory; from the
repository root, pass the deeper path. The shim runs under any plain python,
and manim does its render inside the uv venv.

    cd visualization
    python main.py data/test2 --dry-run
    python main.py data/test2                    # 720p30, writes output_vid.mp4
    python main.py data/test2 -q h --seconds 12 --out take2

## Flags

| flag | meaning | default |
|------|---------|---------|
| `folder` | results folder to render | `data` |
| `-q` | `l` 480p15, `m` 720p30, `h` 1080p60, `p` 4K | `m` |
| `--no-trails` | render without trail paths | trails on |
| `--seconds N` | override the playback duration | auto: span/6 clamped to [4, 30] |
| `--out NAME` | output video file name | `output_vid` |
| `--dry-run` | print the track inventory, render nothing | render |

The CLI prints the absolute path of the finished video as its last stdout line,
so it is safe to use in scripts. Videos land in
`media/videos/einstein_launcher/<quality>/<out>.mp4` and `media/` is gitignored.
Missing folders, empty folders, and failed renders exit with status 1 and a
one-line error instead of a traceback.

## Folder contract

One CSV file is one object; the file name is the object name unless the CSV
says otherwise. Required columns: `t,x,y,z`. Also accepted: `time` for `t`
and `kind` for `type`. Optional hint columns: `type` (`particle` or `matter`),
`display_size` (screen radius), `color` (any manim color name), `visible`
(`0` hides the object from the fit and the scene).

Files that are not trajectories, such as diagnostics or summaries, are skipped
without crashing. An unfamiliar `type` value falls back to `particle`. A track
with a single sample is valid: it holds its recorded position for the whole
playback. Use `--dry-run` to see exactly what a folder yields.

## Layout

| path | role |
|------|------|
| `src/visualization/io.py` | reads a folder into Track objects with error context |
| `src/visualization/model.py` | the Track and ObjectKind definitions |
| `src/visualization/mapping.py` | one global frame: center, fit, playback sampling |
| `src/visualization/factory.py` | kind to mobject: glow bodies for both kinds |
| `src/visualization/scene.py` | the playback scene: clock, trails, legend, camera |
| `src/visualization/cli.py` + `main.py` | the command line |
| `data/` | example folders and test fixtures |

## Development

    uv sync            # install the environment
    uv run pytest      # 34 tests across io, mapping, factory, model, cli

The render runs as a subprocess (`uv run manim`) so the CLI itself stays
stdlib-only. Every engine results folder renders with zero configuration: drop
the CSVs into `data/<name>` and play. There is no physics in this layer, so no
test here replaces the engine's own numerical suites.
