# Einstein Engine: visualization layer

The display layer of the einstein engine simulations. It replays the engine's
CSV output as a synchronized 3D video. It computes nothing: if a number moves
on screen, the engine put it there. Interpolation exists only to make playback
smooth between engine samples.

## Status

The manim based Python renderer was removed. The renderer is being rebuilt on
three.js (WebGL) with headless Chrome frame capture, and the file skeleton
below already exists with empty modules. This README documents the target
state and, because it is already final, the data contract.

## Structure

```
visualization/
├── web/
│   ├── index.html            # entry page: WebGL canvas, DOM overlay (HUD, legend, title), dev controls
│   └── src/
│       ├── ingest.js         # CSV load and normalization (replaces io.py)
│       ├── layout.js         # screen fitting and time sampling (replaces mapping.py)
│       ├── bodies.js         # body visuals and palette (replaces factory.py)
│       ├── staging.js        # camera drift, starfield, trails, playback clock (replaces scene.py)
│       └── main.js           # bootstrap: ingest, layout, build the scene, play
├── bin/
│   ├── render.mjs            # headless render command (replaces main.py and cli.py)
│   └── dev.mjs               # serves the page for interactive visual tuning
├── test/
│   ├── ingest.test.mjs       # runs on node's built in test runner, no extra dependencies
│   └── layout.test.mjs       # pure math tests for the screen frame
├── data/
└── README.md
```

Implementation work adds `package.json` (three.js and puppeteer support) and
`.gitignore` entries for `node_modules` and capture workdirs.

## Data contract

One folder under `data/` renders as one synchronized video; each CSV holds one
or more bodies.

* Required columns: `t` (or `time`), `x`, `y`, `z`
* Naming columns: `label`, `mode`, or `name` group rows into bodies named
  `<value>-<k>`; k counts the bodies that share the value inside the file.
  Without a naming column the whole file is one body named after the file stem
* Hint columns: `type` (`particle` or `matter`; unknown values fall back to
  `particle`), `display_size` (body radius in world units), `color` (free
  text color hint), `visible` (`0`, `false`, or `no` hides the body)

Files without the required columns, such as summary tables, are skipped with a
reason instead of failing the run. A track with a single sample is valid: it
holds its recorded position for the whole playback.

Example folders: `data/schwarzschild` (a horizon body, a photon sphere orbit,
an isco orbit, and staging markers from the engine example) and `data/test2`
(a hand made five body sanity set).

## Rendering pipeline

* `bin/render.mjs` launches headless Chrome, steps the scene with a
  deterministic frame counter (no wall clock anywhere), pipes rendered frames
  into ffmpeg, and prints the absolute path of the finished MP4 as its last
  stdout line
* `bin/dev.mjs` serves the same page locally so the scene can be tuned
  interactively in a normal browser
* Glow is native to the web stack instead of layered translucent shells:
  particles are additive luminous sprites with a small halo, matter is one
  solid sphere per body, and a matter body with a dark color hint (black or
  grey) gets an occluding black core with one thin luminous ring, the way
  black holes photograph
* The stage includes a slow camera drift, a seeded starfield, fading comet
  trails, a legend built from the real body names, and a live engine time
  readout

## Command line

    node bin/render.mjs data/schwarzschild -q m --seconds 14 --out schwarzschild

Target flags for parity with the tool this replaces; they go live together
with `bin/render.mjs`:

* `-q l|m|h|p`: render quality, `l` (480p15), `m` (720p30), `h` (1080p60),
  `p` (4K); default `m`
* `--seconds N`: override the playback duration; the default is the data span
  divided by three, clamped to 6 to 30 seconds
* `--no-trails`: render without trail lines
* `--no-bloom`: skip the glow pass
* `--out NAME`: output video name; default `output_vid`
* `--dry-run`: print the track inventory and render nothing

Missing folders, empty folders, and failed renders exit with status 1 and a
single line error instead of a traceback.

## Development

The landing order is ingest, layout, bodies, staging, the capture driver, and
then the first render review of `data/schwarzschild`. Display only principle:
there is no physics in this layer, so no test here replaces the engine's own
numerical suites.
