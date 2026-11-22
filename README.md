# shape-compare

Find which shape in a candidate set a target silhouette most resembles —
**invariant to scale and rotation**, and **tolerant to moderate stretch**
(anisotropic scaling within set parameters). Optionally treat mirror images as a
match.

The tool is **generic**: it compares one target polygon against a *directory* of
candidate polygons and ranks them. The candidate set is just data you supply —
any set of closed outlines works, and the set is swappable without touching the
code. How you produce the candidate polygons is up to you; the tool only ever
reads the JSON format described below.

## How it works

Matching is built on **closed-form invariant descriptors**, so there is no
iterative alignment and therefore no local minimum to get stuck in:

- **Elliptic Fourier Descriptors** (`pyefd`, `normalize=True`) remove
  translation, scale, rotation, and start-point analytically. The distance is
  additionally made invariant to the 180°-phase sign ambiguity by comparing a
  coefficient set against both itself and its negation and taking the minimum.
- **Hu moments** (from the filled silhouette) act as a light, independent
  tie-breaker (default weight 10%).
- **Stretch** is absorbed by **enumerating** a small discrete set of aspect
  ratios and keeping the global best — enumeration, never fragile optimization.
- **Mirror** is handled by **enumerating** a precomputed mirrored descriptor,
  gated behind the `--mirror` flag (default: a mirror image is *not* a match).
- An optional **Shape Context** re-rank (`--shape-context`) refines the top
  survivors using an **exhaustive rotation sweep** (global minimum), kept off the
  full set because it is expensive.

Candidate descriptors are computed in **parallel** and **cached** on disk, so
repeated queries against the same directory are fast.

## Input format

There is exactly **one** input format, used for both the target and every
candidate: a JSON file holding a single closed polygon (outer ring only, no
holes).

```json
{ "contour": [[x0, y0], [x1, y1], [x2, y2], "..."] }
```

- At least 3 numeric `[x, y]` pairs.
- The app does **no** format conversion and accepts no other format — you adapt
  your data to this contract. It was chosen over SVG because it is already an
  ordered point list (no curve-flattening or path-selection ambiguity) and is
  trivial and safe to parse with the standard library (no XML/XXE surface).

## Installation

No host-Python installs — use a virtual environment.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .[dev]     # Windows
# source .venv/bin/activate && pip install -e .[dev]   # POSIX
```

Core runtime needs only `numpy`, `opencv-python-headless`, `pillow`, `pyefd`,
and `scipy`.

### Docker

A minimal, non-root multi-stage image is provided. Your local `./data` is
mounted into the container; nothing is baked into the image.

```bash
docker compose build
docker compose run --rm shape-compare \
    data/targets/outline.json data/candidates --top-k 5
```

Or with plain Docker:

```bash
docker build -t shape-compare .
docker run --rm -v "$PWD/data:/work/data" shape-compare \
    data/targets/outline.json data/candidates --top-k 5
```

## Usage

```bash
shape-compare TARGET.json CANDIDATES_DIR [options]
```

Example:

```bash
shape-compare data/targets/outline.json data/candidates --top-k 5
```

Output is a ranked table by default:

```
rank  candidate                        score  mir  aspect
--------------------------------------------------------
   1  candidate_017.json              0.9641   no    1.10
   2  candidate_042.json              0.9123   no    1.00
   ...
```

### Key options

| Option | Default | Meaning |
| --- | --- | --- |
| `--mirror` | off | Allow a mirror image to count as a match. |
| `--harmonics N` | 20 | Number of EFD harmonics. |
| `--points N` | 128 | Contour resample count. |
| `--aspect-ratios a,b,c` | `0.8,0.9,1.0,1.1,1.25` | Stretch ratios to enumerate. |
| `--shape-context` | off | Enable the Shape Context re-rank. |
| `--sc-steps N` | 36 | Rotation steps in the SC sweep. |
| `--stage1-keep N` | 50 | Survivors passed to the SC re-rank. |
| `--top-k N` | 20 | Rows to report. |
| `--workers N` | all cores | Processes (`1` = serial). |
| `--cache-dir DIR` | `CANDIDATES_DIR/.cache` | Descriptor cache location. |
| `--no-cache` / `--rebuild-cache` | — | Disable or clear the cache. |
| `--weights efd=..,hu=..` | `efd=0.9,hu=0.1` | Distance combination weights. |
| `--recursive` | off | Recurse into candidate subdirectories. |
| `--format {table,json}` | table | Output format. |
| `--json PATH` | — | Also write results to a JSON file. |

Exit codes: `0` success, `1` no results, `2` bad/missing input.

## Producing candidate polygons

The tool does not generate candidates — you supply a directory of JSON polygons
in the format above. Produce them however suits your domain (export from a
drawing, trace an image, dump coordinates from a dataset). Each file must contain
exactly one closed outer ring; drop one file per candidate into a directory and
point `shape-compare` at it.

## Project layout

```
src/shapematch/
  types.py         # frozen value objects: Shape, Descriptor, MatchResult
  config.py        # MatchParams, defaults, cache key
  loader.py        # load_polygon(path) -> Shape  (the single input path)
  geometry.py      # mask/contour, PCA normalize, resample, aspect, mirror
  descriptors/     # efd, hu, shape_context + compute_descriptor
  matching.py      # pairwise_score over aspect/mirror variants
  scoring.py       # robust set-relative normalization + weighting
  engine.py        # RankingEngine: two-stage parallel pipeline
  parallel.py      # ParallelExecutor (multiprocessing; workers=1 = serial)
  cache.py         # DescriptorCache (.npz keyed by file + params hash)
  cli.py           # argparse entry point
tests/             # pytest; fully synthetic fixtures, no external data
```

## Development

```bash
.venv/Scripts/python -m pytest        # tests + coverage (floor: 90%)
.venv/Scripts/python -m ruff check .  # lint + complexity
.venv/Scripts/python -m mypy src      # strict type check
```

## License

MIT — see [LICENSE](LICENSE).
