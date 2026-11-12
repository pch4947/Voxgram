# Contributing

This is the day-to-day workflow for the Voxgram team. Branch, commit and pull
request rules are in [docs/GIT_CONVENTION.md](docs/GIT_CONVENTION.md).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Python 3.10 or newer. In Visual Studio or VS Code, select `.venv` as the
interpreter.

## Workflow

1. Pick or open an issue and assign yourself.
2. Branch from an up-to-date `main`: `git switch -c feat/12-marker-detection`.
3. Commit in small steps using the commit convention.
4. Run the checks locally (below).
5. Open a pull request and link the issue. Ask a teammate to look at it if you
   want a second opinion; approval is not required.
6. Once CI is green, squash and merge, then delete the branch.

## Checks

CI runs these on every pull request. Run them before pushing.

```powershell
ruff check .
ruff format --check .
python tools/check_library_boundary.py
pytest
```

`ruff format .` fixes formatting. `ruff check --fix .` fixes most lint errors.

## Code rules

### Implement the image processing yourself

This is a course requirement, not a style preference. See the
[library boundary](README.md#library-boundary).

- OpenCV is for reading and writing image files and nothing else. Allowed:
  `cv2.imread`, `cv2.imwrite`, `cv2.imdecode`, `cv2.imencode` and their
  `IMREAD_*` / `IMWRITE_*` flags.
- NumPy is for arrays, arithmetic and solving linear systems
  (`np.linalg.solve`, `np.linalg.svd`, `np.linalg.lstsq`).
- Librosa is for the audio side only: the STFT and Griffin-Lim, always with
  the settings in `voxgram.format`, plus resampling a recording to 16 kHz when
  it is loaded. Call it through `voxgram.audio_backend`, not directly.
- Read and write image files through `voxgram.imageio`, which is the only
  module that imports OpenCV.
- Do not import `skimage` or `scipy.ndimage`. The report does not name them,
  but they provide the same ready-made routines that it excludes.

`tools/check_library_boundary.py` checks the OpenCV and import rules. If you
believe a call should be allowed, raise it with the team before changing the
allowlist.

`voxgram.imageio` reads and writes through NumPy buffers, because on Windows
`cv2.imread` can return `None` without an error for paths that contain
non-ASCII characters, such as Korean folder names.

### Use the shared format constants

Import sample rate, FFT size, grid shape and dynamic range from
`voxgram.format`. Do not write `16000`, `512`, `257` or `376` as literals in
module code. A change to the format follows the
[format change rule](docs/GIT_CONVENTION.md#format-changes).

### Keep grayscale values intact

The binary image exists only for marker detection. Rectification, calibration
and extraction work on the grayscale image, in floating point, and round only
at the end.

### Fail clearly

If markers are missing or ambiguous, or the quadrilateral is degenerate, raise
an error that says so. The decoder must never turn a bad detection into
plausible-sounding audio.

### Style

- Formatting and linting are handled by Ruff. Line length is 100.
- Type hints on public functions.
- Docstrings state array shapes, dtypes and value ranges, for example
  `(H, W) float64 in [0, 255]`.
- Prefer vectorized NumPy over Python loops over pixels.

## Tests

- Tests live in `tests/` and mirror the module names: `tests/test_geometry.py`.
- Use synthetic inputs where you can: a known homography applied to a known
  image, a synthetic card with known patch values.
- Small fixtures go in `tests/data/`.

## Experiments

- Scripts and records go in `experiments/`. Commit them with the `exp` type.
- Each record states the card and format version, printer, paper, print
  settings, phone, capture conditions, and the commit the results came from.
- Raw photographs and recordings stay out of the repository. Reference where
  they are stored.
