"""Decoder: photograph of a card to a WAV file.

Command line:
    python -m voxgram.decoder photo.jpg -o voice.wav

Pipeline. Every stage is timed and logged:

    read -> grayscale -> detect -> homography -> rectify -> calibrate
         -> extract -> magnitude -> griffin-lim -> write

* ``grayscale`` and ``detect`` come from ``voxgram.detection``.
* ``homography`` and ``rectify`` come from ``voxgram.geometry``.
* ``calibrate`` comes from ``voxgram.photometric``.
* ``extract`` and ``magnitude`` come from ``voxgram.extraction`` and
  ``voxgram.intensity``; ``griffin-lim`` and ``write`` from
  ``voxgram.audio_backend``.

Canonical card image. ``rectify`` and ``calibrate`` produce the card as an image
of ``print_shape(card_dpi)`` pixels, with the pixel to millimetre convention of
``voxgram.card.print_matrix``. The data region is then the draft layout of
``voxgram.format`` times the pixels per millimetre.

Unmeasurable cells. ``calibrate`` returns NaN for samples it cannot trust. A
sample that is NaN only because the raw photograph is clipped at black or at
white is given that clipped value, 0 or 255: it is as dark or as light as the
camera can record. Any other NaN is unmeasurable (low contrast, or outside the
photograph) and counts as silence. If more than ``max_unmeasurable`` of the
data cells are unmeasurable, decoding stops instead of inventing audio.

Switches for the ablation (``DecodeOptions.skip``, ``--skip`` on the command
line):

* ``detect``: no markers, no correction. The photograph is taken to be an
  undistorted card, the whole card and nothing else. This also skips
  ``homography``, ``rectify`` and ``calibrate``.
* ``rectify``: geometry off. The grayscale photograph is used as the card.
  Photometric calibration samples through the homography, so it is skipped too.
* ``calibrate``: photometric calibration off; the rectified card is used.

Failure. A stage reports a failure with ``ValueError``. The decoder turns it into
a :class:`DecodeError` that names the stage, the command line prints it, and no
audio file is written.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from voxgram.audio_backend import GRIFFIN_LIM_ITERATIONS, griffin_lim, save_audio
from voxgram.card import MM_PER_INCH, print_matrix, print_shape
from voxgram.detection import (
    DEFAULT_LOCAL_OFFSET,
    default_window,
    detect_markers,
    otsu_threshold,
    threshold_global,
    threshold_local_mean,
    to_grayscale,
)
from voxgram.extraction import (
    DEFAULT_CELL_FRACTION,
    DataRegion,
    binomial_kernel,
    extract_gray,
)
from voxgram.format import (
    CARD_HEIGHT_MM,
    CARD_WIDTH_MM,
    DATA_HEIGHT_MM,
    DATA_WIDTH_MM,
    DATA_X_MM,
    DATA_Y_MM,
    GRAY_BLACK,
    GRAY_WHITE,
    MARKER_CENTERS_MM,
    TARGET_PRINT_DPI,
)
from voxgram.geometry import estimate_homography, warp_to_canonical
from voxgram.imageio import read_image, write_image
from voxgram.intensity import gray_to_magnitude, image_to_grid
from voxgram.photometric import calibrate_samples

log = logging.getLogger("voxgram.decoder")

STAGES = (
    "read",
    "grayscale",
    "detect",
    "homography",
    "rectify",
    "calibrate",
    "extract",
    "magnitude",
    "griffin-lim",
    "write",
)
SKIPPABLE_STAGES = ("detect", "rectify", "calibrate")
THRESHOLD_METHODS = ("local", "global")

# Resolution of the canonical card image. 300 dpi gives about 3.4 pixels per
# cell. At TARGET_PRINT_DPI the peak memory is about 2 GB instead of 0.6 GB.
DEFAULT_CARD_DPI = 300.0

# Share of the data cells that may be unmeasurable before decoding stops.
# Proposed value, to be confirmed on real photographs.
MAX_UNMEASURABLE_FRACTION = 0.25

# Raw photograph values at or beyond these are treated as clipped. They are the
# thresholds given to ``calibrate_samples``, and the same ones classify its NaN.
SATURATION_LOW = 0.0
SATURATION_HIGH = 255.0

# Value written for an unmeasurable pixel in an intermediate image.
INVALID_PIXEL = 128

# Intermediate images, numbered in pipeline order.
INTERMEDIATE_FILES = {
    "grayscale": "1_grayscale.png",
    "binary": "2_binary.png",
    "rectified": "3_rectified.png",
    "corrected": "4_corrected.png",
    "spectrogram": "5_spectrogram.png",
}


class DecodeError(Exception):
    """A decoding failure. The message is meant for the person running the command."""


def _default_detect(gray: np.ndarray, method: str) -> np.ndarray:
    return detect_markers(gray, method=method)


def _default_binarize(gray: np.ndarray, method: str) -> np.ndarray:
    """The binary image detection works on, for diagnosis only."""
    if method == "global":
        return threshold_global(gray, otsu_threshold(gray))
    return threshold_local_mean(gray, default_window(gray.shape), DEFAULT_LOCAL_OFFSET)


@dataclass(frozen=True)
class Stages:
    """The stages that come from ``voxgram.detection``, replaceable for tests.

    * ``to_gray(photo) -> gray``: 2-D float64 in [0, 255].
    * ``detect(gray, method) -> centres``: (4, 2) marker centres ``(x, y)`` in
      photograph pixels, in the order of ``MARKER_CENTERS_MM``. Raises
      ``ValueError`` if the markers are missing or ambiguous.
    * ``binarize(gray, method) -> binary``: the binary image, only to save it.
    """

    to_gray: Callable[[np.ndarray], np.ndarray] = to_grayscale
    detect: Callable[[np.ndarray, str], np.ndarray] = _default_detect
    binarize: Callable[[np.ndarray, str], np.ndarray] = _default_binarize


def default_stages() -> Stages:
    """The stages used by the command line."""
    return Stages()


@dataclass(frozen=True)
class DecodeOptions:
    """Choices for one decoding run.

    Attributes:
        skip: stages to disable, a subset of ``SKIPPABLE_STAGES``.
        threshold: ``"local"`` or ``"global"``, passed to detection.
        card_dpi: resolution of the canonical card image.
        samples_per_axis: samples per cell and axis, see ``extract_gray``.
        cell_fraction: averaging window as a fraction of a cell.
        smoothing_size: size of the binomial smoothing kernel, 0 for none.
        iterations: Griffin-Lim iterations.
        max_unmeasurable: largest share of unmeasurable data cells accepted.
        keep_binary: also produce the binary image of detection, for diagnosis.
    """

    skip: frozenset[str] = frozenset()
    threshold: str = "local"
    card_dpi: float = DEFAULT_CARD_DPI
    samples_per_axis: int = 1
    cell_fraction: float = DEFAULT_CELL_FRACTION
    smoothing_size: int = 0
    iterations: int = GRIFFIN_LIM_ITERATIONS
    max_unmeasurable: float = MAX_UNMEASURABLE_FRACTION
    keep_binary: bool = False

    def __post_init__(self) -> None:
        unknown = set(self.skip) - set(SKIPPABLE_STAGES)
        if unknown:
            raise ValueError(f"cannot skip {sorted(unknown)}; choose from {SKIPPABLE_STAGES}")
        if self.threshold not in THRESHOLD_METHODS:
            raise ValueError(f"threshold must be one of {THRESHOLD_METHODS}")
        if not np.isfinite(self.card_dpi) or self.card_dpi <= 0:
            raise ValueError("card dpi must be positive")
        if self.smoothing_size != 0 and (self.smoothing_size < 3 or self.smoothing_size % 2 == 0):
            raise ValueError("smoothing size must be 0 (off) or an odd number of at least 3")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if not 0.0 <= self.max_unmeasurable <= 1.0:
            raise ValueError("max unmeasurable must be between 0 and 1")


@dataclass(frozen=True)
class DecodeResult:
    """Outputs and bookkeeping of one successful run.

    Attributes:
        samples: the reconstructed signal.
        gray: recovered gray values ``p``, (257, 376), image layout.
        magnitude: recovered magnitude, (257, 376), grid layout.
        timings: ``(stage, seconds)`` for every stage that ran, in order.
        skipped: stages that did not run because of the switches.
        unmeasurable_fraction: share of data cells that could not be measured.
        homography: card millimetres to photograph pixels, ``None`` if not run.
    """

    samples: np.ndarray
    gray: np.ndarray
    magnitude: np.ndarray
    timings: list[tuple[str, float]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    unmeasurable_fraction: float = 0.0
    homography: np.ndarray | None = None


class _Timer:
    """Runs stages, logs and records the time of each, and names the failing stage."""

    def __init__(self) -> None:
        self.records: list[tuple[str, float]] = []

    def run(self, stage: str, function: Callable, *args):
        start = time.perf_counter()
        try:
            result = function(*args)
        except DecodeError:
            raise
        except (ValueError, OSError) as exc:
            raise DecodeError(f"{stage}: {exc}") from exc
        elapsed = time.perf_counter() - start
        self.records.append((stage, elapsed))
        log.info("%-12s %8.3f s", stage, elapsed)
        return result


def _check_image(image: np.ndarray, stage: str) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise DecodeError(f"{stage}: expected a 2-D grayscale image, got {image.ndim}-D")
    return image


def _check_centres(centres: np.ndarray) -> np.ndarray:
    centres = np.asarray(centres, dtype=np.float64)
    if centres.shape != (4, 2) or not np.all(np.isfinite(centres)):
        raise DecodeError(
            f"detect: expected four finite marker centres of shape (4, 2), got {centres.shape}"
        )
    return centres


def data_region(scale_x: float, scale_y: float) -> DataRegion:
    """Data region of the draft layout, in pixels of a canonical card image.

    Args:
        scale_x, scale_y: pixels per millimetre along x and y.
    """
    return DataRegion(
        left=DATA_X_MM * scale_x,
        top=DATA_Y_MM * scale_y,
        width=DATA_WIDTH_MM * scale_x,
        height=DATA_HEIGHT_MM * scale_y,
    )


def _detect(
    gray: np.ndarray,
    options: DecodeOptions,
    stages: Stages,
    timer: _Timer,
    intermediates: dict[str, np.ndarray],
) -> np.ndarray:
    try:
        return _check_centres(timer.run("detect", stages.detect, gray, options.threshold))
    except DecodeError:
        # The binary image is what shows why detection failed.
        if options.keep_binary:
            _keep_binary(gray, options, stages, intermediates)
        raise


def _keep_binary(
    gray: np.ndarray, options: DecodeOptions, stages: Stages, intermediates: dict[str, np.ndarray]
) -> None:
    try:
        intermediates["binary"] = np.asarray(stages.binarize(gray, options.threshold))
    except ValueError as exc:
        log.info("binary image not saved: %s", exc)


def _calibrate(
    gray: np.ndarray, homography: np.ndarray, shape: tuple[int, int], scale: float
) -> np.ndarray:
    rows, columns = np.mgrid[0 : shape[0], 0 : shape[1]]
    # Pixel (u, v) covers millimetres [u, u + 1) / scale, so its centre is half a pixel in.
    x_mm = (columns + 0.5) / scale
    y_mm = (rows + 0.5) / scale
    return calibrate_samples(
        gray,
        homography,
        x_mm,
        y_mm,
        saturation_low=SATURATION_LOW,
        saturation_high=SATURATION_HIGH,
    )


def _fill_unmeasurable(card: np.ndarray, raw: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Replace NaN so the card can be sampled.

    Args:
        card: canonical card, NaN where a value is not available.
        raw: the uncalibrated rectified card, to tell clipping from a real gap.

    Returns:
        ``(filled, unmeasurable)``: the card without NaN, and the mask of pixels
        that are NaN for a reason other than clipping. Clipped black becomes 0,
        clipped white 255, unmeasurable pixels 255 (silence).
    """
    missing = np.isnan(card)
    filled = card.copy()
    unmeasurable = missing
    if raw is not None:
        black = missing & (raw <= SATURATION_LOW)
        white = missing & (raw >= SATURATION_HIGH)
        filled[black] = GRAY_BLACK
        filled[white] = GRAY_WHITE
        unmeasurable = missing & ~black & ~white
    filled[unmeasurable] = GRAY_WHITE
    return filled, unmeasurable


def _unmeasurable_fraction(
    unmeasurable: np.ndarray, region: DataRegion, options: DecodeOptions
) -> float:
    """Share of data cells whose samples touch an unmeasurable pixel."""
    if not unmeasurable.any():
        return 0.0
    mask = np.where(unmeasurable, float(GRAY_WHITE), 0.0)
    touched = extract_gray(
        mask,
        region,
        samples_per_axis=options.samples_per_axis,
        cell_fraction=options.cell_fraction,
    )
    return float(np.mean(touched > 0.0))


def _decode(
    photo: np.ndarray,
    options: DecodeOptions,
    stages: Stages,
    timer: _Timer,
    intermediates: dict[str, np.ndarray],
) -> DecodeResult:
    gray = _check_image(timer.run("grayscale", stages.to_gray, photo), "grayscale")
    intermediates["grayscale"] = gray

    homography = None
    if "detect" not in options.skip:
        centres = _detect(gray, options, stages, timer, intermediates)
        if options.keep_binary:
            _keep_binary(gray, options, stages, intermediates)
        reference = np.array(MARKER_CENTERS_MM, dtype=np.float64)
        homography = timer.run("homography", estimate_homography, reference, centres)

    do_rectify = homography is not None and "rectify" not in options.skip
    do_calibrate = do_rectify and "calibrate" not in options.skip
    skipped = [
        stage
        for stage, ran in (
            ("detect", homography is not None),
            ("rectify", do_rectify),
            ("calibrate", do_calibrate),
        )
        if not ran
    ]
    for stage in skipped:
        log.info("%-12s skipped", stage)

    rectified = None
    raw_for_clipping = None
    if do_rectify:
        scale = options.card_dpi / MM_PER_INCH
        scale_x = scale_y = scale
        shape = print_shape(options.card_dpi)
        to_pixels = homography @ np.linalg.inv(print_matrix(options.card_dpi))
        rectified = timer.run("rectify", warp_to_canonical, gray, to_pixels, shape)
        intermediates["rectified"] = rectified
        card = rectified
        if do_calibrate:
            card = timer.run("calibrate", _calibrate, gray, homography, shape, scale)
            intermediates["corrected"] = card
            raw_for_clipping = rectified
    else:
        # The photograph is taken to be the whole card, edge to edge.
        card = gray
        scale_x = gray.shape[1] / CARD_WIDTH_MM
        scale_y = gray.shape[0] / CARD_HEIGHT_MM

    kernel = binomial_kernel(options.smoothing_size) if options.smoothing_size else None

    def extract() -> tuple[np.ndarray, float]:
        region = data_region(scale_x, scale_y)
        filled, unmeasurable = _fill_unmeasurable(card, raw_for_clipping)
        fraction = _unmeasurable_fraction(unmeasurable, region, options)
        if fraction > options.max_unmeasurable:
            raise DecodeError(
                f"extract: {fraction:.0%} of the data cells could not be measured "
                f"(limit {options.max_unmeasurable:.0%}). Check for glare, a shadow, "
                "or a card that is partly out of the frame."
            )
        if fraction > 0.0:
            log.warning(
                "%.1f%% of the data cells could not be measured, read as silence", 100 * fraction
            )
        recovered = extract_gray(
            filled,
            region,
            samples_per_axis=options.samples_per_axis,
            cell_fraction=options.cell_fraction,
            smoothing=kernel,
        )
        return recovered, fraction

    recovered, fraction = timer.run("extract", extract)
    intermediates["spectrogram"] = recovered
    magnitude = timer.run("magnitude", lambda: gray_to_magnitude(image_to_grid(recovered)))
    samples = timer.run("griffin-lim", lambda: griffin_lim(magnitude, n_iter=options.iterations))

    return DecodeResult(
        samples=samples,
        gray=recovered,
        magnitude=magnitude,
        timings=timer.records,
        skipped=skipped,
        unmeasurable_fraction=fraction,
        homography=homography,
    )


def decode_photo(
    photo: np.ndarray,
    options: DecodeOptions | None = None,
    stages: Stages | None = None,
    *,
    intermediates: dict[str, np.ndarray] | None = None,
) -> DecodeResult:
    """Decode a photograph held in memory. Nothing is written.

    Args:
        photo: as returned by ``imageio.read_image``.
        options: see :class:`DecodeOptions`.
        stages: stage implementations, :func:`default_stages` if omitted.
        intermediates: if given, filled in place with the intermediate images as
            they are produced, so they are available even when a later stage
            fails.

    Raises:
        DecodeError: if any stage fails. No signal is produced.
    """
    return _decode(
        photo,
        options or DecodeOptions(),
        stages or default_stages(),
        _Timer(),
        intermediates if intermediates is not None else {},
    )


def decode_file(
    photo_path: str | Path,
    output_path: str | Path,
    options: DecodeOptions | None = None,
    stages: Stages | None = None,
    *,
    intermediates: dict[str, np.ndarray] | None = None,
) -> DecodeResult:
    """Read a photograph and write the WAV file.

    The WAV file is written last, only after every earlier stage has succeeded.
    A failure leaves no audio file behind.

    Raises:
        DecodeError: if reading, any stage or writing fails.
    """
    photo_path = Path(photo_path)
    if not photo_path.is_file():
        raise DecodeError(f"read: photograph not found: {photo_path}")
    timer = _Timer()
    photo = timer.run("read", read_image, photo_path)
    result = _decode(
        photo,
        options or DecodeOptions(),
        stages or default_stages(),
        timer,
        intermediates if intermediates is not None else {},
    )
    timer.run("write", save_audio, output_path, result.samples)
    return result


def _as_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.bool_:
        return image.astype(np.uint8) * 255
    if image.dtype == np.uint8:
        return image
    image = np.nan_to_num(image.astype(np.float64), nan=INVALID_PIXEL)
    return np.rint(np.clip(image, 0, 255)).astype(np.uint8)


def save_intermediates(directory: str | Path, intermediates: dict[str, np.ndarray]) -> list[Path]:
    """Write the intermediate images as PNG files and return their paths.

    Pixels that are NaN (unmeasurable or outside the photograph) are written as
    ``INVALID_PIXEL``.
    """
    directory = Path(directory)
    written = []
    for key, name in INTERMEDIATE_FILES.items():
        if key in intermediates:
            path = directory / name
            write_image(path, _as_uint8(intermediates[key]))
            written.append(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m voxgram.decoder",
        description="Decode a photograph of a Voxgram card into a WAV file.",
    )
    parser.add_argument("photo", type=Path, help="photograph of the card")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output WAV file (default: outputs/decoded/<photo name>.wav)",
    )
    parser.add_argument(
        "--skip",
        action="append",
        choices=SKIPPABLE_STAGES,
        default=None,
        metavar="STAGE",
        help=(
            "disable a stage, for the ablation; repeat to disable several "
            f"({', '.join(SKIPPABLE_STAGES)}). Skipping detect also skips the geometry and "
            "the calibration, and skipping rectify also skips the calibration"
        ),
    )
    parser.add_argument(
        "--threshold",
        choices=THRESHOLD_METHODS,
        default="local",
        help="thresholding used by marker detection (default: local)",
    )
    parser.add_argument(
        "--card-dpi",
        type=float,
        default=DEFAULT_CARD_DPI,
        help=(
            "resolution of the rectified card (default: %(default)s). "
            f"{TARGET_PRINT_DPI:g} dpi matches the print but needs about 2 GB of memory"
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="samples per cell and axis, averaged (default: 1, the cell centre)",
    )
    parser.add_argument(
        "--cell-fraction",
        type=float,
        default=DEFAULT_CELL_FRACTION,
        help="share of the cell covered by the averaging window (default: %(default)s)",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        metavar="SIZE",
        help="smooth the grid with a binomial kernel of this odd size (default: 0, off)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=GRIFFIN_LIM_ITERATIONS,
        help="Griffin-Lim iterations (default: %(default)s)",
    )
    parser.add_argument(
        "--max-unmeasurable",
        type=float,
        default=MAX_UNMEASURABLE_FRACTION,
        help="largest share of unmeasurable data cells accepted (default: %(default)s)",
    )
    parser.add_argument(
        "--save-intermediates",
        type=Path,
        metavar="DIR",
        help="save the grayscale photograph, binary image, rectified card, "
        "corrected card and recovered spectrogram in DIR",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="do not log the stage times")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        options = DecodeOptions(
            skip=frozenset(args.skip or ()),
            threshold=args.threshold,
            card_dpi=args.card_dpi,
            samples_per_axis=args.samples,
            cell_fraction=args.cell_fraction,
            smoothing_size=args.smooth,
            iterations=args.iterations,
            max_unmeasurable=args.max_unmeasurable,
            keep_binary=args.save_intermediates is not None,
        )
    except ValueError as exc:
        parser.error(str(exc))

    output = args.output or Path("outputs") / "decoded" / f"{args.photo.stem}.wav"

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.WARNING if args.quiet else logging.INFO)

    intermediates: dict[str, np.ndarray] = {}
    try:
        try:
            result = decode_file(args.photo, output, options, intermediates=intermediates)
        except DecodeError as exc:
            if args.save_intermediates:
                save_intermediates(args.save_intermediates, intermediates)
            parser.exit(1, f"error: {exc}\nno audio was written\n")
    finally:
        log.removeHandler(handler)

    if args.save_intermediates:
        for path in save_intermediates(args.save_intermediates, intermediates):
            print(f"wrote {path}")
    total = sum(seconds for _, seconds in result.timings)
    print(f"wrote {output} ({total:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
