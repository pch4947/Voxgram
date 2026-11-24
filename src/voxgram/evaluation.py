"""Evaluation: what each correction stage contributes.

Every photograph is decoded once per *variant* of the decoder (the full
pipeline, and the pipeline with a stage switched off or an option changed),
and each run is compared with the gray values ``q`` that were printed.

Two sources of photographs:

* ``captures``: real photographs listed in ``experiments/captures.csv``, read
  from ``data/`` with the layout of ``experiments/README.md`` section 5. The
  encoded values come from ``data/cards/<card>_source.png`` (the spectrogram
  image or the print image of the card) or, failing that,
  ``data/cards/<card>_source.wav``.
* ``synthetic``: photographs rendered from recordings through known
  homographies, illumination and tone curves (``voxgram.synthetic``), saved
  as JPEG and read back. The true homography is known, so geometric errors
  can be measured. There is no blur, halftone or lens distortion, so these
  test the logic of the pipeline, not its robustness.

Metrics (``docs/PROJECT_PLAN.md`` section 7; the spectral error is Proposed):

* ``mae_gray``: mean absolute error between the recovered ``p`` and the encoded
  ``q`` over the 257 x 376 grid, in gray levels. ``mae_db``: the same in dB.
* ``marker_error_mm``: largest distance between where the estimated
  homography puts the four markers and where they are, in card millimetres.
  ``cell_error_mm``: the same over every data cell centre, which is what
  extraction depends on. Synthetic photographs only: a real photograph has no
  ground truth, and with four markers the fit itself is exact.
* ``spectral_error_db``: the reconstructed audio is encoded again, the way the
  encoder does it, and compared with ``q`` in dB. The digital baseline, which
  runs Griffin-Lim with the same settings on ``q`` itself, gives the lower
  bound: the part of the error that comes from phase reconstruction alone.
* ``detect_ok``: ``yes``, ``no`` or ``ambiguous``, as in ``captures.csv``.

Command line:
    python -m voxgram.evaluation synthetic [--audio voice.wav ...]
    python -m voxgram.evaluation captures [--append-captures]

Outputs go in ``outputs/eval/<commit>/`` (ignored by Git): ``results.csv``
with one row per photograph and variant, ``summary.md`` with the tables, and
the reconstructed WAV files for listening. ``--report`` also writes the
summary where it can be committed, for example under ``experiments/``.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from voxgram import card, synthetic
from voxgram.audio_backend import (
    GRIFFIN_LIM_ITERATIONS,
    griffin_lim,
    load_audio,
    save_audio,
    stft_magnitude,
)
from voxgram.decoder import STAGES, DecodeError, DecodeOptions, decode_photo
from voxgram.detection import MarkerDetectionError, to_grayscale
from voxgram.encoder import encode_samples
from voxgram.format import (
    CARD_HEIGHT_MM,
    CARD_WIDTH_MM,
    DB_FLOOR,
    FORMAT_VERSION,
    GRAY_WHITE,
    GRID_SHAPE,
    MARKER_CENTERS_MM,
)
from voxgram.geometry import apply_homography
from voxgram.imageio import read_image, write_image
from voxgram.intensity import gray_to_decibels, gray_to_magnitude, image_to_grid

# --- variants ----------------------------------------------------------------

# Order in which skipped stages are written in the ``stages`` column.
_SKIP_ORDER = ("detect", "rectify", "calibrate")


@dataclass(frozen=True)
class Variant:
    """One configuration of the decoder.

    Attributes:
        name: short name used in file names and tables.
        skip: stages switched off, as for ``--skip``.
        threshold: ``"local"`` or ``"global"``.
        smoothing_size: binomial smoothing kernel size, 0 for none.
        samples_per_axis: samples per cell and axis, averaged.
    """

    name: str
    skip: frozenset[str] = frozenset()
    threshold: str = "local"
    smoothing_size: int = 0
    samples_per_axis: int = 1

    @property
    def stages(self) -> str:
        """The ``stages`` column of ``captures.csv``: ``all`` or ``rectify+calibrate``.

        Skipping a stage also skips the stages that depend on it, as in the
        decoder, and the column records what actually ran.
        """
        skipped = set(self.skip)
        if "detect" in skipped:
            skipped |= {"rectify", "calibrate"}
        if "rectify" in skipped:
            skipped.add("calibrate")
        if not skipped:
            return "all"
        return "+".join(stage for stage in _SKIP_ORDER if stage in skipped)

    @property
    def notes(self) -> str:
        """Options other than ``--skip``, written as on the decoder's command line."""
        parts = []
        if self.threshold != "local":
            parts.append(f"--threshold {self.threshold}")
        if self.smoothing_size:
            parts.append(f"--smooth {self.smoothing_size}")
        if self.samples_per_axis != 1:
            parts.append(f"--samples {self.samples_per_axis}")
        return " ".join(parts)

    def options(self, iterations: int) -> DecodeOptions:
        return DecodeOptions(
            skip=self.skip,
            threshold=self.threshold,
            smoothing_size=self.smoothing_size,
            samples_per_axis=self.samples_per_axis,
            iterations=iterations,
        )


VARIANTS: dict[str, Variant] = {
    variant.name: variant
    for variant in (
        Variant("full"),
        Variant("no-calibrate", skip=frozenset({"calibrate"})),
        Variant("no-rectify", skip=frozenset({"rectify"})),
        Variant("no-detect", skip=frozenset({"detect"})),
        Variant("global-threshold", threshold="global"),
        Variant("smooth-3", smoothing_size=3),
        Variant("samples-3", samples_per_axis=3),
    )
}
FULL = "full"

# --- metrics -----------------------------------------------------------------

_FLOOR_AMPLITUDE = 10.0 ** (DB_FLOOR / 20.0)


def gray_error(recovered: np.ndarray, encoded: np.ndarray) -> tuple[float, float]:
    """Mean absolute error between recovered ``p`` and encoded ``q``.

    Args:
        recovered: (257, 376) gray values in image layout, float.
        encoded: (257, 376) encoded image in the same layout.

    Returns:
        ``(mae_gray, mae_db)``: in gray levels and in dB.
    """
    recovered = np.asarray(recovered, dtype=np.float64)
    encoded = np.asarray(encoded, dtype=np.float64)
    if recovered.shape != GRID_SHAPE or encoded.shape != GRID_SHAPE:
        raise ValueError(f"expected two {GRID_SHAPE} images")
    mae_gray = float(np.mean(np.abs(recovered - encoded)))
    mae_db = float(np.mean(np.abs(gray_to_decibels(recovered) - gray_to_decibels(encoded))))
    return mae_gray, mae_db


def spectral_error_db(samples: np.ndarray, encoded: np.ndarray) -> float:
    """How far reconstructed audio is from the encoded spectrogram, in dB.

    The STFT magnitude of ``samples`` is normalized to its peak and clipped at
    ``DB_FLOOR``, as the encoder does, and compared cell by cell with the dB
    values of ``encoded``.

    Args:
        samples: reconstructed signal, NUM_SAMPLES long.
        encoded: (257, 376) encoded image, image layout.

    Returns:
        Mean absolute difference in dB over the 257 x 376 grid.
    """
    magnitude = stft_magnitude(np.asarray(samples, dtype=np.float32)).astype(np.float64)
    peak = magnitude.max()
    if peak <= 0:
        return float(-DB_FLOOR)
    achieved = 20.0 * np.log10(np.maximum(magnitude / peak, _FLOOR_AMPLITUDE))
    target = gray_to_decibels(image_to_grid(np.asarray(encoded, dtype=np.float64)))
    return float(np.mean(np.abs(achieved - target)))


def _card_points_seen_as(
    estimated: np.ndarray, true: np.ndarray, points_mm: np.ndarray
) -> np.ndarray:
    """Where card points land on the card when sampled through ``estimated``.

    The decoder samples the photograph at ``estimated @ point``; that pixel
    shows the card point ``true^-1 @ estimated @ point``.
    """
    pixels = apply_homography(estimated, points_mm)
    return apply_homography(np.linalg.inv(true), pixels)


def marker_error_mm(estimated: np.ndarray, true: np.ndarray) -> float:
    """Largest marker position error, in card millimetres.

    Args:
        estimated: (3, 3) homography card mm to photo pixels, from the decoder.
        true: (3, 3) homography the photograph was rendered with.
    """
    markers = np.asarray(MARKER_CENTERS_MM, dtype=np.float64)
    seen = _card_points_seen_as(estimated, true, markers)
    return float(np.max(np.hypot(*(seen - markers).T)))


def cell_error_mm(estimated: np.ndarray, true: np.ndarray) -> float:
    """Largest sampling position error over the data cell centres, in mm."""
    x, y = card.data_cell_centers_mm()
    cells = np.column_stack([x.ravel(), y.ravel()])
    seen = _card_points_seen_as(estimated, true, cells)
    return float(np.max(np.hypot(*(seen - cells).T)))


def detection_outcome(error: DecodeError | None, variant: Variant) -> str:
    """The ``detect_ok`` value of one run.

    Empty when detection was switched off. A failure in a later stage still
    counts as a successful detection.
    """
    if "detect" in variant.skip:
        return ""
    if error is None:
        return "yes"
    message = str(error)
    if not message.startswith(("detect:", "homography:")):
        return "yes"
    if isinstance(error.__cause__, MarkerDetectionError) and "refusing to guess" in message:
        return "ambiguous"
    return "no"


# --- inputs ------------------------------------------------------------------


@dataclass
class Capture:
    """One photograph to evaluate.

    Attributes:
        capture_id: name of the photograph without extension.
        condition: capture condition used to group detection rates.
        card: card code, or the recording name for synthetic cards.
        photo_path: the photograph.
        encoded: (257, 376) uint8 gray values that were printed.
        true_homography: card mm to photo pixels, synthetic photographs only.
        setup_ok: False for a capture that did not follow the protocol. Kept
            in ``results.csv`` but left out of the summary.
        row: the row of ``captures.csv`` the capture comes from, if any.
    """

    capture_id: str
    condition: str
    card: str
    photo_path: Path
    encoded: np.ndarray
    true_homography: np.ndarray | None = None
    setup_ok: bool = True
    row: dict[str, str] | None = None


def encoded_from_image(image: np.ndarray) -> np.ndarray:
    """The encoded gray values from a spectrogram image or a card print image.

    A (257, 376) image is returned as is. Anything else is taken to be the
    print image of the whole card, as written by ``voxgram.encoder --card``,
    and is sampled at the data cell centres.
    """
    gray = np.rint(to_grayscale(image))
    if gray.shape == GRID_SHAPE:
        return gray.astype(np.uint8)
    height, width = gray.shape
    scale = width / CARD_WIDTH_MM
    if abs(height / scale - CARD_HEIGHT_MM) > 1.0:
        raise ValueError(
            f"image of {width} x {height} pixels is neither a {GRID_SHAPE[1]} x {GRID_SHAPE[0]} "
            f"spectrogram nor a {CARD_WIDTH_MM:g} x {CARD_HEIGHT_MM:g} mm card"
        )
    x, y = card.data_cell_centers_mm()
    # Pixel centres are at integer coordinates (voxgram.card.print_matrix).
    columns = np.clip(np.rint(x * scale - 0.5).astype(int), 0, width - 1)
    rows = np.clip(np.rint(y * scale - 0.5).astype(int), 0, height - 1)
    return gray[rows, columns].astype(np.uint8)


def load_encoded(card_code: str, cards_dir: Path) -> np.ndarray:
    """Encoded gray values of a printed card, from ``cards_dir``."""
    image_path = cards_dir / f"{card_code}_source.png"
    if image_path.is_file():
        return encoded_from_image(read_image(image_path))
    audio_path = cards_dir / f"{card_code}_source.wav"
    if audio_path.is_file():
        return encode_samples(load_audio(audio_path), truncate=True)
    raise FileNotFoundError(f"no {image_path.name} or {audio_path.name} in {cards_dir}")


PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".JPG", ".JPEG", ".png")


def _find_photo(capture_id: str, photos_dir: Path) -> Path:
    folder = photos_dir / capture_id[:8]
    for extension in PHOTO_EXTENSIONS:
        for path in (folder / f"{capture_id}{extension}", photos_dir / f"{capture_id}{extension}"):
            if path.is_file():
                return path
    raise FileNotFoundError(f"photograph {capture_id} not found under {photos_dir}")


def capture_condition(capture_id: str) -> str:
    """``<tilt>_<light>`` from a capture ID such as ``20261117_c01_p1_t30p_l0_02``."""
    parts = capture_id.split("_")
    if len(parts) < 6:
        return "unknown"
    return f"{parts[3]}_{parts[4]}"


def read_captures(captures_csv: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(captures_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def real_captures(
    captures_csv: Path, data_dir: Path, *, only: Sequence[str] = ()
) -> tuple[list[Capture], list[str]]:
    """The photographs listed in ``captures.csv``, one per capture ID.

    Returns:
        ``(captures, problems)``: captures that could be loaded, and a message
        for every one that could not.
    """
    _, rows = read_captures(captures_csv)
    first_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        first_rows.setdefault(row["capture_id"], row)

    encoded_cache: dict[str, np.ndarray] = {}
    captures, problems = [], []
    for capture_id, row in first_rows.items():
        if only and not any(pattern in capture_id for pattern in only):
            continue
        try:
            photo = _find_photo(capture_id, data_dir / "photos")
            code = row["card"]
            if code not in encoded_cache:
                encoded_cache[code] = load_encoded(code, data_dir / "cards")
        except (FileNotFoundError, ValueError) as exc:
            problems.append(f"{capture_id}: {exc}")
            continue
        captures.append(
            Capture(
                capture_id=capture_id,
                condition=capture_condition(capture_id),
                card=code,
                photo_path=photo,
                encoded=encoded_cache[code],
                setup_ok=row.get("setup_ok", "yes").strip().lower() != "no",
                row=row,
            )
        )
    return captures, problems


@dataclass(frozen=True)
class SyntheticCondition:
    """Viewing and lighting parameters of one synthetic photograph.

    ``scale`` is in pixels per millimetre at the card centre, ``perspective``
    as in ``synthetic.make_homography``, ``gradient`` the lowest gain of a
    linear illumination ramp (1.0 for none), ``gamma`` the tone curve exponent
    and ``noise`` the standard deviation of Gaussian sensor noise in gray levels.
    """

    name: str
    rotation_deg: float = 0.0
    perspective: tuple[float, float] = (0.0, 0.0)
    gradient: float = 1.0
    gradient_angle_deg: float = 0.0
    gamma: float = 1.0
    gain: float = 1.0
    noise: float = 1.5


SYNTHETIC_CONDITIONS = (
    SyntheticCondition("front"),
    SyntheticCondition("roll-15", rotation_deg=15.0),
    SyntheticCondition("tilt-mild", perspective=(0.0008, 0.0)),
    SyntheticCondition("tilt-moderate", rotation_deg=5.0, perspective=(0.0015, 0.0008)),
    SyntheticCondition("dim", gain=0.6, gamma=1.3),
    SyntheticCondition("gradient", gradient=0.5, gradient_angle_deg=20.0),
    SyntheticCondition(
        "combined",
        rotation_deg=8.0,
        perspective=(0.0012, -0.0006),
        gradient=0.6,
        gradient_angle_deg=120.0,
        gamma=1.4,
        noise=3.0,
    ),
)

# Photograph size and card scale: the data region is about 1,450 pixels wide,
# close to the 1,500 pixels the capture protocol asks for.
SYNTHETIC_SHAPE = (1800, 2400)
SYNTHETIC_SCALE = 13.0


def render_synthetic(
    encoded: np.ndarray,
    condition: SyntheticCondition,
    *,
    shape: tuple[int, int] = SYNTHETIC_SHAPE,
    scale: float = SYNTHETIC_SCALE,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic photograph of the card and the homography it was made with.

    Returns:
        ``(photo, homography)``: (H, W) uint8 photograph and (3, 3) card mm to
        photo pixels.
    """
    height, width = shape
    homography = synthetic.make_homography(
        scale=scale,
        center=(width / 2, height / 2),
        rotation_deg=condition.rotation_deg,
        perspective=condition.perspective,
    )
    photo = card.render_view(encoded, homography, shape, supersample=2)
    if condition.gradient < 1.0:
        photo = photo * synthetic.illumination_gradient(
            shape, low=condition.gradient, high=1.0, angle_deg=condition.gradient_angle_deg
        )
    photo = photo * condition.gain
    if condition.gamma != 1.0:
        photo = synthetic.apply_tone_curve(photo, condition.gamma)
    if condition.noise > 0:
        photo = photo + np.random.default_rng(seed).normal(0.0, condition.noise, photo.shape)
    return np.clip(np.rint(photo), 0, GRAY_WHITE).astype(np.uint8), homography


def synthetic_captures(
    recordings: Iterable[tuple[str, np.ndarray]],
    photos_dir: Path,
    *,
    conditions: Sequence[SyntheticCondition] = SYNTHETIC_CONDITIONS,
    shape: tuple[int, int] = SYNTHETIC_SHAPE,
    scale: float = SYNTHETIC_SCALE,
) -> list[Capture]:
    """Render and save one JPEG photograph per recording and condition.

    Args:
        recordings: ``(name, samples)``; recordings longer than 3.0 s are cut.
        photos_dir: where the photographs are written.
    """
    captures = []
    for card_index, (name, samples) in enumerate(recordings):
        encoded = encode_samples(samples, truncate=True)
        for condition_index, condition in enumerate(conditions):
            photo, homography = render_synthetic(
                encoded,
                condition,
                shape=shape,
                scale=scale,
                seed=1000 * card_index + condition_index,
            )
            capture_id = f"{name}_{condition.name}"
            path = photos_dir / f"{capture_id}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_image(path, photo)
            captures.append(
                Capture(
                    capture_id=capture_id,
                    condition=condition.name,
                    card=name,
                    photo_path=path,
                    encoded=encoded,
                    true_homography=homography,
                )
            )
    return captures


# --- running -----------------------------------------------------------------


@dataclass
class RunResult:
    """One photograph decoded with one variant."""

    capture: Capture
    variant: Variant
    detect_ok: str
    error: str = ""
    mae_gray: float = math.nan
    mae_db: float = math.nan
    marker_error_mm: float = math.nan
    cell_error_mm: float = math.nan
    unmeasurable: float = math.nan
    spectral_error_db: float = math.nan
    baseline_spectral_error_db: float = math.nan
    total_s: float = math.nan
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def decoded(self) -> bool:
        return not self.error


@dataclass
class Baseline:
    """Griffin-Lim on the encoded values themselves, with the same settings."""

    samples: np.ndarray
    spectral_error_db: float


def digital_baseline(encoded: np.ndarray, iterations: int) -> Baseline:
    samples = griffin_lim(gray_to_magnitude(image_to_grid(encoded)), n_iter=iterations)
    return Baseline(samples, spectral_error_db(samples, encoded))


def evaluate_capture(
    capture: Capture,
    variant: Variant,
    baseline: Baseline,
    *,
    iterations: int,
    audio_dir: Path | None = None,
) -> RunResult:
    """Decode one photograph with one variant and measure the result."""
    photo = read_image(capture.photo_path)
    start = time.perf_counter()
    try:
        decoded = decode_photo(photo, variant.options(iterations))
    except DecodeError as exc:
        return RunResult(
            capture,
            variant,
            detect_ok=detection_outcome(exc, variant),
            error=str(exc),
            baseline_spectral_error_db=baseline.spectral_error_db,
            total_s=time.perf_counter() - start,
        )
    result = RunResult(
        capture,
        variant,
        detect_ok=detection_outcome(None, variant),
        unmeasurable=decoded.unmeasurable_fraction,
        spectral_error_db=spectral_error_db(decoded.samples, capture.encoded),
        baseline_spectral_error_db=baseline.spectral_error_db,
        total_s=time.perf_counter() - start,
        timings=dict(decoded.timings),
    )
    result.mae_gray, result.mae_db = gray_error(decoded.gray, capture.encoded)
    if capture.true_homography is not None and decoded.homography is not None:
        result.marker_error_mm = marker_error_mm(decoded.homography, capture.true_homography)
        result.cell_error_mm = cell_error_mm(decoded.homography, capture.true_homography)
    if audio_dir is not None:
        save_audio(audio_dir / capture.capture_id / f"{variant.name}.wav", decoded.samples)
    return result


def evaluate(
    captures: Sequence[Capture],
    variants: Sequence[Variant],
    *,
    iterations: int = GRIFFIN_LIM_ITERATIONS,
    audio_dir: Path | None = None,
    progress: bool = True,
) -> list[RunResult]:
    """Decode every capture with every variant."""
    baselines: dict[str, Baseline] = {}
    results = []
    total = len(captures) * len(variants)
    for capture in captures:
        if capture.card not in baselines:
            baselines[capture.card] = digital_baseline(capture.encoded, iterations)
            if audio_dir is not None:
                save_audio(
                    audio_dir / "baseline" / f"{capture.card}.wav",
                    baselines[capture.card].samples,
                )
        for variant in variants:
            if progress:
                print(
                    f"[{len(results) + 1}/{total}] {capture.capture_id} {variant.name}",
                    file=sys.stderr,
                )
            results.append(
                evaluate_capture(
                    capture,
                    variant,
                    baselines[capture.card],
                    iterations=iterations,
                    audio_dir=audio_dir,
                )
            )
    return results


# --- outputs -----------------------------------------------------------------

RESULT_COLUMNS = (
    "capture_id",
    "condition",
    "card",
    "setup_ok",
    "variant",
    "stages",
    "notes",
    "detect_ok",
    "mae_gray",
    "mae_db",
    "marker_error_mm",
    "cell_error_mm",
    "unmeasurable",
    "spectral_error_db",
    "baseline_spectral_error_db",
    "total_s",
    *(f"t_{stage}" for stage in STAGES if stage not in ("read", "write")),
    "error",
)


def _number(value: float, digits: int = 3) -> str:
    return "" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def write_results_csv(results: Sequence[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(RESULT_COLUMNS)
        for r in results:
            timings = [
                _number(r.timings.get(stage, math.nan))
                for stage in STAGES
                if stage not in ("read", "write")
            ]
            writer.writerow(
                [
                    r.capture.capture_id,
                    r.capture.condition,
                    r.capture.card,
                    "yes" if r.capture.setup_ok else "no",
                    r.variant.name,
                    r.variant.stages,
                    r.variant.notes,
                    r.detect_ok,
                    _number(r.mae_gray),
                    _number(r.mae_db),
                    _number(r.marker_error_mm, 4),
                    _number(r.cell_error_mm, 4),
                    _number(r.unmeasurable, 4),
                    _number(r.spectral_error_db),
                    _number(r.baseline_spectral_error_db),
                    _number(r.total_s),
                    *timings,
                    r.error,
                ]
            )


def _mean(values: Iterable[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else math.nan


def _cell(value: float, digits: int = 2, suffix: str = "") -> str:
    return "-" if math.isnan(value) else f"{value:.{digits}f}{suffix}"


def _rate(results: Sequence[RunResult]) -> str:
    attempted = [r for r in results if r.detect_ok]
    if not attempted:
        return "-"
    ok = sum(r.detect_ok == "yes" for r in attempted)
    return f"{ok}/{len(attempted)}"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def summary_markdown(
    results: Sequence[RunResult],
    *,
    source: str,
    commit: str,
    problems: Sequence[str] = (),
) -> str:
    """The result tables, as Markdown."""
    counted = [r for r in results if r.capture.setup_ok]
    variants = list(dict.fromkeys(r.variant.name for r in results))
    captures = list(dict.fromkeys(r.capture.capture_id for r in counted))
    has_geometry = any(r.capture.true_homography is not None for r in counted)
    excluded = len({r.capture.capture_id for r in results if not r.capture.setup_ok})

    lines = [
        f"# Evaluation, {source} photographs",
        "",
        f"Commit `{commit}`, format `{FORMAT_VERSION}`, {date.today().isoformat()}. "
        f"{len(captures)} photographs, {len(variants)} decoder variants."
        + (f" {excluded} photographs with `setup_ok = no` left out." if excluded else ""),
        "",
        "Means are over the runs that decoded. `MAE` compares the recovered gray "
        "values with the encoded ones; `spectral` compares the reconstructed audio "
        "with the encoded spectrogram, and the digital baseline is its lower bound "
        "(same Griffin-Lim settings, no printing).",
        "",
        "## Stage-by-stage comparison",
        "",
    ]
    baseline = _mean(r.baseline_spectral_error_db for r in counted if r.variant.name == variants[0])
    rows = [["digital baseline", "-", "-", "-", "-", "0.00", "0.00", _cell(baseline), "-"]]
    for name in variants:
        runs = [r for r in counted if r.variant.name == name]
        ok = [r for r in runs if r.decoded]
        variant = runs[0].variant if runs else VARIANTS.get(name, Variant(name))
        rows.append(
            [
                name,
                variant.stages,
                variant.notes or "-",
                _rate(runs),
                f"{len(ok)}/{len(runs)}",
                _cell(_mean(r.mae_gray for r in ok)),
                _cell(_mean(r.mae_db for r in ok)),
                _cell(_mean(r.spectral_error_db for r in ok)),
                _cell(_mean(r.total_s for r in ok), 1),
            ]
        )
    lines += _table(
        (
            "Variant",
            "Skipped",
            "Options",
            "Detected",
            "Decoded",
            "MAE gray",
            "MAE dB",
            "Spectral dB",
            "Time s",
        ),
        rows,
    )

    lines += ["", "## Detection per capture condition", ""]
    # Only the threshold changes what detection does; the other variants repeat it.
    detect_variants = [n for n in variants if n in (FULL, "global-threshold")]
    rows = []
    for condition in dict.fromkeys(r.capture.condition for r in counted):
        runs = [r for r in counted if r.capture.condition == condition]
        row = [condition, str(len({r.capture.capture_id for r in runs}))]
        row += [_rate([r for r in runs if r.variant.name == n]) for n in detect_variants]
        rows.append(row)
    lines += _table(("Condition", "Photos", *(f"Detected, {n}" for n in detect_variants)), rows)

    if FULL in variants:
        lines += ["", "## Full pipeline, per photograph", ""]
        header = ["Photograph", "Condition", "Detected", "MAE gray", "MAE dB"]
        if has_geometry:
            header += ["Marker error mm", "Cell error mm"]
        header += ["Unmeasurable", "Spectral dB", "Baseline dB", "Error"]
        rows = []
        for r in (r for r in counted if r.variant.name == FULL):
            row = [r.capture.capture_id, r.capture.condition, r.detect_ok or "-"]
            row += [_cell(r.mae_gray), _cell(r.mae_db)]
            if has_geometry:
                row += [_cell(r.marker_error_mm, 3), _cell(r.cell_error_mm, 3)]
            row += [
                _cell(100 * r.unmeasurable, 1, " %"),
                _cell(r.spectral_error_db),
                _cell(r.baseline_spectral_error_db),
                r.error.replace("|", "/") or "-",
            ]
            rows.append(row)
        lines += _table(header, rows)

        full_ok = [r for r in counted if r.variant.name == FULL and r.decoded]
        if full_ok:
            lines += ["", "## Time per stage, full pipeline", ""]
            stages = [s for s in STAGES if any(s in r.timings for r in full_ok)]
            lines += _table(
                ("Stage", "Mean s"),
                ([s, _cell(_mean(r.timings.get(s, math.nan) for r in full_ok), 3)] for s in stages),
            )

    if problems:
        lines += ["", "## Photographs not evaluated", ""]
        lines += [f"- {p}" for p in problems]
    return "\n".join(lines) + "\n"


def append_to_captures(results: Sequence[RunResult], captures_csv: Path, commit: str) -> int:
    """Record the results in ``captures.csv``, following its rules.

    A capture row that has not been processed yet (empty ``commit``) receives
    the result of the full pipeline. Every other result is a copy of the
    capture's first row with ``commit``, ``stages``, the results and the
    variant's options in ``notes``. A result already recorded for the same
    capture, commit, stages and notes is not written again.

    Returns:
        The number of rows filled in or added.
    """
    header, rows = read_captures(captures_csv)
    raw = captures_csv.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"

    def key(row: dict[str, str]) -> tuple[str, str, str, str]:
        return (row["capture_id"], row["commit"], row["stages"], row["notes"])

    recorded = {key(row) for row in rows if row["commit"]}
    first_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        first_rows.setdefault(row["capture_id"], row)
    changed = 0
    for r in results:
        # The row as it is in the file now, so that filling it in changes the file.
        base = first_rows.get(r.capture.capture_id)
        if base is None:
            continue
        notes = "; ".join(n for n in (base.get("notes", ""), r.variant.notes) if n)
        values = {
            "commit": commit,
            "stages": r.variant.stages,
            "detect_ok": r.detect_ok,
            "mae_gray": _number(r.mae_gray, 2),
            "mae_db": _number(r.mae_db, 2),
            "intelligibility": "",
            "notes": notes,
        }
        new = {**base, **values}
        if key(new) in recorded:
            continue
        if not base["commit"] and r.variant.name == FULL and not r.variant.notes:
            base.update(values)
        else:
            rows.append(new)
        recorded.add(key(new))
        changed += 1

    with open(captures_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, lineterminator=newline)
        writer.writeheader()
        writer.writerows(rows)
    return changed


def current_commit() -> str:
    """Seven-character hash of HEAD, with ``-dirty`` if the tree has changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


# --- command line ------------------------------------------------------------

DEFAULT_RECORDINGS = ("libri1", "libri2", "libri3")


def _recordings(paths: Sequence[Path]) -> list[tuple[str, np.ndarray]]:
    if paths:
        return [(path.stem, load_audio(path)) for path in paths]
    import librosa  # only for the example recordings

    return [(name, load_audio(librosa.ex(name))) for name in DEFAULT_RECORDINGS]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m voxgram.evaluation",
        description="Decode photographs with every decoder variant and measure each stage.",
    )
    parser.add_argument("source", choices=("synthetic", "captures"), help="which photographs")
    parser.add_argument("-o", "--output-dir", type=Path, help="default: outputs/eval/<commit>")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=list(VARIANTS),
        help="decoder variants to run (default: all)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=GRIFFIN_LIM_ITERATIONS,
        help="Griffin-Lim iterations, for the decoder and the baseline alike "
        "(default: %(default)s)",
    )
    parser.add_argument("--no-audio", action="store_true", help="do not write WAV files")
    parser.add_argument("--report", type=Path, help="also write the summary to this file")
    synthetic_group = parser.add_argument_group("synthetic")
    synthetic_group.add_argument(
        "--audio",
        type=Path,
        nargs="+",
        default=[],
        help="recordings to put on the synthetic cards (default: the Librosa LibriSpeech examples)",
    )
    synthetic_group.add_argument(
        "--conditions",
        nargs="+",
        choices=[c.name for c in SYNTHETIC_CONDITIONS],
        help="synthetic conditions to render (default: all)",
    )
    captures_group = parser.add_argument_group("captures")
    captures_group.add_argument(
        "--captures", type=Path, default=Path("experiments/captures.csv"), help="capture table"
    )
    captures_group.add_argument("--data", type=Path, default=Path("data"), help="raw data folder")
    captures_group.add_argument(
        "--only", nargs="+", default=[], help="evaluate only capture IDs containing these strings"
    )
    captures_group.add_argument(
        "--append-captures",
        action="store_true",
        help="record the results in the capture table",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("iterations must be at least 1")

    commit = current_commit()
    if commit.endswith("-dirty"):
        print("warning: uncommitted changes; results are not tied to a commit", file=sys.stderr)
    output_dir = args.output_dir or Path("outputs") / "eval" / commit
    variants = [VARIANTS[name] for name in args.variants]

    problems: list[str] = []
    if args.source == "synthetic":
        conditions = [
            c for c in SYNTHETIC_CONDITIONS if not args.conditions or c.name in args.conditions
        ]
        try:
            recordings = _recordings(args.audio)
        except (OSError, ValueError) as exc:
            parser.exit(1, f"error: {exc}\n")
        print(f"rendering {len(recordings) * len(conditions)} photographs", file=sys.stderr)
        captures = synthetic_captures(recordings, output_dir / "photos", conditions=conditions)
    else:
        if not args.captures.is_file():
            parser.exit(1, f"error: {args.captures} not found\n")
        captures, problems = real_captures(args.captures, args.data, only=args.only)
        for problem in problems:
            print(f"skip {problem}", file=sys.stderr)
    if not captures:
        parser.exit(1, "error: no photographs to evaluate\n")

    results = evaluate(
        captures,
        variants,
        iterations=args.iterations,
        audio_dir=None if args.no_audio else output_dir / "audio",
    )
    write_results_csv(results, output_dir / "results.csv")
    summary = summary_markdown(results, source=args.source, commit=commit, problems=problems)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    written = [output_dir / "results.csv", output_dir / "summary.md"]
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(summary, encoding="utf-8")
        written.append(args.report)
    if args.source == "captures" and args.append_captures:
        if commit.endswith("-dirty") or commit == "unknown":
            parser.exit(1, "error: commit the code before recording results in the capture table\n")
        count = append_to_captures(results, args.captures, commit)
        print(f"recorded {count} result(s) in {args.captures}")
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
