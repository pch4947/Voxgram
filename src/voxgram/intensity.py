"""Mapping between spectral magnitude and stored grayscale.

Both directions of the mapping in docs/CARD_FORMAT.md live here, so the
encoder and the decoder cannot drift apart.

Two array layouts are used:

* grid:  (NUM_FREQ_BINS, NUM_FRAMES), row 0 is the lowest frequency bin.
* image: the same values as they are printed, row 0 is the top of the picture,
  so the lowest frequency bin is the bottom row.
"""

from __future__ import annotations

import numpy as np

from voxgram.format import DB_FLOOR, GRAY_WHITE, GRID_SHAPE

# Amplitude ratio that corresponds to DB_FLOOR.
_FLOOR_AMPLITUDE = 10.0 ** (DB_FLOOR / 20.0)

# Width of one gray level in dB.
DB_PER_GRAY_LEVEL = -DB_FLOOR / GRAY_WHITE


def _require_grid(array: np.ndarray, name: str) -> None:
    if array.shape != GRID_SHAPE:
        raise ValueError(f"{name} must have shape {GRID_SHAPE}, got {array.shape}")


def magnitude_to_gray(magnitude: np.ndarray) -> np.ndarray:
    """Encode STFT magnitude as stored grayscale values.

    Args:
        magnitude: (257, 376) non-negative float grid, any scale.

    Returns:
        (257, 376) uint8 grid ``q``. The peak maps to 0 (black) and everything
        at or below DB_FLOOR relative to the peak maps to 255 (white).

    Raises:
        ValueError: if the shape is wrong, values are negative or not finite,
            or the input is silent.
    """
    magnitude = np.asarray(magnitude, dtype=np.float64)
    _require_grid(magnitude, "magnitude")
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("magnitude contains NaN or infinite values")
    if np.any(magnitude < 0):
        raise ValueError("magnitude must be non-negative")

    peak = magnitude.max()
    if peak <= 0:
        raise ValueError("input is silent: the magnitude is zero everywhere")

    amplitude = np.maximum(magnitude / peak, _FLOOR_AMPLITUDE)
    decibels = 20.0 * np.log10(amplitude)  # in [DB_FLOOR, 0]
    gray = np.rint(GRAY_WHITE * decibels / DB_FLOOR)
    return gray.astype(np.uint8)


def gray_to_decibels(gray: np.ndarray) -> np.ndarray:
    """Decode grayscale values into amplitude dB relative to the peak.

    This is the exact inverse of the encoding formula, and the scale on which
    extraction accuracy is measured. Accepts fractional values, because a
    photographed and corrected card does not land on integer gray levels.

    Args:
        gray: (257, 376) grid ``p`` in [0, 255], integer or float.

    Returns:
        (257, 376) float64 grid in [DB_FLOOR, 0].

    Raises:
        ValueError: if the shape is wrong or values fall outside [0, 255].
    """
    gray = np.asarray(gray, dtype=np.float64)
    _require_grid(gray, "gray")
    if not np.all(np.isfinite(gray)):
        raise ValueError("gray contains NaN or infinite values")
    if gray.min() < 0 or gray.max() > GRAY_WHITE:
        raise ValueError("gray values must lie in [0, 255]")
    return DB_FLOOR * gray / GRAY_WHITE


def gray_to_magnitude(gray: np.ndarray) -> np.ndarray:
    """Decode grayscale values into normalized linear magnitude for synthesis.

    The floor amplitude is subtracted, so white decodes to silence. Without
    that, every white cell would carry DB_FLOOR of energy, and across 257 bins
    Griffin-Lim renders it as an audible broadband hiss. The subtraction is
    continuous, so paper white that lands slightly below 255 after
    calibration still decodes to almost nothing.

    Args:
        gray: (257, 376) grid ``p`` in [0, 255], integer or float.

    Returns:
        (257, 376) float64 grid in [0, 1]. Black is 1 and white is 0.

    Raises:
        ValueError: if the shape is wrong or values fall outside [0, 255].
    """
    amplitude = 10.0 ** (gray_to_decibels(gray) / 20.0)
    return (amplitude - _FLOOR_AMPLITUDE) / (1.0 - _FLOOR_AMPLITUDE)


def grid_to_image(grid: np.ndarray) -> np.ndarray:
    """Flip a grid so the lowest frequency is the bottom row of the picture."""
    _require_grid(grid, "grid")
    return np.ascontiguousarray(grid[::-1])


def image_to_grid(image: np.ndarray) -> np.ndarray:
    """Inverse of :func:`grid_to_image`."""
    _require_grid(image, "image")
    return np.ascontiguousarray(image[::-1])
