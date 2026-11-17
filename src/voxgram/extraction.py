"""Sample the corrected data region onto the 257 x 376 grid.

This is the extraction stage of the decoder. It takes the corrected card, in
canonical card coordinates, and the rectangle of the data region inside it,
and produces the grid of recovered gray values ``p``, then the magnitude.

Coordinate conventions
----------------------
* Canonical card coordinates are continuous. Pixel (row ``r``, column ``c``) of
  the corrected image covers ``[c, c + 1) x [r, r + 1)``, so its centre is at
  ``(c + 0.5, r + 0.5)``. ``x`` runs along the columns and ``y`` along the rows.
* The data region is an edge-to-edge rectangle in these coordinates. It is
  split into 257 rows and 376 columns of equal cells.
* Row 0 of the sampled picture is the top of the printed spectrogram, which is
  the highest frequency. :func:`extract_gray` keeps this image layout.
  :func:`extract_magnitude` returns the grid layout, where row 0 is the lowest
  frequency, by calling :func:`voxgram.intensity.image_to_grid`.
* A *sampler* is a function ``sampler(image, x, y)`` that returns the image
  value at the fractional positions ``x`` and ``y``, which are given in pixel
  *index* coordinates: pixel centres are at integers, as in ``cv2.remap``.
  This module subtracts the half pixel before calling it.

The sampler defaults to a small bilinear sampler kept in this module. The one
from issue #8 (``voxgram.geometry``) can be passed with ``sampler=``, and then
the local one can be deleted.

Library boundary: the sampling and the smoothing are written with NumPy only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import comb

import numpy as np

from voxgram.format import GRAY_BLACK, GRAY_WHITE, GRID_SHAPE, NUM_FRAMES, NUM_FREQ_BINS
from voxgram.intensity import gray_to_magnitude, image_to_grid

Sampler = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

# Averaging window as a fraction of the cell width and height. Samples are
# spread over the middle of the cell, away from the borders where a halftone
# or a neighbouring cell leaks in.
DEFAULT_CELL_FRACTION = 0.5

_EPS = 1e-9


@dataclass(frozen=True)
class DataRegion:
    """Rectangle of the data region in canonical card coordinates.

    ``left`` and ``top`` are the coordinates of the top-left corner, ``width``
    and ``height`` the size, all in the unit of the corrected image pixels.
    """

    left: float
    top: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.width, self.height)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("data region values must be finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("data region width and height must be positive")


def cell_centres(region: DataRegion) -> tuple[np.ndarray, np.ndarray]:
    """Positions of the centre of every cell, in canonical card coordinates.

    Returns:
        ``(x, y)``, two float64 arrays of shape (257, 376). Row 0 is the top row
        of the printed spectrogram.
    """
    cell_width = region.width / NUM_FRAMES
    cell_height = region.height / NUM_FREQ_BINS
    columns = region.left + (np.arange(NUM_FRAMES) + 0.5) * cell_width
    rows = region.top + (np.arange(NUM_FREQ_BINS) + 0.5) * cell_height
    x, y = np.meshgrid(columns, rows)
    return x, y


def bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear interpolation at fractional index coordinates.

    Positions outside the image are clamped to its border.

    Temporary: replace by the sampler of issue #8 once it is merged.
    """
    height, width = image.shape
    x = np.clip(x, 0.0, width - 1.0)
    y = np.clip(y, 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = x - x0
    fy = y - y0
    upper = image[y0, x0] * (1.0 - fx) + image[y0, x1] * fx
    lower = image[y1, x0] * (1.0 - fx) + image[y1, x1] * fx
    return upper * (1.0 - fy) + lower * fy


def box_kernel(size: int) -> np.ndarray:
    """Normalized 1-D box (moving average) kernel of odd length."""
    _require_odd_size(size)
    return np.full(size, 1.0 / size)


def binomial_kernel(size: int) -> np.ndarray:
    """Normalized 1-D binomial kernel of odd length, ``[1, 2, 1] / 4`` for 3.

    A binomial kernel is the discrete approximation of a Gaussian.
    """
    _require_odd_size(size)
    kernel = np.array([comb(size - 1, k) for k in range(size)], dtype=np.float64)
    return kernel / kernel.sum()


def smooth_separable(grid: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Smooth a 2-D array with the same 1-D kernel along rows and columns.

    The convolution is written as a sum of shifted copies. The border is
    extended by repeating the edge value, so the output keeps the input shape
    and a constant array stays constant.

    Args:
        grid: 2-D array.
        kernel: odd-length, symmetric, non-negative 1-D kernel that sums to 1.
            Symmetry makes convolution and correlation the same.

    Raises:
        ValueError: if the kernel does not meet these conditions.
    """
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 2:
        raise ValueError(f"grid must be 2-D, got {grid.ndim}-D")
    kernel = _check_kernel(kernel)
    radius = kernel.size // 2
    rows, columns = grid.shape

    padded = np.pad(grid, ((0, 0), (radius, radius)), mode="edge")
    horizontal = np.zeros_like(grid)
    for offset, weight in enumerate(kernel):
        horizontal += weight * padded[:, offset : offset + columns]

    padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode="edge")
    result = np.zeros_like(grid)
    for offset, weight in enumerate(kernel):
        result += weight * padded[offset : offset + rows, :]
    return result


def extract_gray(
    corrected: np.ndarray,
    region: DataRegion,
    *,
    samples_per_axis: int = 1,
    cell_fraction: float = DEFAULT_CELL_FRACTION,
    smoothing: np.ndarray | None = None,
    sampler: Sampler | None = None,
) -> np.ndarray:
    """Recover the gray values ``p`` of every cell, in image layout.

    Args:
        corrected: 2-D corrected card (rectified and brightness-calibrated), in
            canonical card coordinates. Any numeric dtype; values may leave
            [0, 255] after calibration and are clipped at the end.
        region: the data region inside ``corrected``. It must lie inside the
            image.
        samples_per_axis: ``n``. Each cell is sampled at ``n x n`` points and
            the values are averaged. ``1`` samples only the cell centre.
        cell_fraction: the ``n x n`` points are spread over this fraction of the
            cell width and height, centred on the cell. Ignored when ``n`` is 1.
        smoothing: optional 1-D kernel (see :func:`smooth_separable`), applied
            to the grid after sampling. ``None`` (the default) means no
            smoothing.
        sampler: bilinear sampler, ``sampler(image, x, y)`` in index
            coordinates. Defaults to :func:`bilinear_sample`.

    Returns:
        (257, 376) float64 grid in [0, 255], not rounded. Row 0 is the top of
        the printed spectrogram.

    Raises:
        ValueError: for a bad image, region, sampling parameter or kernel.
    """
    image = _as_float_image(corrected)
    _require_region_inside(region, image.shape)
    _require_sampling(samples_per_axis, cell_fraction)
    sample = bilinear_sample if sampler is None else sampler

    centre_x, centre_y = cell_centres(region)
    cell_width = region.width / NUM_FRAMES
    cell_height = region.height / NUM_FREQ_BINS
    offsets = ((np.arange(samples_per_axis) + 0.5) / samples_per_axis - 0.5) * cell_fraction

    total = np.zeros(GRID_SHAPE)
    for offset_y in offsets:
        for offset_x in offsets:
            # Edge-based card coordinates to pixel index coordinates: -0.5.
            x = centre_x + offset_x * cell_width - 0.5
            y = centre_y + offset_y * cell_height - 0.5
            values = np.asarray(sample(image, x, y), dtype=np.float64)
            if values.shape != GRID_SHAPE:
                raise ValueError(f"sampler returned shape {values.shape}, expected {GRID_SHAPE}")
            total += values
    gray = total / (samples_per_axis * samples_per_axis)

    if smoothing is not None:
        gray = smooth_separable(gray, smoothing)
    return np.clip(gray, GRAY_BLACK, GRAY_WHITE)


def extract_magnitude(
    corrected: np.ndarray,
    region: DataRegion,
    *,
    samples_per_axis: int = 1,
    cell_fraction: float = DEFAULT_CELL_FRACTION,
    smoothing: np.ndarray | None = None,
    sampler: Sampler | None = None,
) -> np.ndarray:
    """Recover the normalized magnitude, in grid layout.

    Same arguments as :func:`extract_gray`. The gray values are flipped to grid
    layout and converted with :func:`voxgram.intensity.gray_to_magnitude`.

    Returns:
        (257, 376) float64 magnitude grid. Row 0 is the lowest frequency.
    """
    gray = extract_gray(
        corrected,
        region,
        samples_per_axis=samples_per_axis,
        cell_fraction=cell_fraction,
        smoothing=smoothing,
        sampler=sampler,
    )
    return gray_to_magnitude(image_to_grid(gray))


def _as_float_image(corrected: np.ndarray) -> np.ndarray:
    image = np.asarray(corrected, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"corrected card must be a 2-D grayscale image, got {image.ndim}-D")
    if image.size == 0:
        raise ValueError("corrected card is empty")
    if not np.all(np.isfinite(image)):
        raise ValueError("corrected card contains NaN or infinite values")
    return image


def _require_region_inside(region: DataRegion, shape: tuple[int, int]) -> None:
    height, width = shape
    inside = (
        region.left >= -_EPS
        and region.top >= -_EPS
        and region.left + region.width <= width + _EPS
        and region.top + region.height <= height + _EPS
    )
    if not inside:
        raise ValueError(
            f"data region {region} does not lie inside the corrected card ({width} x {height})"
        )


def _require_sampling(samples_per_axis: int, cell_fraction: float) -> None:
    if samples_per_axis < 1 or int(samples_per_axis) != samples_per_axis:
        raise ValueError("samples_per_axis must be a positive integer")
    if not 0.0 < cell_fraction <= 1.0:
        raise ValueError("cell_fraction must be in (0, 1]")


def _require_odd_size(size: int) -> None:
    if size < 1 or size % 2 == 0:
        raise ValueError("kernel size must be a positive odd integer")


def _check_kernel(kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 1 or kernel.size % 2 == 0:
        raise ValueError("kernel must be 1-D with an odd length")
    if not np.all(np.isfinite(kernel)) or np.any(kernel < 0):
        raise ValueError("kernel must be finite and non-negative")
    if not np.isclose(kernel.sum(), 1.0):
        raise ValueError("kernel must sum to 1")
    if not np.allclose(kernel, kernel[::-1]):
        raise ValueError("kernel must be symmetric")
    return kernel
