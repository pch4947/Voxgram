"""Photometric calibration utilities for recovering encoded grayscale values."""

import numpy as np

from voxgram.format import CALIBRATION_PATCHES
from voxgram.geometry import apply_homography, sample_bilinear


def measure_calibration_patches(
    gray: np.ndarray,
    h: np.ndarray,
    *,
    samples_per_axis: int = 7,
    inset_fraction: float = 0.15,
) -> np.ndarray:
    """Measure median grayscale intensity inside each calibration patch.

    Args:
        gray: A 2D grayscale photograph.
        h: Homography mapping card millimetres to photograph pixels.
        samples_per_axis: Number of sample positions along each axis.
        inset_fraction: Fraction of each patch width and height excluded
            from each edge to avoid sampling patch boundaries.

    Returns:
        A (N,) float64 array containing measured patch intensities,
        in the same order as voxgram.format.CALIBRATION_PATCHES.

    Raises:
        ValueError: If parameters are invalid or a calibration patch
            cannot be sampled completely inside the photograph.
    """
    if not isinstance(samples_per_axis, (int, np.integer)):
        raise ValueError("samples_per_axis must be a positive integer")

    if samples_per_axis < 1:
        raise ValueError("samples_per_axis must be positive")

    if not np.isfinite(inset_fraction) or not 0 <= inset_fraction < 0.5:
        raise ValueError("inset_fraction must be in [0, 0.5)")

    # Sample positions lie inside the selected interior region.
    fractions = (np.arange(samples_per_axis) + 0.5) / samples_per_axis

    measurements = []

    for x0, y0, x1, y1, _level in CALIBRATION_PATCHES:
        width = x1 - x0
        height = y1 - y0

        if width <= 0 or height <= 0:
            raise ValueError("calibration patch has invalid dimensions")

        # Exclude patch boundaries to reduce edge contamination.
        interior_width = width * (1.0 - 2.0 * inset_fraction)
        interior_height = height * (1.0 - 2.0 * inset_fraction)

        x = x0 + width * inset_fraction + fractions * interior_width
        y = y0 + height * inset_fraction + fractions * interior_height

        # Generate sampling positions in canonical card millimetres.
        grid_x, grid_y = np.meshgrid(x, y)
        card_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

        # Transform card coordinates into photograph pixels.
        photo_points = apply_homography(h, card_points)

        # Use the bilinear sampler implemented in Issue #8.
        values, valid = sample_bilinear(
            gray, photo_points[:, 0], photo_points[:, 1], return_mask=True
        )

        if not np.all(valid):
            raise ValueError("calibration patch extends outside the photograph")

        # Median is robust to isolated noisy pixel measurements.
        measurements.append(float(np.median(values)))

    return np.asarray(measurements, dtype=np.float64)
