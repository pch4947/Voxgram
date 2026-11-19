"""Photometric calibration utilities for recovering encoded grayscale values."""

import numpy as np

from voxgram.card import patch_centers_mm, patch_levels
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


def _interpolate_reference(
    positions: np.ndarray,
    centers: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Interpolate reference intensities using inverse distance weighting.

    Distances are measured in canonical card millimetres.
    Samples exactly at a reference center return its measured value.
    """
    # Squared distances from every query point to every reference.
    differences = positions[:, None, :] - centers[None, :, :]
    distance_sq = np.sum(differences**2, axis=2)

    # Avoid division by zero for points at reference centers.
    weights = 1.0 / np.maximum(distance_sq, 1e-12)

    weighted_sum = weights @ values
    weight_sum = np.sum(weights, axis=1)

    interpolated = weighted_sum / weight_sum

    # Preserve exact measured values at calibration centers.
    at_center = distance_sq == 0.0

    if np.any(at_center):
        rows = np.any(at_center, axis=1)
        indices = np.argmax(at_center[rows], axis=1)
        interpolated[rows] = values[indices]

    return interpolated


def interpolate_reference_fields(
    measurements: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate photographed black and white reference fields.

    Args:
        measurements: Measured intensities for all calibration patches,
            in the order defined by CALIBRATION_PATCHES.
        x: Horizontal coordinates in canonical card millimetres.
        y: Vertical coordinates in canonical card millimetres.

    Returns:
        Two float64 arrays (black_field, white_field), with the
        broadcast shape of x and y.

    Raises:
        ValueError: If measurements or coordinates are invalid.
    """
    measurements = np.asarray(measurements, dtype=np.float64)

    centers = patch_centers_mm()
    levels = patch_levels()

    if measurements.shape != (len(levels),):
        raise ValueError("measurements must contain one value per calibration patch")

    if not np.all(np.isfinite(measurements)):
        raise ValueError("measurements must contain finite values")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    try:
        x, y = np.broadcast_arrays(x, y)
    except ValueError as exc:
        raise ValueError("x and y must have compatible shapes") from exc

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("coordinates must contain finite values")

    # Select only the black and white reference patches.
    black_mask = levels == 0
    white_mask = levels == 255

    if not np.any(black_mask) or not np.any(white_mask):
        raise ValueError("black and white reference patches are required")

    # Convert the query coordinates into an (N, 2) point array.
    original_shape = x.shape
    positions = np.column_stack((x.ravel(), y.ravel()))

    # Interpolate black and white independently.
    black_field = _interpolate_reference(positions, centers[black_mask], measurements[black_mask])
    white_field = _interpolate_reference(positions, centers[white_mask], measurements[white_mask])

    return (black_field.reshape(original_shape), white_field.reshape(original_shape))


def normalize_brightness(
    image: np.ndarray,
    black_field: np.ndarray,
    white_field: np.ndarray,
    *,
    min_contrast: float = 1.0,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Normalize grayscale intensities using local black/white references.

    Args:
        image: A 2D array of photographed grayscale intensities.
        black_field: Local photographed black reference values.
        white_field: Local photographed white reference values.
        min_contrast: Minimum acceptable difference between white
            and black references, in grayscale intensity units.
        return_mask: Whether to also return a boolean validity mask.

    Returns:
        A float64 image normalized to the encoded 0-255 scale.
        Invalid positions are assigned NaN.

        If return_mask is True, also returns a boolean array
        identifying positions with sufficient local contrast.

    Raises:
        ValueError: If image data, reference fields, or parameters
            have invalid shapes or contain non-finite values.
    """
    image = np.asarray(image, dtype=np.float64)
    black_field = np.asarray(black_field, dtype=np.float64)
    white_field = np.asarray(white_field, dtype=np.float64)

    if image.ndim != 2:
        raise ValueError("image must be a 2D grayscale array")

    if not np.all(np.isfinite(image)):
        raise ValueError("image must contain finite values")

    if not np.isfinite(min_contrast) or min_contrast <= 0:
        raise ValueError("min_contrast must be positive and finite")

    # Broadcast reference fields to the image dimensions.
    try:
        black_field = np.broadcast_to(black_field, image.shape)
        white_field = np.broadcast_to(white_field, image.shape)
    except ValueError as exc:
        raise ValueError("reference fields must be compatible with image shape") from exc

    if not np.all(np.isfinite(black_field)):
        raise ValueError("black reference must contain finite values")

    if not np.all(np.isfinite(white_field)):
        raise ValueError("white reference must contain finite values")

    # Calculate the local grayscale contrast.
    contrast = white_field - black_field

    # Reject locations where contrast is too small or reversed.
    valid = contrast >= min_contrast

    # Allocate output with NaN for invalid locations.
    normalized = np.full(image.shape, np.nan, dtype=np.float64)

    # Avoid division at invalid positions.
    np.divide(
        255.0 * (image - black_field),
        contrast,
        out=normalized,
        where=valid,
    )

    if return_mask:
        return normalized, valid

    return normalized
