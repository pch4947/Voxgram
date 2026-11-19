"""Geometry utilities for planar homography estimation."""

import numpy as np


def _points4(points: np.ndarray, name: str) -> np.ndarray:
    """Validate four points forming a convex quadrilateral."""
    points = np.asarray(points, dtype=np.float64)

    if points.shape != (4, 2):
        raise ValueError(f"{name} must have shape (4, 2)")

    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite coordinates")

    # Normalize scale for numerical comparisons.
    scale = np.max(np.ptp(points, axis=0))

    if scale == 0:
        raise ValueError(f"{name} contains repeated points")

    normalized = points / scale

    # Check for repeated or nearly identical points.
    differences = normalized[:, None, :] - normalized[None, :, :]
    distances = np.sum(differences**2, axis=-1)
    np.fill_diagonal(distances, np.inf)

    if np.any(distances < 1e-24):
        raise ValueError(f"{name} contains repeated points")

    # Check that all corners make turns in the same direction.
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)

    if np.any(np.abs(cross) < 1e-12) or not (np.all(cross > 0) or np.all(cross < 0)):
        raise ValueError(f"{name} must form a convex quadrilateral")

    return points


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize points to zero centroid and mean distance sqrt(2)."""
    centroid = np.mean(points, axis=0)
    shifted = points - centroid

    distances = np.linalg.norm(shifted, axis=1)
    mean_distance = np.mean(distances)

    if not np.isfinite(mean_distance) or mean_distance <= 0:
        raise ValueError("cannot normalize coincident points")

    scale = np.sqrt(2.0) / mean_distance

    transform = np.array(
        [
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    normalized = shifted * scale

    return normalized, transform
