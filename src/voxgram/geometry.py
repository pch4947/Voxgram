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


def estimate_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate the homography mapping source points to destination points.

    Uses normalized Direct Linear Transformation (DLT) with SVD.

    Args:
        src: Four source (x, y) points with shape (4, 2).
            For Voxgram, these are canonical card coordinates in mm.
        dst: Four corresponding destination (x, y) points with shape (4, 2).
            For Voxgram, these are photograph pixel coordinates.

    Returns:
        A (3, 3) float64 homography mapping src to dst.

    Raises:
        ValueError: If the input points are invalid or the linear system
            is degenerate.
    """
    # Validate both sets of point correspondences.
    src = _points4(src, "src")
    dst = _points4(dst, "dst")

    # Normalize coordinates to improve numerical stability.
    src_norm, t_src = _normalize_points(src)
    dst_norm, t_dst = _normalize_points(dst)

    # Build the 8x9 DLT matrix from four point correspondences.
    a = np.zeros((8, 9), dtype=np.float64)

    for i, ((x, y), (u, v)) in enumerate(zip(src_norm, dst_norm, strict=True)):
        a[2 * i] = [
            -x,
            -y,
            -1,
            0,
            0,
            0,
            u * x,
            u * y,
            u,
        ]

        a[2 * i + 1] = [
            0,
            0,
            0,
            -x,
            -y,
            -1,
            v * x,
            v * y,
            v,
        ]

    # Solve A @ h = 0 using singular value decomposition.
    _, singular_values, vt = np.linalg.svd(a, full_matrices=True)

    # Reject rank-deficient or numerically degenerate systems.
    if singular_values[-1] <= singular_values[0] * 1e-12:
        raise ValueError("point correspondences form a degenerate system")

    # The last right singular vector contains the homography entries.
    h_norm = vt[-1].reshape(3, 3)

    # Undo normalization: H = inv(T_dst) @ H_norm @ T_src.
    h = np.linalg.solve(t_dst, h_norm @ t_src)

    # Homographies are defined up to a nonzero scale.
    h /= np.linalg.norm(h)

    if abs(h[2, 2]) > 1e-12:
        h /= h[2, 2]

    return h


def apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Transform an array of 2D points using a homography matrix.

    Args:
        h: A (3, 3) homography matrix.
        points: An (N, 2) array of source (x, y) coordinates.

    Returns:
        An (N, 2) float64 array of transformed coordinates.

    Raises:
        ValueError: If the inputs have invalid shapes, contain non-finite
            values, or a point maps to infinity (zero denominator).
    """
    h = np.asarray(h, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)

    # Validate the homography matrix.
    if h.shape != (3, 3) or not np.all(np.isfinite(h)):
        raise ValueError("h must be a finite (3, 3) matrix")

    # Validate the points array.
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite coordinates")

    # Convert Cartesian coordinates to homogeneous coordinates.
    homogeneous = np.column_stack((points, np.ones(len(points))))

    # Apply the homography to all points.
    transformed = homogeneous @ h.T

    # Extract the homogeneous denominator.
    denominator = transformed[:, 2]

    # Use a scale-relative tolerance because H has arbitrary scale.
    row_scale = np.sum(np.abs(homogeneous) * np.abs(h[2]), axis=1)

    if np.any(np.abs(denominator) <= 1e-12 * row_scale):
        raise ValueError("homography maps a point to infinity (zero denominator)")

    # Convert back to Cartesian coordinates.
    result = transformed[:, :2] / denominator[:, None]

    if not np.all(np.isfinite(result)):
        raise ValueError("homography produced non-finite coordinates")

    return result
