"""Tests for planar homography estimation and point transformation."""

import numpy as np

from voxgram.geometry import apply_homography, estimate_homography

# Four canonical card coordinates in perimeter order.
CARD_POINTS = np.array(
    [
        [8.0, 7.0],
        [142.0, 7.0],
        [142.0, 93.0],
        [8.0, 93.0],
    ],
    dtype=np.float64,
)

# Known homography containing scaling, translation, rotation/shear, and perspective distortion.
KNOWN_H = np.array(
    [
        [8.0, -0.4, 125.0],
        [0.3, 7.5, 80.0],
        [0.0006, -0.0008, 1.0],
    ],
    dtype=np.float64,
)


def test_recover_known_homography():
    """Recover a known perspective homography from four correspondences."""
    photo_points = apply_homography(KNOWN_H, CARD_POINTS)

    estimated_h = estimate_homography(CARD_POINTS, photo_points)

    # Homographies are equivalent up to a nonzero scale factor.
    expected = KNOWN_H / KNOWN_H[2, 2]
    actual = estimated_h / estimated_h[2, 2]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


def test_transform_known_points():
    """Check the estimated homography on points beyond the four corners."""
    photo_points = apply_homography(KNOWN_H, CARD_POINTS)

    estimated_h = estimate_homography(CARD_POINTS, photo_points)

    interior_points = np.array(
        [
            [50.0, 40.0],
            [60.0, 80.0],
            [120.0, 10.0],
        ],
        dtype=np.float64,
    )

    expected = apply_homography(KNOWN_H, interior_points)
    actual = apply_homography(estimated_h, interior_points)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


def test_recover_large_coordinate_homography():
    """Check numerical stability with large photograph coordinates."""
    source = CARD_POINTS * 1e-3

    known_h = np.array(
        [
            [2000.0, 40.0, 1e6],
            [-25.0, 1800.0, -2e6],
            [0.005, 0.003, 1.0],
        ],
        dtype=np.float64,
    )

    destination = apply_homography(known_h, source)

    estimated_h = estimate_homography(source, destination)

    np.testing.assert_allclose(
        apply_homography(estimated_h, source),
        destination,
        rtol=1e-10,
        atol=1e-6,
    )


def test_identity_homography():
    """Applying an identity homography must preserve point coordinates."""
    points = np.array(
        [
            [0.0, 0.0],
            [10.0, 20.0],
            [-5.0, 12.0],
        ]
    )

    result = apply_homography(np.eye(3), points)

    np.testing.assert_allclose(result, points, atol=1e-12)


def test_empty_points():
    """Transforming an empty collection should return an empty array."""
    result = apply_homography(np.eye(3), np.empty((0, 2)))

    assert result.shape == (0, 2)
    assert result.dtype == np.float64
