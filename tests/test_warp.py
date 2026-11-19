"""Tests for bilinear sampling and inverse homography warping."""

import numpy as np

from voxgram.geometry import sample_bilinear, warp_to_canonical


def test_bilinear_sampling_at_integer_coordinates():
    """Sampling at pixel centers must recover exact pixel values."""
    image = np.arange(20, dtype=np.float64).reshape(4, 5)

    x, y = np.meshgrid(np.arange(5), np.arange(4))

    sampled = sample_bilinear(image, x, y)

    np.testing.assert_array_equal(sampled, image)


def test_bilinear_sampling_at_fractional_coordinates():
    """Bilinear interpolation should weight surrounding pixels."""
    image = np.array(
        [
            [0.0, 100.0],
            [150.0, 250.0],
        ]
    )

    sampled = sample_bilinear(image, 0.5, 0.5)

    np.testing.assert_allclose(sampled, 125.0, atol=1e-12)


def test_bilinear_sampling_preserves_linear_gradient():
    """Bilinear interpolation reproduces a linear intensity gradient."""
    x_grid, y_grid = np.meshgrid(np.arange(8), np.arange(6))
    image = 2.0 * x_grid + 3.0 * y_grid + 5.0

    x = np.array([1.25, 2.5, 4.75])
    y = np.array([0.5, 3.25, 2.75])

    expected = 2.0 * x + 3.0 * y + 5.0
    actual = sample_bilinear(image, x, y)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_identity_warp():
    """An identity homography must reproduce the input image."""
    image = np.arange(30, dtype=np.float64).reshape(5, 6)

    result = warp_to_canonical(image, np.eye(3), image.shape)

    np.testing.assert_array_equal(result, image)


def test_translation_warp():
    """Inverse warping should sample the correct translated region."""
    image = np.arange(25, dtype=np.float64).reshape(5, 5)

    h = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    result = warp_to_canonical(image, h, (3, 3))

    expected = image[1:4, 1:4]

    np.testing.assert_array_equal(result, expected)


def test_fractional_translation_warp():
    """A fractional translation should use bilinear interpolation."""
    x_grid, y_grid = np.meshgrid(np.arange(6), np.arange(6))
    image = 10.0 * x_grid + 20.0 * y_grid

    h = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.25],
            [0.0, 0.0, 1.0],
        ]
    )

    result = warp_to_canonical(image, h, (4, 4))

    out_x, out_y = np.meshgrid(np.arange(4), np.arange(4))
    expected = 10.0 * (out_x + 0.5) + 20.0 * (out_y + 0.25)

    np.testing.assert_allclose(result, expected, atol=1e-12)
