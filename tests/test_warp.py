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


def test_perspective_warp():
    """Verify inverse warping with genuine perspective distortion."""
    x_grid, y_grid = np.meshgrid(np.arange(80), np.arange(80))

    # Synthetic grayscale photograph with a known linear gradient.
    image = 2.0 * x_grid + 3.0 * y_grid + 10.0

    # Canonical output coordinates -> photograph coordinates.
    h = np.array(
        [
            [1.0, 0.1, 5.0],
            [0.05, 1.0, 4.0],
            [0.001, 0.002, 1.0],
        ]
    )

    result = warp_to_canonical(image, h, (40, 40))

    # Calculate expected values independently from the mapping.
    x, y = np.meshgrid(np.arange(40), np.arange(40))

    denominator = 0.001 * x + 0.002 * y + 1.0

    photo_x = (x + 0.1 * y + 5.0) / denominator
    photo_y = (0.05 * x + y + 4.0) / denominator

    expected = 2.0 * photo_x + 3.0 * photo_y + 10.0

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-12,
        atol=1e-10,
    )


def test_bilinear_out_of_bounds():
    """Out-of-bounds samples receive the fill value and invalid mask."""
    image = np.array(
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ]
    )

    x = np.array([0.0, 0.5, 1.0, -1.0, 2.0])
    y = np.array([0.0, 0.5, 1.0, 0.0, 1.0])

    values, mask = sample_bilinear(
        image,
        x,
        y,
        fill_value=-1.0,
        return_mask=True,
    )

    np.testing.assert_array_equal(
        values,
        [10.0, 25.0, 40.0, -1.0, -1.0],
    )

    np.testing.assert_array_equal(
        mask,
        [True, True, True, False, False],
    )


def test_warp_out_of_bounds():
    """Warping beyond photograph boundaries produces a validity mask."""
    image = np.arange(4, dtype=np.float64).reshape(2, 2)

    result, mask = warp_to_canonical(
        image,
        np.eye(3),
        (3, 3),
        fill_value=-1.0,
        return_mask=True,
    )

    expected = np.array(
        [
            [0.0, 1.0, -1.0],
            [2.0, 3.0, -1.0],
            [-1.0, -1.0, -1.0],
        ]
    )

    expected_mask = np.array(
        [
            [True, True, False],
            [True, True, False],
            [False, False, False],
        ]
    )

    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(mask, expected_mask)


def test_default_out_of_bounds_returns_nan():
    """Default invalid samples should remain distinguishable from black."""
    image = np.ones((3, 3))

    values, mask = sample_bilinear(
        image,
        np.array([1.0, 5.0]),
        np.array([1.0, 1.0]),
        return_mask=True,
    )

    assert values[0] == 1.0
    assert np.isnan(values[1])
    np.testing.assert_array_equal(mask, [True, False])


def test_perspective_round_trip():
    """A synthetic image survives forward and reverse warping."""
    x, y = np.meshgrid(np.arange(80), np.arange(80))

    # Smooth grayscale image with values approximately in [0, 1].
    original = 0.5 + 0.2 * np.sin(x / 12.0) + 0.2 * np.cos(y / 15.0)

    # Canonical -> photograph coordinate transformation.
    h = np.array(
        [
            [1.0, 0.015, 2.0],
            [-0.01, 1.0, 3.0],
            [0.00015, -0.00012, 1.0],
        ]
    )

    # Generate a synthetic photographed image.
    photographed = warp_to_canonical(
        original,
        np.linalg.inv(h),
        original.shape,
        fill_value=0.0,
    )

    # Reconstruct the canonical image.
    recovered, mask = warp_to_canonical(
        photographed,
        h,
        original.shape,
        fill_value=0.0,
        return_mask=True,
    )

    # Ignore boundaries affected by incomplete image coverage.
    interior = np.s_[10:-10, 10:-10]

    assert np.all(mask[interior])

    np.testing.assert_allclose(
        recovered[interior],
        original[interior],
        rtol=0.0,
        atol=0.02,
    )


def test_reject_invalid_output_shape():
    """Warping requires positive integer output dimensions."""
    image = np.ones((5, 5))
    h = np.eye(3)

    invalid_shapes = [
        (0, 5),
        (-1, 5),
        (5, 0),
        (5.5, 5),
        (5,),
    ]

    for shape in invalid_shapes:
        with np.testing.assert_raises(ValueError):
            warp_to_canonical(image, h, shape)
