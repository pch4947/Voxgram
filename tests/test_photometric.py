"""Tests for photometric calibration and grayscale recovery."""

import numpy as np
import pytest

from voxgram.card import patch_levels
from voxgram.format import CALIBRATION_PATCHES
from voxgram.photometric import (
    apply_tone_curve,
    calibrate_samples,
    fit_tone_curve,
    interpolate_reference_fields,
    measure_calibration_patches,
    normalize_brightness,
)


def make_synthetic_card(
    encoded_value: float = 128.0,
    *,
    gradient: bool = False,
    gamma: float = 1.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic photograph with known calibration patches.

    Encoded grayscale values undergo a gamma response, followed by
    a black-level offset and optional spatial illumination gradient.
    """
    scale = 4
    height, width = 410, 610

    pixel_y, pixel_x = np.indices((height, width))
    x_mm = pixel_x / scale
    y_mm = pixel_y / scale

    if gradient:
        black = 20.0 + 0.04 * x_mm + 0.02 * y_mm
    else:
        black = np.full((height, width), 20.0)

    contrast = 200.0

    def photograph_response(encoded):
        return black + contrast * (encoded / 255.0) ** gamma

    # Fill the card with a known encoded grayscale intensity.
    image = photograph_response(encoded_value)

    # Render calibration patches using the shared card specification.
    for x0, y0, x1, y1, level in CALIBRATION_PATCHES:
        left = round(x0 * scale)
        top = round(y0 * scale)
        right = round(x1 * scale)
        bottom = round(y1 * scale)

        image[top:bottom, left:right] = photograph_response(level)[top:bottom, left:right]

    # Canonical card millimetres -> photograph pixels.
    h = np.diag([scale, scale, 1.0])

    return image, h


def test_measure_calibration_patches():
    """Patch measurements recover known values under uniform lighting."""
    image, h = make_synthetic_card(
        gradient=False,
        gamma=1.0,
    )

    measured = measure_calibration_patches(image, h)
    levels = patch_levels()

    expected = 20.0 + 200.0 * levels / 255.0

    assert measured.shape == levels.shape

    np.testing.assert_allclose(
        measured,
        expected,
        rtol=0,
        atol=1e-10,
    )


def test_constant_reference_fields():
    """Constant calibration references produce constant fields."""
    levels = patch_levels()

    measurements = np.full(len(levels), 120.0)
    measurements[levels == 0] = 20.0
    measurements[levels == 255] = 220.0

    x, y = np.meshgrid(
        [20.0, 75.0, 130.0],
        [20.0, 50.0, 80.0],
    )

    black, white = interpolate_reference_fields(measurements, x, y)

    np.testing.assert_allclose(black, 20.0, atol=1e-10)
    np.testing.assert_allclose(white, 220.0, atol=1e-10)


def test_local_brightness_normalization():
    """Normalization accounts for different black and white levels."""
    image = np.array([[60.0, 120.0, 180.0]])

    black = np.array([[10.0, 20.0, 30.0]])
    white = np.array([[210.0, 220.0, 230.0]])

    corrected, valid = normalize_brightness(
        image,
        black,
        white,
        return_mask=True,
    )

    expected = np.array([[63.75, 127.5, 191.25]])

    np.testing.assert_allclose(corrected, expected, atol=1e-12)
    assert np.all(valid)


def test_recover_known_tone_curve():
    """Inverse tone fitting recovers known nonlinear patch intensities."""
    levels = patch_levels().astype(np.float64)

    # Known nonlinear camera response with black and white offsets.
    measured = 20.0 + 200.0 * (levels / 255.0) ** 1.4

    measured_knots, encoded_knots = fit_tone_curve(measured)

    normalized = 255.0 * (measured - 20.0) / 200.0

    recovered = apply_tone_curve(
        normalized,
        measured_knots,
        encoded_knots,
    )

    np.testing.assert_allclose(
        recovered,
        levels,
        rtol=0,
        atol=1e-8,
    )

    assert np.all(np.diff(measured_knots) > 0)
    assert np.all(np.diff(encoded_knots) >= 0)


def test_full_calibration_uniform_lighting():
    """Recover encoded intensities after a nonlinear camera response."""
    image, h = make_synthetic_card(
        encoded_value=128.0,
        gradient=False,
    )

    x = np.array([60.0, 75.0, 90.0])
    y = np.array([40.0, 50.0, 60.0])

    corrected, valid = calibrate_samples(image, h, x, y, return_mask=True)

    assert np.all(valid)

    np.testing.assert_allclose(
        corrected,
        128.0,
        rtol=0,
        atol=1.0,
    )


def test_full_calibration_illumination_gradient():
    """Recover encoded values despite smooth spatial illumination changes."""
    image, h = make_synthetic_card(
        encoded_value=128.0,
        gradient=True,
    )

    x = np.array([40.0, 75.0, 110.0])
    y = np.array([30.0, 50.0, 70.0])

    corrected, valid = calibrate_samples(image, h, x, y, return_mask=True)

    assert np.all(valid)

    # Allow interpolation error from the spatial reference fields.
    np.testing.assert_allclose(
        corrected,
        128.0,
        rtol=0,
        atol=5.0,
    )


def test_insufficient_contrast_is_invalid():
    """Regions without enough contrast must not be normalized."""
    image = np.array([[100.0, 120.0]])

    black = np.array([[99.8, 20.0]])
    white = np.array([[100.2, 220.0]])

    corrected, valid = normalize_brightness(
        image,
        black,
        white,
        return_mask=True,
    )

    assert not valid[0, 0]
    assert np.isnan(corrected[0, 0])

    assert valid[0, 1]
    np.testing.assert_allclose(corrected[0, 1], 127.5)


def test_saturated_sample_is_flagged_when_requested():
    """A completely clipped measurement is marked as unreliable."""
    image, h = make_synthetic_card()

    # Place an intensity of 255 at an interior sampling location.
    image[196:204, 296:304] = 255.0

    corrected, valid = calibrate_samples(
        image,
        h,
        np.array([75.0]),
        np.array([50.0]),
        reject_saturated=True,
        return_mask=True,
    )

    assert not valid[0]
    assert np.isnan(corrected[0])


def test_incomplete_calibration_is_rejected():
    """Calibration cannot succeed when required patches are outside the image."""
    image = np.full((100, 100), 128.0)

    h = np.diag([4.0, 4.0, 1.0])

    with pytest.raises(ValueError):
        measure_calibration_patches(image, h)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (0.0, 0.0),
        (255.0, 255.0),
    ],
)
def test_grayscale_endpoints_remain_valid_by_default(raw_value, expected):
    """Valid black and white pixels must not become NaN."""
    image, h = make_synthetic_card()

    # Replace the center sampling region with a grayscale endpoint.
    image[196:204, 296:304] = raw_value

    corrected, valid = calibrate_samples(
        image,
        h,
        np.array([75.0]),
        np.array([50.0]),
        return_mask=True,
    )

    assert valid[0]
    assert np.isfinite(corrected[0])

    np.testing.assert_allclose(
        corrected[0],
        expected,
        rtol=0,
        atol=1e-10,
    )
