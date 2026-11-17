"""Tests for grayscale conversion and thresholding."""

import numpy as np
import pytest

from voxgram.detection import (
    box_mean,
    default_window,
    integral_image,
    otsu_threshold,
    threshold_global,
    threshold_local_mean,
    to_grayscale,
)

# ---------------------------------------------------------------------------
# Grayscale conversion (#4)
# ---------------------------------------------------------------------------


def test_grayscale_of_gray_image_is_unchanged_float():
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    gray = to_grayscale(image)
    assert gray.dtype == np.float64
    np.testing.assert_array_equal(gray, image)


def test_grayscale_uses_bt601_weights_in_bgr_order():
    image = np.zeros((1, 3, 3), dtype=np.uint8)
    image[0, 0] = (255, 0, 0)  # blue
    image[0, 1] = (0, 255, 0)  # green
    image[0, 2] = (0, 0, 255)  # red
    np.testing.assert_allclose(to_grayscale(image)[0], [0.114 * 255, 0.587 * 255, 0.299 * 255])


def test_grayscale_of_neutral_colour_keeps_its_level_and_ignores_alpha():
    image = np.full((2, 2, 4), 200, dtype=np.uint8)
    image[..., 3] = 0
    np.testing.assert_allclose(to_grayscale(image), 200.0)


def test_grayscale_scales_uint16_to_0_255():
    image = np.array([[0, 65535]], dtype=np.uint16)
    np.testing.assert_allclose(to_grayscale(image), [[0.0, 255.0]])


@pytest.mark.parametrize(
    "image", [np.zeros((2, 2, 2), np.uint8), np.zeros(5, np.uint8), np.zeros((2, 2), np.int32)]
)
def test_grayscale_rejects_unsupported_input(image):
    with pytest.raises(ValueError):
        to_grayscale(image)


# ---------------------------------------------------------------------------
# Thresholding (#4)
# ---------------------------------------------------------------------------


def test_global_threshold_marks_strictly_darker_pixels():
    gray = np.array([[0.0, 99.9, 100.0, 255.0]])
    np.testing.assert_array_equal(threshold_global(gray, 100.0), [[True, True, False, False]])


def test_otsu_separates_two_levels():
    gray = np.full((10, 10), 200.0)
    gray[:4] = 30.0
    t = otsu_threshold(gray)
    assert 30 < t <= 200
    np.testing.assert_array_equal(threshold_global(gray, t), gray == 30.0)


def test_integral_image_matches_direct_sums():
    gray = np.random.default_rng(0).uniform(0, 255, size=(7, 9))
    table = integral_image(gray)
    assert table.shape == (8, 10)
    assert np.all(table[0] == 0) and np.all(table[:, 0] == 0)
    for r0, r1, c0, c1 in [(0, 7, 0, 9), (2, 5, 3, 8), (6, 7, 0, 1)]:
        box = table[r1, c1] - table[r0, c1] - table[r1, c0] + table[r0, c0]
        assert box == pytest.approx(gray[r0:r1, c0:c1].sum())


def test_box_mean_matches_direct_mean_including_borders():
    gray = np.random.default_rng(1).uniform(0, 255, size=(11, 13))
    window = 5
    mean = box_mean(gray, window)
    for r in range(11):
        for c in range(13):
            block = gray[max(r - 2, 0) : r + 3, max(c - 2, 0) : c + 3]
            assert mean[r, c] == pytest.approx(block.mean())


@pytest.mark.parametrize("window", [0, 4, -3, 2.5])
def test_box_mean_rejects_bad_window(window):
    with pytest.raises(ValueError):
        box_mean(np.zeros((5, 5)), window)


def test_default_window_is_odd_and_scales_with_the_image():
    assert default_window((3000, 4000)) == 499
    assert default_window((1000, 1000)) == 125
    assert default_window((20, 30)) == 15


def gradient_scene():
    """Dark squares on paper under illumination falling to 20 % at the left."""
    height, width = 120, 400
    reflectance = np.full((height, width), 0.9)
    truth = np.zeros((height, width), dtype=bool)
    for left in range(20, width - 30, 60):
        truth[40:70, left : left + 30] = True
    reflectance[truth] = 0.3
    illumination = np.linspace(0.2, 1.0, width)[np.newaxis, :]
    return 255.0 * reflectance * illumination, truth


def test_strong_gradient_defeats_every_global_threshold():
    gray, truth = gradient_scene()
    best_error = min(np.count_nonzero(threshold_global(gray, t) != truth) for t in range(256))
    # The darkest paper is darker than the brightest ink, so no level separates them.
    assert best_error > 0.05 * truth.size


def test_strong_gradient_is_handled_by_the_local_mean_threshold():
    gray, truth = gradient_scene()
    # Near the image border the window is cut off on one side, so under a
    # gradient the mean there is biased by up to about 10 gray levels. The
    # offset absorbs that; ink is still more than 30 levels below the mean.
    binary = threshold_local_mean(gray, window=81, offset=15.0)
    np.testing.assert_array_equal(binary, truth)


def test_local_mean_keeps_blank_paper_noise_out_with_an_offset():
    gray = 230.0 + np.random.default_rng(2).normal(0.0, 2.0, size=(60, 60))
    assert not threshold_local_mean(gray, window=15, offset=10.0).any()
