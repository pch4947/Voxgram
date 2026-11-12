"""Tests for the magnitude <-> grayscale mapping."""

import numpy as np
import pytest

from voxgram.format import DB_FLOOR, GRID_SHAPE
from voxgram.intensity import (
    DB_PER_GRAY_LEVEL,
    gray_to_decibels,
    gray_to_magnitude,
    grid_to_image,
    image_to_grid,
    magnitude_to_gray,
)


def random_magnitude(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Spread over about 80 dB so that some cells fall below the floor.
    return 10.0 ** rng.uniform(-4.0, 0.0, size=GRID_SHAPE)


def test_peak_is_black_and_floor_is_white():
    magnitude = np.full(GRID_SHAPE, 1e-6)
    magnitude[10, 20] = 5.0
    gray = magnitude_to_gray(magnitude)
    assert gray.dtype == np.uint8
    assert gray[10, 20] == 0
    assert gray[0, 0] == 255


def test_encoding_matches_the_specified_formula():
    magnitude = np.full(GRID_SHAPE, 1.0)
    magnitude[0, 0] = 10.0 ** (-30.0 / 20.0)  # -30 dB -> round(127.5) = 128
    magnitude[0, 1] = 10.0 ** (-20.0 / 20.0)  # -20 dB -> 85
    gray = magnitude_to_gray(magnitude)
    assert gray[0, 1] == 85
    assert gray[0, 0] in (127, 128)


def test_encoding_ignores_absolute_level():
    magnitude = random_magnitude()
    assert np.array_equal(magnitude_to_gray(magnitude), magnitude_to_gray(magnitude * 123.0))


def test_larger_magnitude_is_never_lighter():
    magnitude = np.full(GRID_SHAPE, 1.0)
    magnitude[0, :] = np.logspace(-4, 0, GRID_SHAPE[1])
    row = magnitude_to_gray(magnitude)[0].astype(int)
    assert np.all(np.diff(row) <= 0)


def test_round_trip_error_is_at_most_half_a_gray_level():
    magnitude = random_magnitude()
    normalized = magnitude / magnitude.max()
    decoded_db = gray_to_decibels(magnitude_to_gray(magnitude))

    above_floor = normalized > 10.0 ** (DB_FLOOR / 20.0)
    error_db = np.abs(20 * np.log10(normalized[above_floor]) - decoded_db[above_floor])
    assert error_db.max() <= DB_PER_GRAY_LEVEL / 2 + 1e-9
    assert np.all(decoded_db[~above_floor] == DB_FLOOR)


def test_decibel_decoding_range_and_fractional_values():
    gray = np.full(GRID_SHAPE, 127.5)
    gray[0, 0] = 0.0
    gray[0, 1] = 255.0
    decoded_db = gray_to_decibels(gray)
    assert decoded_db[0, 0] == 0.0
    assert decoded_db[0, 1] == DB_FLOOR
    assert decoded_db[1, 1] == pytest.approx(-30.0)


def test_white_decodes_to_silence_and_black_to_the_peak():
    gray = np.full(GRID_SHAPE, 255.0)
    gray[0, 0] = 0.0
    decoded = gray_to_magnitude(gray)
    assert decoded[0, 0] == pytest.approx(1.0)
    assert np.all(decoded[1:, :] == 0.0)


def test_magnitude_decoding_subtracts_the_floor():
    floor = 10.0 ** (DB_FLOOR / 20.0)
    gray = np.full(GRID_SHAPE, 127.5)  # -30 dB
    expected = (10.0 ** (-30.0 / 20.0) - floor) / (1.0 - floor)
    assert gray_to_magnitude(gray)[0, 0] == pytest.approx(expected)


def test_magnitude_decoding_is_monotonic_and_continuous_near_white():
    gray = np.full(GRID_SHAPE, 255.0)
    gray[0, :] = np.linspace(0.0, 255.0, GRID_SHAPE[1])
    row = gray_to_magnitude(gray)[0]
    assert np.all(np.diff(row) < 0)
    # Paper white a few levels short of 255 must still be far below the floor.
    almost_white = np.full(GRID_SHAPE, 250.0)
    assert gray_to_magnitude(almost_white)[0, 0] < 0.2 * 10.0 ** (DB_FLOOR / 20.0)


def test_silent_input_is_rejected():
    with pytest.raises(ValueError, match="silent"):
        magnitude_to_gray(np.zeros(GRID_SHAPE))


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_invalid_magnitude_is_rejected(bad):
    magnitude = np.ones(GRID_SHAPE)
    magnitude[3, 3] = bad
    with pytest.raises(ValueError):
        magnitude_to_gray(magnitude)


@pytest.mark.parametrize("bad", [-0.5, 255.5, np.nan])
def test_invalid_gray_is_rejected(bad):
    gray = np.zeros(GRID_SHAPE)
    gray[3, 3] = bad
    with pytest.raises(ValueError):
        gray_to_magnitude(gray)
    with pytest.raises(ValueError):
        gray_to_decibels(gray)


def test_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        magnitude_to_gray(np.ones((GRID_SHAPE[1], GRID_SHAPE[0])))
    with pytest.raises(ValueError, match="shape"):
        gray_to_magnitude(np.zeros((10, 10)))


def test_image_has_low_frequency_at_the_bottom():
    grid = np.zeros(GRID_SHAPE, dtype=np.uint8)
    grid[0, :] = 7  # lowest frequency bin
    image = grid_to_image(grid)
    assert np.all(image[-1, :] == 7)
    assert np.all(image[0, :] == 0)
    assert np.array_equal(image_to_grid(image), grid)
