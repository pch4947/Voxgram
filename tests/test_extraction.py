"""Tests for sampling the data region onto the 257 x 376 grid."""

import numpy as np
import pytest

from voxgram import extraction
from voxgram.extraction import (
    DataRegion,
    binomial_kernel,
    box_kernel,
    cell_centres,
    extract_gray,
    extract_magnitude,
    smooth_separable,
)
from voxgram.format import GRID_SHAPE
from voxgram.intensity import gray_to_magnitude, image_to_grid

ROWS, COLUMNS = GRID_SHAPE


def smooth_spectrogram() -> np.ndarray:
    """A smooth synthetic spectrogram image in [27.5, 227.5]."""
    rows, columns = np.mgrid[0:ROWS, 0:COLUMNS]
    return 127.5 + 100.0 * np.sin(columns / 15.0) * np.cos(rows / 20.0)


def random_spectrogram(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.0, 255.0, size=GRID_SHAPE)


def replicate(image: np.ndarray, factor: int) -> np.ndarray:
    """Upscale by an integer factor: every cell becomes a factor x factor block."""
    return np.kron(image, np.ones((factor, factor)))


def full_region(factor: int) -> DataRegion:
    return DataRegion(0.0, 0.0, COLUMNS * factor, ROWS * factor)


def linear_upscale(image: np.ndarray, factor: int) -> np.ndarray:
    """Upscale by linear interpolation with aligned pixel centres."""

    def along_axis(values: np.ndarray, axis: int) -> np.ndarray:
        size = values.shape[axis]
        new_index = np.arange(size * factor)
        position = np.clip((new_index + 0.5) / factor - 0.5, 0.0, size - 1.0)
        low = np.floor(position).astype(int)
        high = np.minimum(low + 1, size - 1)
        fraction = position - low
        shape = [1, 1]
        shape[axis] = -1
        fraction = fraction.reshape(shape)
        return (
            np.take(values, low, axis=axis) * (1 - fraction)
            + np.take(values, high, axis=axis) * fraction
        )

    return along_axis(along_axis(image, 0), 1)


# --- sampling positions ---------------------------------------------------


def test_cell_centres_shape_and_extremes():
    region = DataRegion(left=10.0, top=20.0, width=376.0, height=257.0)
    x, y = cell_centres(region)
    assert x.shape == GRID_SHAPE
    assert y.shape == GRID_SHAPE
    # One card unit per cell here, so the first centre is half a cell in.
    assert x[0, 0] == pytest.approx(10.5)
    assert y[0, 0] == pytest.approx(20.5)
    assert x[-1, -1] == pytest.approx(10.0 + 376.0 - 0.5)
    assert y[-1, -1] == pytest.approx(20.0 + 257.0 - 0.5)


def test_cell_centres_are_evenly_spaced_and_aligned_to_the_axes():
    region = DataRegion(left=5.0, top=7.0, width=752.0, height=514.0)
    x, y = cell_centres(region)
    assert np.allclose(np.diff(x, axis=1), 2.0)
    assert np.allclose(np.diff(y, axis=0), 2.0)
    assert np.all(x == x[0])  # x varies along columns only
    assert np.all(y.T == y[:, 0])  # y varies along rows only


def test_data_region_rejects_empty_or_non_finite_values():
    with pytest.raises(ValueError):
        DataRegion(0.0, 0.0, 0.0, 10.0)
    with pytest.raises(ValueError):
        DataRegion(0.0, 0.0, 10.0, -1.0)
    with pytest.raises(ValueError):
        DataRegion(float("nan"), 0.0, 10.0, 10.0)


# --- recovering an upscaled spectrogram ------------------------------------


@pytest.mark.parametrize("factor", [1, 2, 3, 4, 5, 7])
def test_replicated_upscale_is_recovered_exactly(factor):
    original = random_spectrogram()
    recovered = extract_gray(replicate(original, factor), full_region(factor))
    assert recovered.shape == GRID_SHAPE
    assert np.allclose(recovered, original, atol=1e-9)


@pytest.mark.parametrize("factor", [2, 3, 4, 7])
def test_averaging_inside_the_cell_keeps_a_replicated_upscale_exact(factor):
    original = random_spectrogram(1)
    recovered = extract_gray(
        replicate(original, factor), full_region(factor), samples_per_axis=3, cell_fraction=0.5
    )
    assert np.allclose(recovered, original, atol=1e-6)


@pytest.mark.parametrize("factor", [2, 3, 5])
def test_linearly_interpolated_upscale_is_recovered_with_small_error(factor):
    original = smooth_spectrogram()
    recovered = extract_gray(linear_upscale(original, factor), full_region(factor))
    assert np.max(np.abs(recovered - original)) < 1.0


def test_region_offset_inside_a_larger_card():
    factor = 4
    original = random_spectrogram(2)
    card = np.full((ROWS * factor + 60, COLUMNS * factor + 90), 128.0)
    top, left = 25, 40
    card[top : top + ROWS * factor, left : left + COLUMNS * factor] = replicate(original, factor)
    region = DataRegion(left=left, top=top, width=COLUMNS * factor, height=ROWS * factor)
    assert np.allclose(extract_gray(card, region), original, atol=1e-9)


def test_unit_choice_does_not_matter_only_the_geometry():
    # The same card at two scales, with a region given in each scale's units.
    original = random_spectrogram(3)
    small = replicate(original, 2)
    large = replicate(original, 6)
    from_small = extract_gray(small, full_region(2))
    from_large = extract_gray(large, full_region(6))
    assert np.allclose(from_small, from_large, atol=1e-9)


def test_averaging_reduces_halftone_like_noise():
    factor = 7  # one cell is about 7 printer dots
    original = smooth_spectrogram()
    rng = np.random.default_rng(4)
    noisy = replicate(original, factor) + rng.normal(
        0.0, 8.0, size=(ROWS * factor, COLUMNS * factor)
    )
    region = full_region(factor)

    centre_only = extract_gray(noisy, region)
    averaged = extract_gray(noisy, region, samples_per_axis=3, cell_fraction=0.8)

    error_centre = np.mean(np.abs(centre_only - original))
    error_averaged = np.mean(np.abs(averaged - original))
    assert error_averaged < 0.6 * error_centre


# --- output range and type -------------------------------------------------


def test_output_is_float_and_not_rounded():
    image = np.full((ROWS * 2, COLUMNS * 2), 100.5)
    recovered = extract_gray(image, full_region(2))
    assert recovered.dtype == np.float64
    assert np.allclose(recovered, 100.5)


def test_output_is_clipped_to_the_gray_range():
    image = np.full((ROWS * 2, COLUMNS * 2), 300.0)
    image[:, :COLUMNS] = -40.0
    recovered = extract_gray(image, full_region(2))
    assert recovered.min() == 0.0
    assert recovered.max() == 255.0


def test_integer_input_is_accepted():
    image = np.full((ROWS * 2, COLUMNS * 2), 77, dtype=np.uint8)
    recovered = extract_gray(image, full_region(2))
    assert recovered.dtype == np.float64
    assert np.allclose(recovered, 77.0)


# --- smoothing -------------------------------------------------------------


def test_smoothing_is_off_by_default():
    image = replicate(random_spectrogram(5), 3)
    default = extract_gray(image, full_region(3))
    explicit_off = extract_gray(image, full_region(3), smoothing=None)
    assert np.array_equal(default, explicit_off)


def test_smoothing_changes_the_result_when_enabled():
    image = replicate(random_spectrogram(6), 3)
    plain = extract_gray(image, full_region(3))
    smoothed = extract_gray(image, full_region(3), smoothing=binomial_kernel(3))
    assert not np.allclose(plain, smoothed)
    # Smoothing random cells lowers the cell-to-cell variation.
    assert smoothed.std() < plain.std()


def test_kernels_are_normalized_and_symmetric():
    for kernel in (box_kernel(3), box_kernel(5), binomial_kernel(3), binomial_kernel(5)):
        assert kernel.sum() == pytest.approx(1.0)
        assert np.allclose(kernel, kernel[::-1])
    assert np.allclose(binomial_kernel(3), [0.25, 0.5, 0.25])
    assert np.allclose(box_kernel(3), [1 / 3, 1 / 3, 1 / 3])


def test_smooth_separable_matches_a_direct_double_loop():
    rng = np.random.default_rng(7)
    grid = rng.uniform(0.0, 255.0, size=(9, 11))
    kernel = binomial_kernel(5)
    radius = 2

    padded = np.pad(grid, radius, mode="edge")
    expected = np.zeros_like(grid)
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            total = 0.0
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    weight = kernel[dy + radius] * kernel[dx + radius]
                    total += weight * padded[row + radius + dy, column + radius + dx]
            expected[row, column] = total
    assert np.allclose(smooth_separable(grid, kernel), expected)


def test_smooth_separable_keeps_a_constant_and_spreads_an_impulse():
    constant = np.full((6, 8), 42.0)
    assert np.allclose(smooth_separable(constant, box_kernel(3)), 42.0)

    impulse = np.zeros((5, 5))
    impulse[2, 2] = 16.0
    result = smooth_separable(impulse, binomial_kernel(3))
    assert result[2, 2] == pytest.approx(16.0 * 0.25)  # 0.5 * 0.5 * 16 = 4
    assert result[2, 1] == pytest.approx(16.0 * 0.125)
    assert result[1, 1] == pytest.approx(16.0 / 16.0)
    assert result.sum() == pytest.approx(16.0)


@pytest.mark.parametrize(
    "kernel",
    [
        np.array([0.5, 0.5]),  # even length
        np.array([0.2, 0.6, 0.2, 0.0, 0.0, 0.0]),  # even length
        np.array([0.5, 0.5, 0.5]),  # does not sum to 1
        np.array([0.6, 0.3, 0.1]),  # not symmetric
        np.array([-0.25, 1.5, -0.25]),  # negative weights
        np.ones((3, 3)) / 9.0,  # not 1-D
    ],
)
def test_invalid_kernels_are_rejected(kernel):
    with pytest.raises(ValueError):
        smooth_separable(np.zeros((5, 5)), kernel)


# --- magnitude -------------------------------------------------------------


def test_magnitude_uses_the_existing_mapping_and_the_vertical_flip():
    original = random_spectrogram(8)
    image = replicate(original, 3)
    magnitude = extract_magnitude(image, full_region(3))
    expected = gray_to_magnitude(image_to_grid(original))
    assert magnitude.shape == GRID_SHAPE
    assert np.allclose(magnitude, expected)


def test_top_row_of_the_picture_is_the_highest_frequency():
    image = np.full((ROWS * 2, COLUMNS * 2), 255.0)
    image[:2, :] = 0.0  # the top row of cells is black
    magnitude = extract_magnitude(image, full_region(2))
    assert np.allclose(magnitude[-1, :], 1.0)  # last grid row: highest frequency
    assert np.all(magnitude[0, :] < 1.0)


def test_extract_magnitude_passes_options_through():
    image = replicate(random_spectrogram(9), 3)
    plain = extract_magnitude(image, full_region(3))
    smoothed = extract_magnitude(image, full_region(3), smoothing=box_kernel(3))
    assert not np.allclose(plain, smoothed)


# --- sampler interface (issue #8) -------------------------------------------


def test_default_sampler_is_the_bilinear_sampler_of_geometry(monkeypatch):
    calls = []
    real = extraction.sample_bilinear

    def spy(image, x, y, **kwargs):
        calls.append(image.shape)
        return real(image, x, y, **kwargs)

    monkeypatch.setattr(extraction, "sample_bilinear", spy)
    image = replicate(random_spectrogram(10), 2)
    recovered = extract_gray(image, full_region(2))
    assert calls == [image.shape]
    assert np.allclose(recovered, random_spectrogram(10), atol=1e-9)


def test_cells_smaller_than_a_pixel_repeat_the_edge_instead_of_becoming_nan():
    # Half as many pixels as cells: the first and last cell centres fall in the
    # outer half pixel, outside the range geometry.sample_bilinear accepts.
    image = np.full((ROWS // 2, COLUMNS // 2), 50.0)
    image[0, :] = 10.0
    recovered = extract_gray(image, DataRegion(0.0, 0.0, COLUMNS // 2, ROWS // 2))
    assert np.all(np.isfinite(recovered))
    assert np.allclose(recovered[0, :], 10.0)
    assert np.allclose(recovered[-1, :], 50.0)


def test_custom_sampler_receives_index_coordinates_and_is_used():
    calls = []

    def stub(image, x, y):
        calls.append((image.shape, x.copy(), y.copy()))
        return np.full(GRID_SHAPE, 50.0)

    image = np.zeros((ROWS * 2, COLUMNS * 2))
    recovered = extract_gray(image, full_region(2), sampler=stub)

    assert np.allclose(recovered, 50.0)
    assert len(calls) == 1
    shape, x, y = calls[0]
    assert shape == image.shape
    assert x.shape == GRID_SHAPE and y.shape == GRID_SHAPE
    # Cell (0, 0) spans pixels 0 and 1 in index space: its centre is at 0.5.
    assert x[0, 0] == pytest.approx(0.5)
    assert y[0, 0] == pytest.approx(0.5)


def test_sampler_is_called_once_per_sample_point():
    count = 0

    def stub(image, x, y):
        nonlocal count
        count += 1
        return np.zeros(GRID_SHAPE)

    extract_gray(np.zeros((ROWS, COLUMNS)), full_region(1), samples_per_axis=3, sampler=stub)
    assert count == 9


def test_sampler_with_a_wrong_output_shape_is_rejected():
    with pytest.raises(ValueError):
        extract_gray(
            np.zeros((ROWS, COLUMNS)),
            full_region(1),
            sampler=lambda image, x, y: np.zeros((3, 3)),
        )


# --- bad input -------------------------------------------------------------


def test_region_outside_the_image_is_rejected():
    image = np.zeros((ROWS * 2, COLUMNS * 2))
    with pytest.raises(ValueError):
        extract_gray(image, DataRegion(10.0, 0.0, COLUMNS * 2, ROWS * 2))
    with pytest.raises(ValueError):
        extract_gray(image, DataRegion(-5.0, 0.0, COLUMNS * 2, ROWS * 2))


def test_bad_image_is_rejected():
    region = full_region(1)
    with pytest.raises(ValueError):
        extract_gray(np.zeros((ROWS, COLUMNS, 3)), region)  # colour image
    bad = np.zeros((ROWS, COLUMNS))
    bad[3, 3] = np.nan
    with pytest.raises(ValueError):
        extract_gray(bad, region)


@pytest.mark.parametrize(
    ("samples", "fraction"),
    [(0, 0.5), (-1, 0.5), (2, 0.0), (2, 1.5)],
)
def test_bad_sampling_parameters_are_rejected(samples, fraction):
    with pytest.raises(ValueError):
        extract_gray(
            np.zeros((ROWS, COLUMNS)),
            full_region(1),
            samples_per_axis=samples,
            cell_fraction=fraction,
        )
