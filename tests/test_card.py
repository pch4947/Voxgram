"""Tests for the card layout, the renderer and synthetic photographs."""

import itertools

import numpy as np
import pytest
import soundfile as sf

from voxgram import card, encoder, synthetic
from voxgram import format as fmt
from voxgram.imageio import read_image, write_image


def spectrogram_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=fmt.GRID_SHAPE, dtype=np.uint8)


def marker_rects() -> list[tuple[float, float, float, float]]:
    half = fmt.MARKER_SIZE_MM / 2
    return [(cx - half, cy - half, cx + half, cy + half) for cx, cy in fmt.MARKER_CENTERS_MM]


def patch_rects() -> list[tuple[float, float, float, float]]:
    return [patch[:4] for patch in fmt.CALIBRATION_PATCHES]


def data_rect() -> tuple[float, float, float, float]:
    return (
        fmt.DATA_X_MM,
        fmt.DATA_Y_MM,
        fmt.DATA_X_MM + fmt.DATA_WIDTH_MM,
        fmt.DATA_Y_MM + fmt.DATA_HEIGHT_MM,
    )


def gap(a, b) -> float:
    """Distance between two axis-aligned rectangles; 0 if they touch or overlap."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return float(np.hypot(dx, dy))


# --- layout -----------------------------------------------------------------


def test_every_element_is_inside_the_card_with_a_margin():
    for x0, y0, x1, y1 in [*marker_rects(), *patch_rects(), data_rect()]:
        assert x0 >= 3.0 and y0 >= 3.0
        assert x1 <= fmt.CARD_WIDTH_MM - 3.0 and y1 <= fmt.CARD_HEIGHT_MM - 3.0


def test_markers_have_a_quiet_zone():
    others = [*patch_rects(), data_rect()]
    for marker in marker_rects():
        assert min(gap(marker, other) for other in others) >= 3.0


def test_patches_do_not_touch_the_data_region_or_each_other():
    for patch in patch_rects():
        assert gap(patch, data_rect()) >= 3.0
    for a, b in itertools.combinations(patch_rects(), 2):
        assert gap(a, b) >= 1.5


def test_data_region_is_centred_and_has_the_specified_size():
    x0, y0, x1, y1 = data_rect()
    assert (x1 - x0, y1 - y0) == (110.0, 75.0)
    assert x0 == fmt.CARD_WIDTH_MM - x1
    assert y0 == fmt.CARD_HEIGHT_MM - y1


def test_every_side_carries_every_level_once():
    assert len(fmt.CALIBRATION_PATCHES) == 4 * len(fmt.PATCH_LEVELS)
    levels = card.patch_levels().reshape(4, -1)
    for side in levels:
        assert sorted(side) == sorted(fmt.PATCH_LEVELS)
    assert fmt.PATCH_LEVELS[0] == 0 and fmt.PATCH_LEVELS[-1] == 255


def test_marker_order_is_clockwise_from_top_left():
    tl, tr, br, bl = card.marker_centers_mm()
    assert tl[0] < tr[0] and tl[1] == tr[1]
    assert bl[0] < br[0] and bl[1] == br[1]
    assert tl[1] < bl[1] and tr[1] < br[1]
    assert fmt.CUE_MARKER_INDEX == 2


def test_frame_to_core_area_ratios_separate_the_two_marker_kinds():
    frame = fmt.MARKER_SIZE_MM**2 - fmt.MARKER_GAP_MM**2
    finder = frame / fmt.MARKER_CORE_MM**2
    cue = frame / fmt.MARKER_CUE_CORE_MM**2
    assert finder == pytest.approx(24 / 9)
    assert cue / finder >= 3.0


# --- card function ----------------------------------------------------------


def value_at(x: float, y: float, image: np.ndarray) -> float:
    return float(card.card_value_at(np.array([x]), np.array([y]), image)[0])


def test_marker_cross_section():
    image = spectrogram_image()
    for index, (cx, cy) in enumerate(fmt.MARKER_CENTERS_MM):
        core = fmt.MARKER_CUE_CORE_MM if index == fmt.CUE_MARKER_INDEX else fmt.MARKER_CORE_MM
        assert value_at(cx, cy, image) == 0  # core
        assert value_at(cx + core / 2 - 0.1, cy, image) == 0
        assert value_at(cx + core / 2 + 0.1, cy, image) == 255  # white gap
        assert value_at(cx + 2.4, cy, image) == 255
        assert value_at(cx + 2.6, cy, image) == 0  # frame
        assert value_at(cx + 3.4, cy + 3.4, image) == 0  # frame corner
        assert value_at(cx + 3.6, cy, image) == 255  # quiet zone


def test_patch_centres_carry_their_levels():
    image = spectrogram_image()
    centres = card.patch_centers_mm()
    values = card.card_value_at(centres[:, 0], centres[:, 1], image)
    assert np.array_equal(values, card.patch_levels())


def test_cell_centres_return_the_spectrogram():
    image = spectrogram_image()
    x, y = card.data_cell_centers_mm()
    assert x.shape == fmt.GRID_SHAPE
    assert np.array_equal(card.card_value_at(x, y, image), image)
    # Row 0 is the top of the data region, column 0 its left edge.
    assert x[0, 0] == pytest.approx(fmt.DATA_X_MM + card.CELL_WIDTH_MM / 2)
    assert y[0, 0] == pytest.approx(fmt.DATA_Y_MM + card.CELL_HEIGHT_MM / 2)
    assert y[-1, 0] < fmt.DATA_Y_MM + fmt.DATA_HEIGHT_MM


def test_outside_the_card_is_nan_and_margins_are_white():
    image = spectrogram_image()
    assert np.isnan(value_at(-0.1, 50.0, image))
    assert np.isnan(value_at(75.0, 100.1, image))
    assert value_at(1.0, 50.0, image) == 255
    assert value_at(17.0, 50.0, image) == 255  # between the left patches and the data


def test_card_value_rejects_a_wrong_image_shape():
    with pytest.raises(ValueError, match="shape"):
        card.card_value_at(np.zeros(1), np.zeros(1), np.zeros((10, 10)))


# --- print rendering --------------------------------------------------------


@pytest.fixture(scope="module")
def printed():
    image = spectrogram_image()
    return image, card.render_card(image, dpi=600)


def test_print_size_at_600_dpi(printed):
    _, rendered = printed
    assert rendered.shape == (2362, 3543)
    assert rendered.dtype == np.uint8


def test_printed_card_holds_exact_values(printed):
    image, rendered = printed
    matrix = card.print_matrix(600)

    def sample(points_mm: np.ndarray) -> np.ndarray:
        pixels = np.rint(synthetic.project_points(matrix, points_mm)).astype(int)
        return rendered[pixels[:, 1], pixels[:, 0]]

    x, y = card.data_cell_centers_mm()
    cells = sample(np.column_stack([x.ravel(), y.ravel()])).reshape(fmt.GRID_SHAPE)
    assert np.array_equal(cells, image)
    assert np.array_equal(sample(card.patch_centers_mm()), card.patch_levels())
    assert np.all(sample(card.marker_centers_mm()) == 0)
    # Only encoded levels appear: there is no blending between neighbours.
    assert set(np.unique(rendered)) <= set(np.unique(image)) | set(fmt.PATCH_LEVELS)


def test_png_records_the_print_resolution(tmp_path):
    rendered = card.render_card(spectrogram_image(), dpi=150)
    path = tmp_path / "card.png"
    write_image(path, rendered, dpi=150)

    data = path.read_bytes()
    index = data.index(b"pHYs")
    pixels_per_metre = int.from_bytes(data[index + 4 : index + 8], "big")
    assert pixels_per_metre == round(150 / 0.0254)
    assert data[index + 12] == 1  # unit: metre
    assert np.array_equal(read_image(path), rendered)


def test_dpi_is_rejected_for_other_formats(tmp_path):
    with pytest.raises(ValueError, match="PNG"):
        write_image(tmp_path / "card.jpg", np.zeros((4, 4), dtype=np.uint8), dpi=300)


def test_encoder_cli_writes_a_card(tmp_path, capsys):
    t = np.arange(fmt.SAMPLE_RATE) / fmt.SAMPLE_RATE
    wav = tmp_path / "voice.wav"
    sf.write(str(wav), (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), fmt.SAMPLE_RATE)
    card_path = tmp_path / "card.png"

    assert encoder.main([str(wav), "--card", str(card_path), "--dpi", "150"]) == 0
    assert "150 x 100 mm" in capsys.readouterr().out
    assert read_image(card_path).shape == card.print_shape(150)
    assert read_image(tmp_path / "voice.png").shape == fmt.GRID_SHAPE


# --- synthetic photographs --------------------------------------------------


def blocky_image() -> np.ndarray:
    """A spectrogram image made of large flat blocks, so sampling is unambiguous."""
    rows = np.arange(fmt.GRID_SHAPE[0])[:, None] // 32
    cols = np.arange(fmt.GRID_SHAPE[1])[None, :] // 47
    return ((rows * 8 + cols) * 4 % 256).astype(np.uint8)


def test_homography_places_the_card_centre_and_scale():
    homography = synthetic.make_homography(scale=8.0, center=(700.0, 500.0), rotation_deg=90.0)
    centre = synthetic.project_points(homography, np.array([[75.0, 50.0]]))
    assert centre[0] == pytest.approx([700.0, 500.0])
    # Rotated a quarter turn clockwise: the card's right edge points down.
    right = synthetic.project_points(homography, np.array([[85.0, 50.0]]))
    assert right[0] == pytest.approx([700.0, 580.0])


def test_tilted_view_matches_ground_truth():
    image = blocky_image()
    shape = (900, 1300)
    homography = synthetic.make_homography(
        scale=7.0, center=(650.0, 450.0), rotation_deg=12.0, perspective=(0.0008, -0.0005)
    )
    photo = synthetic.render_view(image, homography, shape, supersample=2, background=40.0)
    assert photo.shape == shape

    def sample(points_mm: np.ndarray) -> np.ndarray:
        pixels = np.rint(synthetic.project_points(homography, points_mm)).astype(int)
        return photo[pixels[:, 1], pixels[:, 0]]

    assert np.allclose(sample(card.patch_centers_mm()), card.patch_levels())
    assert np.all(sample(card.marker_centers_mm()) == 0)
    # The middle of every flat block of the spectrogram.
    x, y = card.data_cell_centers_mm()
    block_rows = np.arange(16, fmt.GRID_SHAPE[0], 32)
    block_cols = np.arange(23, fmt.GRID_SHAPE[1], 47)
    points = np.column_stack(
        [x[np.ix_(block_rows, block_cols)].ravel(), y[np.ix_(block_rows, block_cols)].ravel()]
    )
    assert np.allclose(sample(points), image[np.ix_(block_rows, block_cols)].ravel())
    # A corner of the photograph is desk, not card.
    assert photo[0, 0] == 40.0


def test_supersampling_blends_edges_and_point_sampling_does_not():
    image = blocky_image()
    homography = synthetic.make_homography(scale=3.3, center=(300.0, 200.0), rotation_deg=7.0)
    sharp = synthetic.render_view(image, homography, (400, 600), supersample=1)
    smooth = synthetic.render_view(image, homography, (400, 600), supersample=3)
    allowed = set(np.unique(image)) | set(fmt.PATCH_LEVELS) | {40}
    assert set(np.unique(sharp)) <= allowed
    assert len(np.unique(smooth)) > len(np.unique(sharp))
    with pytest.raises(ValueError):
        synthetic.render_view(image, homography, (10, 10), supersample=0)


def test_illumination_and_tone():
    gain = synthetic.illumination_gradient((100, 200), low=0.6, high=1.0, angle_deg=0.0)
    assert gain.shape == (100, 200)
    assert gain[:, 0] == pytest.approx(0.6)
    assert gain[:, -1] == pytest.approx(1.0)
    assert np.all(np.diff(gain, axis=1) > 0)

    vertical = synthetic.illumination_gradient((100, 200), low=0.5, high=0.9, angle_deg=90.0)
    assert vertical[0] == pytest.approx(0.5)
    assert vertical[-1] == pytest.approx(0.9)

    values = np.array([0.0, 64.0, 128.0, 255.0])
    toned = synthetic.apply_tone_curve(values, gamma=2.0)
    assert toned[0] == 0.0 and toned[-1] == 255.0
    assert toned[2] == pytest.approx(255.0 * (128.0 / 255.0) ** 2)
    assert np.all(np.diff(toned) > 0)


def test_as_bgr_matches_the_photograph_layout():
    photo = np.array([[0.4, 254.6], [300.0, -5.0]])
    bgr = synthetic.as_bgr(photo)
    assert bgr.shape == (2, 2, 3)
    assert bgr.dtype == np.uint8
    assert np.array_equal(bgr[:, :, 0], [[0, 255], [255, 0]])
    assert np.array_equal(bgr[:, :, 0], bgr[:, :, 2])
