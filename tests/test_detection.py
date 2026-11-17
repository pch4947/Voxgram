"""Tests for grayscale conversion, thresholding, labelling and marker detection."""

from collections import deque
from functools import cache

import numpy as np
import pytest

from voxgram import card, detection, synthetic
from voxgram import format as fmt
from voxgram.detection import (
    MarkerCandidate,
    MarkerDetectionError,
    box_mean,
    connected_components,
    default_window,
    detect_markers,
    find_marker_candidates,
    integral_image,
    label_components,
    otsu_threshold,
    select_markers,
    threshold_global,
    threshold_local_mean,
    to_grayscale,
)

# ---------------------------------------------------------------------------
# Synthetic photographs of the card, from voxgram.card and voxgram.synthetic
# ---------------------------------------------------------------------------

PHOTO_SHAPE = (1000, 1000)
# 5 px/mm: markers about 35 px wide, the orientation core about 7 px.
SCALE = 5.0
PERSPECTIVE = (0.0006, -0.0004)


def spectrogram_image(seed: int = 0) -> np.ndarray:
    """Random cells: a busy data region full of dark blobs, the hard case."""
    return np.random.default_rng(seed).integers(0, 256, size=fmt.GRID_SHAPE, dtype=np.uint8)


def view_homography(rotation_deg: float) -> np.ndarray:
    return synthetic.make_homography(
        scale=SCALE, center=(500.0, 500.0), rotation_deg=rotation_deg, perspective=PERSPECTIVE
    )


@cache
def _rendered(rotation_deg: float) -> np.ndarray:
    photo = synthetic.render_view(
        spectrogram_image(), view_homography(rotation_deg), PHOTO_SHAPE, supersample=2
    )
    photo.setflags(write=False)
    return photo


def card_view(rotation_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A photograph of the card and the true marker centres in it, in card order."""
    homography = view_homography(rotation_deg)
    truth = synthetic.project_points(homography, card.marker_centers_mm())
    return _rendered(rotation_deg).copy(), truth


def cover_marker(photo: np.ndarray, rotation_deg: float, index: int) -> np.ndarray:
    """Paint white paper over one marker and 1 mm around it, like a sticker."""
    inverse = np.linalg.inv(view_homography(rotation_deg))
    px, py = np.meshgrid(np.arange(photo.shape[1]), np.arange(photo.shape[0]))
    w = inverse[2, 0] * px + inverse[2, 1] * py + inverse[2, 2]
    x = (inverse[0, 0] * px + inverse[0, 1] * py + inverse[0, 2]) / w
    y = (inverse[1, 0] * px + inverse[1, 1] * py + inverse[1, 2]) / w
    cx, cy = fmt.MARKER_CENTERS_MM[index]
    covered = np.maximum(np.abs(x - cx), np.abs(y - cy)) < fmt.MARKER_SIZE_MM / 2 + 1.0
    photo = photo.copy()
    photo[covered] = fmt.GRAY_WHITE
    return photo


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


# ---------------------------------------------------------------------------
# Connected components (#5)
# ---------------------------------------------------------------------------


def as_binary(rows):
    return np.array([[ch == "#" for ch in row] for row in rows])


def flood_fill_labels(binary, connectivity):
    """Reference labelling by breadth-first search, raster order of first pixel."""
    height, width = binary.shape
    labels = np.zeros((height, width), dtype=np.int32)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    count = 0
    for r in range(height):
        for c in range(width):
            if binary[r, c] and not labels[r, c]:
                count += 1
                labels[r, c] = count
                queue = deque([(r, c)])
                while queue:
                    pr, pc = queue.popleft()
                    for dr, dc in steps:
                        nr, nc = pr + dr, pc + dc
                        inside = 0 <= nr < height and 0 <= nc < width
                        if inside and binary[nr, nc] and not labels[nr, nc]:
                            labels[nr, nc] = count
                            queue.append((nr, nc))
    return labels


def test_separate_blobs_and_their_statistics():
    binary = as_binary(
        [
            "##....",
            "##..#.",
            "....##",
            "......",
        ]
    )
    labels, components = label_components(binary, connectivity=4)
    assert components.count == 2
    np.testing.assert_array_equal(components.area, [4, 3])
    np.testing.assert_array_equal(components.x_min, [0, 4])
    np.testing.assert_array_equal(components.x_max, [1, 5])
    np.testing.assert_array_equal(components.y_min, [0, 1])
    np.testing.assert_array_equal(components.y_max, [1, 2])
    np.testing.assert_allclose(components.centroid, [[0.5, 0.5], [13 / 3, 5 / 3]])
    np.testing.assert_allclose(components.fill_ratio, [1.0, 0.75])
    np.testing.assert_array_equal(labels[0], [1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(labels[2], [0, 0, 0, 0, 2, 2])


def test_u_shape_merges_two_provisional_labels():
    binary = as_binary(
        [
            "#...#",
            "#...#",
            "#####",
        ]
    )
    labels, components = label_components(binary)
    assert components.count == 1
    assert components.area[0] == 9
    assert set(np.unique(labels)) == {0, 1}


def test_diagonal_neighbours_depend_on_connectivity():
    binary = as_binary(
        [
            "#..",
            ".#.",
            "..#",
        ]
    )
    assert connected_components(binary, connectivity=4).count == 3
    assert connected_components(binary, connectivity=8).count == 1


def test_blob_touching_the_image_border():
    binary = np.zeros((5, 6), dtype=bool)
    binary[:, 0] = True
    binary[4, :] = True
    binary[0, 5] = True
    components = connected_components(binary)
    assert components.count == 2
    np.testing.assert_array_equal(components.area, [10, 1])
    np.testing.assert_array_equal(components.x_max, [5, 5])


def test_empty_and_full_images():
    empty = connected_components(np.zeros((4, 7), dtype=bool))
    assert empty.count == 0
    assert empty.centroid.shape == (0, 2)
    labels, full = label_components(np.ones((4, 7), dtype=bool))
    assert full.count == 1
    assert full.area[0] == 28
    assert np.all(labels == 1)


def test_ring_is_one_component_with_a_hole():
    binary = as_binary(
        [
            "#####",
            "#...#",
            "#.#.#",
            "#...#",
            "#####",
        ]
    )
    components = connected_components(binary)
    assert components.count == 2
    np.testing.assert_array_equal(components.area, [16, 1])
    np.testing.assert_allclose(components.centroid, [[2.0, 2.0], [2.0, 2.0]])


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("seed", range(6))
def test_labels_match_flood_fill_on_random_images(connectivity, seed):
    rng = np.random.default_rng(seed)
    binary = rng.random((37, 53)) < [0.3, 0.45, 0.6][seed % 3]
    labels, components = label_components(binary, connectivity=connectivity)
    reference = flood_fill_labels(binary, connectivity)
    np.testing.assert_array_equal(labels, reference)
    assert components.count == reference.max()
    np.testing.assert_array_equal(components.area, np.bincount(reference.ravel())[1:])


def test_rejects_bad_connectivity_and_shape():
    with pytest.raises(ValueError):
        connected_components(np.zeros((3, 3), dtype=bool), connectivity=6)
    with pytest.raises(ValueError):
        connected_components(np.zeros(9, dtype=bool))


# ---------------------------------------------------------------------------
# Marker detection (#6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rotation", [0, 90, 180, 270, 37, 45])
def test_markers_found_in_card_order_for_any_rotation(rotation):
    photo, truth = card_view(rotation_deg=rotation)
    markers = detect_markers(photo)
    assert markers.shape == (4, 2)
    np.testing.assert_allclose(markers, truth, atol=0.25)


def test_markers_found_from_a_bgr_photograph():
    photo, truth = card_view(rotation_deg=12)
    bgr = synthetic.as_bgr(photo)
    np.testing.assert_allclose(detect_markers(to_grayscale(bgr)), truth, atol=0.25)


@pytest.mark.parametrize("angle", [0.0, 90.0, 225.0])
def test_markers_found_under_an_illumination_gradient(angle):
    photo, truth = card_view(rotation_deg=8)
    gain = synthetic.illumination_gradient(PHOTO_SHAPE, low=0.35, high=1.0, angle_deg=angle)
    np.testing.assert_allclose(detect_markers(photo * gain), truth, atol=0.25)


def test_markers_found_through_a_tone_curve_and_gradient():
    photo, truth = card_view(rotation_deg=-15)
    gain = synthetic.illumination_gradient(PHOTO_SHAPE, low=0.5, high=1.0, angle_deg=30.0)
    photo = synthetic.apply_tone_curve(photo * gain, gamma=1.8)
    np.testing.assert_allclose(detect_markers(photo), truth, atol=0.25)


def test_global_baseline_finds_markers_in_even_light():
    photo, truth = card_view(rotation_deg=20)
    np.testing.assert_allclose(detect_markers(photo, method="global"), truth, atol=0.25)


@pytest.mark.parametrize("covered", range(4))
def test_covered_marker_is_reported_not_guessed(covered):
    photo, _ = card_view(rotation_deg=0)
    with pytest.raises(MarkerDetectionError, match="found 3"):
        detect_markers(cover_marker(photo, 0, covered))


def test_card_without_orientation_cue_is_rejected(monkeypatch):
    # A card printed with four identical markers has no orientation.
    monkeypatch.setattr(card, "MARKER_CUE_CORE_MM", fmt.MARKER_CORE_MM)
    homography = view_homography(0.0)
    photo = synthetic.render_view(spectrogram_image(), homography, PHOTO_SHAPE, supersample=2)
    with pytest.raises(MarkerDetectionError, match="form a card"):
        detect_markers(photo)


def test_blank_image_is_rejected():
    with pytest.raises(MarkerDetectionError, match="found 0"):
        detect_markers(np.full((200, 300), 230.0))


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method"):
        detect_markers(np.full((50, 50), 230.0), method="otsu")


def test_candidates_report_frame_to_core_ratios():
    photo, _ = card_view()
    binary = threshold_local_mean(photo, default_window(photo.shape), 10.0)
    ratios = sorted(c.area_ratio for c in find_marker_candidates(binary))
    # Three finders and the orientation marker, with the ratios the card prints.
    assert len(ratios) == 4
    np.testing.assert_allclose(ratios[:3], detection.FINDER_AREA_RATIO, rtol=0.2)
    assert ratios[3] == pytest.approx(detection.CUE_AREA_RATIO, rel=0.3)


def test_expected_ratios_follow_the_card_format():
    frame = fmt.MARKER_SIZE_MM**2 - fmt.MARKER_GAP_MM**2
    assert detection.FINDER_AREA_RATIO == pytest.approx(frame / fmt.MARKER_CORE_MM**2)
    assert detection.CUE_AREA_RATIO == pytest.approx(frame / fmt.MARKER_CUE_CORE_MM**2)
    # The two kinds must not overlap once the measurement tolerance is applied.
    tolerance = detection.RATIO_TOLERANCE
    assert detection.FINDER_AREA_RATIO * tolerance < detection.CUE_AREA_RATIO / tolerance


FINDER, CUE = detection.FINDER_AREA_RATIO, detection.CUE_AREA_RATIO


def candidate(x, y, ratio=FINDER, frame_area=600):
    return MarkerCandidate((x, y), frame_area, round(frame_area / ratio), 35.0)


def cue_at(index):
    return CUE if index == fmt.CUE_MARKER_INDEX else FINDER


def test_select_markers_rejects_two_equally_good_cards():
    corners = [(100, 100), (800, 100), (800, 500), (100, 500)]
    first = [candidate(*p, ratio=cue_at(i)) for i, p in enumerate(corners)]
    second = [
        candidate(x + 50, y + 600, ratio=r.area_ratio)
        for (x, y), r in zip(corners, first, strict=True)
    ]
    with pytest.raises(MarkerDetectionError, match="refusing to guess"):
        select_markers(first + second)


def test_select_markers_rejects_collinear_markers():
    points = [(100, 100), (400, 100), (700, 100), (1000, 100)]
    group = [candidate(*p, ratio=cue_at(i)) for i, p in enumerate(points)]
    with pytest.raises(MarkerDetectionError, match="form a card"):
        select_markers(group)
