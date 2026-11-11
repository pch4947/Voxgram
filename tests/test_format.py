"""Consistency checks for the shared format constants."""

from voxgram import format as fmt


def test_sample_count_matches_duration():
    assert fmt.NUM_SAMPLES == 48_000


def test_grid_shape_matches_stft_settings():
    assert fmt.NUM_FREQ_BINS == 257
    assert fmt.NUM_FRAMES == 376
    assert fmt.GRID_SHAPE == (257, 376)


def test_frame_count_formula_assumes_centering():
    # NUM_FRAMES = 1 + NUM_SAMPLES // HOP_LENGTH only holds for a centered STFT.
    assert fmt.CENTER is True


def test_intensity_range():
    assert fmt.DB_FLOOR < 0
    assert (fmt.GRAY_BLACK, fmt.GRAY_WHITE) == (0, 255)


def test_data_region_cells_are_nearly_square():
    cell_w = fmt.DATA_WIDTH_MM / fmt.NUM_FRAMES
    cell_h = fmt.DATA_HEIGHT_MM / fmt.NUM_FREQ_BINS
    assert abs(cell_w - cell_h) / cell_h < 0.01
