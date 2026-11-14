"""Card layout: where everything is on the card, and how to draw it.

The card is described as a function of position: :func:`card_value_at` gives
the gray value at any point in canonical card coordinates. :func:`render_view`
evaluates that function through a homography. A print-ready image is the view
through a pure scaling, and a synthetic photograph (voxgram.synthetic) is the
view through a tilted one, so nothing here resizes or warps an image.

A homography ``H`` maps card millimetres to pixels,
``[x_px, y_px, 1]^T ~ H [x_mm, y_mm, 1]^T``.

Canonical card coordinates are millimetres from the top-left corner of the
card, x to the right and y down. Points are always written (x, y).

Pixel coordinates are also written (x, y) = (column, row), and pixel centres
sit at integer coordinates.
"""

from __future__ import annotations

import numpy as np

from voxgram.format import (
    CALIBRATION_PATCHES,
    CARD_HEIGHT_MM,
    CARD_WIDTH_MM,
    CUE_MARKER_INDEX,
    DATA_HEIGHT_MM,
    DATA_WIDTH_MM,
    DATA_X_MM,
    DATA_Y_MM,
    GRAY_BLACK,
    GRAY_WHITE,
    GRID_SHAPE,
    MARKER_CENTERS_MM,
    MARKER_CORE_MM,
    MARKER_CUE_CORE_MM,
    MARKER_GAP_MM,
    MARKER_SIZE_MM,
    NUM_FRAMES,
    NUM_FREQ_BINS,
    TARGET_PRINT_DPI,
)

MM_PER_INCH = 25.4

# Rows evaluated at a time when rendering, to bound memory.
_ROWS_PER_BLOCK = 256

# Size of one spectrogram cell on the card.
CELL_WIDTH_MM = DATA_WIDTH_MM / NUM_FRAMES
CELL_HEIGHT_MM = DATA_HEIGHT_MM / NUM_FREQ_BINS


def marker_centers_mm() -> np.ndarray:
    """Marker centres as a (4, 2) array of (x, y) in mm.

    Order: top-left, top-right, bottom-right, bottom-left. The bottom-right
    marker (index CUE_MARKER_INDEX) is the orientation marker.
    """
    return np.array(MARKER_CENTERS_MM, dtype=np.float64)


def patch_centers_mm() -> np.ndarray:
    """Calibration patch centres as an (N, 2) array of (x, y) in mm.

    Row ``i`` belongs to ``CALIBRATION_PATCHES[i]``.
    """
    rects = np.array([patch[:4] for patch in CALIBRATION_PATCHES], dtype=np.float64)
    return np.stack([(rects[:, 0] + rects[:, 2]) / 2, (rects[:, 1] + rects[:, 3]) / 2], axis=1)


def patch_levels() -> np.ndarray:
    """Encoded gray level of every calibration patch, (N,) float64."""
    return np.array([patch[4] for patch in CALIBRATION_PATCHES], dtype=np.float64)


def data_cell_centers_mm() -> tuple[np.ndarray, np.ndarray]:
    """Centre of every spectrogram cell in canonical card coordinates.

    Returns:
        ``(x, y)``, two (257, 376) float64 arrays in mm, indexed like the
        spectrogram image: row 0 is the top row (highest frequency) and
        column 0 is the first frame.
    """
    x = DATA_X_MM + (np.arange(NUM_FRAMES) + 0.5) * CELL_WIDTH_MM
    y = DATA_Y_MM + (np.arange(NUM_FREQ_BINS) + 0.5) * CELL_HEIGHT_MM
    return np.meshgrid(x, y)


def card_value_at(x: np.ndarray, y: np.ndarray, spectrogram_image: np.ndarray) -> np.ndarray:
    """Gray value of the card at points in canonical card coordinates.

    Args:
        x, y: arrays of the same shape, in mm.
        spectrogram_image: (257, 376) image from the encoder, row 0 at the top.

    Returns:
        float64 array shaped like ``x``, in [0, 255]. Points outside the card
        are NaN.
    """
    image = np.asarray(spectrogram_image)
    if image.shape != GRID_SHAPE:
        raise ValueError(f"spectrogram image must have shape {GRID_SHAPE}, got {image.shape}")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    shape = x.shape
    x = x.ravel()
    y = y.ravel()
    out = np.full(x.shape, np.nan)

    on_card = (x >= 0) & (x < CARD_WIDTH_MM) & (y >= 0) & (y < CARD_HEIGHT_MM)
    out[on_card] = GRAY_WHITE

    # Spectrogram cells.
    u = (x - DATA_X_MM) / CELL_WIDTH_MM
    v = (y - DATA_Y_MM) / CELL_HEIGHT_MM
    in_data = (u >= 0) & (u < NUM_FRAMES) & (v >= 0) & (v < NUM_FREQ_BINS)
    out[in_data] = image[v[in_data].astype(np.intp), u[in_data].astype(np.intp)]

    # Everything else is in the margin; only look at those points.
    margin = np.flatnonzero(on_card & ~in_data)
    xm = x[margin]
    ym = y[margin]
    values = out[margin]

    for x0, y0, x1, y1, level in CALIBRATION_PATCHES:
        values[(xm >= x0) & (xm < x1) & (ym >= y0) & (ym < y1)] = level

    for index, (cx, cy) in enumerate(MARKER_CENTERS_MM):
        core = MARKER_CUE_CORE_MM if index == CUE_MARKER_INDEX else MARKER_CORE_MM
        # Chebyshev distance from the centre: squares are level sets of it.
        distance = np.maximum(np.abs(xm - cx), np.abs(ym - cy))
        frame = (distance < MARKER_SIZE_MM / 2) & (distance >= MARKER_GAP_MM / 2)
        values[frame | (distance < core / 2)] = GRAY_BLACK

    out[margin] = values
    return out.reshape(shape)


def render_view(
    spectrogram_image: np.ndarray,
    homography: np.ndarray,
    shape: tuple[int, int],
    *,
    supersample: int = 3,
    background: float = 40.0,
) -> np.ndarray:
    """Render the card as seen through a homography.

    Args:
        spectrogram_image: (257, 376) image from the encoder.
        homography: (3, 3) card mm to photograph pixels.
        shape: (height, width) of the photograph.
        supersample: each pixel is the mean of ``supersample ** 2`` point
            samples spread over its area. Use 1 for point sampling.
        background: gray value of everything that is not the card.

    Returns:
        (height, width) float64 photograph in [0, 255].
    """
    if supersample < 1:
        raise ValueError("supersample must be at least 1")
    height, width = shape
    inverse = np.linalg.inv(np.asarray(homography, dtype=np.float64))
    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
    columns = np.arange(width, dtype=np.float64)
    photo = np.empty((height, width))

    for start in range(0, height, _ROWS_PER_BLOCK):
        rows = np.arange(start, min(start + _ROWS_PER_BLOCK, height), dtype=np.float64)
        total = np.zeros((rows.size, width))
        for dy in offsets:
            for dx in offsets:
                px, py = np.meshgrid(columns + dx, rows + dy)
                divisor = inverse[2, 0] * px + inverse[2, 1] * py + inverse[2, 2]
                visible = divisor > 1e-12
                divisor = np.where(visible, divisor, 1.0)
                card_x = (inverse[0, 0] * px + inverse[0, 1] * py + inverse[0, 2]) / divisor
                card_y = (inverse[1, 0] * px + inverse[1, 1] * py + inverse[1, 2]) / divisor
                values = card_value_at(card_x, card_y, spectrogram_image)
                values[~visible | np.isnan(values)] = background
                total += values
        photo[start : start + rows.size] = total / supersample**2
    return photo


def print_matrix(dpi: float = TARGET_PRINT_DPI) -> np.ndarray:
    """Homography from card mm to the pixels of a print image at ``dpi``.

    A (3, 3) matrix ``H`` with ``[x_px, y_px, 1]^T ~ H [x_mm, y_mm, 1]^T``.
    Pixel centres are at integer coordinates, hence the half-pixel shift.
    """
    scale = dpi / MM_PER_INCH
    return np.array([[scale, 0.0, -0.5], [0.0, scale, -0.5], [0.0, 0.0, 1.0]])


def print_shape(dpi: float = TARGET_PRINT_DPI) -> tuple[int, int]:
    """(height, width) in pixels of the print image at ``dpi``."""
    scale = dpi / MM_PER_INCH
    return round(CARD_HEIGHT_MM * scale), round(CARD_WIDTH_MM * scale)


def render_card(spectrogram_image: np.ndarray, dpi: float = TARGET_PRINT_DPI) -> np.ndarray:
    """Draw the print-ready card.

    Args:
        spectrogram_image: (257, 376) uint8 image from the encoder.
        dpi: print resolution. Write the result with
            ``write_image(path, card, dpi=dpi)`` so it prints at 150 x 100 mm.

    Returns:
        (H, W) uint8 image; (2362, 3543) at 600 dpi. Each pixel takes the
        value at its centre, so every pixel holds an exact encoded level.
    """
    view = render_view(
        spectrogram_image, print_matrix(dpi), print_shape(dpi), supersample=1, background=GRAY_WHITE
    )
    return np.rint(view).astype(np.uint8)
