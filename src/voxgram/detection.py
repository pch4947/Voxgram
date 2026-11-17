"""Marker detection: grayscale conversion and thresholding.

This is the first stage of the decoder. It turns a photograph into a grayscale
image and makes a binary working image from it for marker detection.

The binary image exists only for detection. The grayscale image returned by
:func:`to_grayscale` is what rectification and extraction work on, unchanged.

Conventions
-----------
* Gray values are float64 in [0, 255], 0 black and 255 white.
* In binary images ``True`` means dark: marker, patch or spectrogram ink.

Library boundary: everything here is written with NumPy arithmetic only. No
OpenCV, SciPy or scikit-image routine is used.
"""

from __future__ import annotations

import numpy as np

from voxgram.format import GRAY_BLACK, GRAY_WHITE

__all__ = [
    "box_mean",
    "default_window",
    "integral_image",
    "otsu_threshold",
    "threshold_global",
    "threshold_local_mean",
    "to_grayscale",
]

# ---------------------------------------------------------------------------
# Grayscale conversion and thresholding (issue #4)
# ---------------------------------------------------------------------------

# ITU-R BT.601 luma weights, listed in the B, G, R order of voxgram.imageio.
_LUMA_WEIGHTS_BGR = (0.114, 0.587, 0.299)

_UINT16_MAX = 65535

# Default darkness margin for the local-mean threshold, in gray levels. A pixel
# is dark only if it is this much darker than its neighbourhood, which keeps
# sensor noise on blank paper out of the binary image.
DEFAULT_LOCAL_OFFSET = 10.0

# The default local-mean window is this fraction of the longer image side.
# It must be well above the marker size, otherwise the inside of a solid core
# is compared with a mean that is itself dark and turns white.
DEFAULT_WINDOW_FRACTION = 1 / 8
MIN_WINDOW = 15


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image from ``voxgram.imageio.read_image`` to grayscale.

    Args:
        image: (H, W) grayscale, (H, W, 3) BGR or (H, W, 4) BGRA. uint8,
            uint16 (scaled to [0, 255]) or floating point already in
            [0, 255]. The alpha channel is ignored.

    Returns:
        (H, W) float64 in [0, 255]. Colour images use the BT.601 luma weights
        0.299 R + 0.587 G + 0.114 B.

    Raises:
        ValueError: for any other shape or dtype.
    """
    image = np.asarray(image)
    if image.dtype == np.uint8 or np.issubdtype(image.dtype, np.floating):
        scale = 1.0
    elif image.dtype == np.uint16:
        scale = GRAY_WHITE / _UINT16_MAX
    else:
        raise ValueError(f"unsupported image dtype {image.dtype}")

    if image.ndim == 2:
        gray = image.astype(np.float64)
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        # Channel by channel, so that a 12-megapixel photograph does not need
        # a full (H, W, 3) float64 copy.
        gray = np.zeros(image.shape[:2], dtype=np.float64)
        for channel, weight in enumerate(_LUMA_WEIGHTS_BGR):
            gray += weight * image[:, :, channel]
    else:
        raise ValueError(f"expected (H, W), (H, W, 3) or (H, W, 4), got shape {image.shape}")

    if scale != 1.0:
        gray *= scale
    return gray


def _as_gray(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.float64)
    if gray.ndim != 2:
        raise ValueError(f"expected a (H, W) grayscale image, got shape {gray.shape}")
    return gray


def threshold_global(gray: np.ndarray, threshold: float) -> np.ndarray:
    """Global threshold: the detection baseline.

    Args:
        gray: (H, W) float64 in [0, 255].
        threshold: gray level. Pixels strictly darker than it are dark.

    Returns:
        (H, W) bool, ``True`` for dark pixels.
    """
    return _as_gray(gray) < threshold


def otsu_threshold(gray: np.ndarray) -> float:
    """Choose a global threshold by Otsu's method.

    The image is histogrammed over the 256 integer levels (values are rounded
    and clipped to [0, 255]). The returned threshold maximizes the
    between-class variance of the dark and light classes.

    Args:
        gray: (H, W) float64 in [0, 255].

    Returns:
        A threshold ``t`` to pass to :func:`threshold_global`. It lies halfway
        between two integer levels, so ``gray < t`` puts every pixel that
        rounds to the dark class in it.

    Raises:
        ValueError: for an empty image.
    """
    gray = _as_gray(gray)
    if gray.size == 0:
        raise ValueError("cannot threshold an empty image")
    levels = np.rint(np.clip(gray, GRAY_BLACK, GRAY_WHITE)).astype(np.intp)
    histogram = np.bincount(levels.ravel(), minlength=GRAY_WHITE + 1).astype(np.float64)
    probability = histogram / histogram.sum()
    values = np.arange(GRAY_WHITE + 1, dtype=np.float64)

    # Class 0 is levels 0..k, class 1 is levels k+1..255.
    weight0 = np.cumsum(probability)
    mean_sum0 = np.cumsum(probability * values)
    total_mean = mean_sum0[-1]
    weight1 = 1.0 - weight0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (total_mean * weight0 - mean_sum0) ** 2 / (weight0 * weight1)
    between[~np.isfinite(between)] = -1.0
    # Several levels can share the maximum when the histogram has empty bins
    # between the two classes. Take the middle of that plateau.
    best = np.flatnonzero(between == between.max())
    level = (best[0] + best[-1]) / 2
    return float(level + 0.5)


def integral_image(gray: np.ndarray) -> np.ndarray:
    """Summed-area table of an image.

    Args:
        gray: (H, W) array.

    Returns:
        (H + 1, W + 1) float64 ``S`` with a zero first row and column, so that
        ``S[r, c]`` is the sum of ``gray[:r, :c]`` and the sum over rows
        ``r0..r1-1`` and columns ``c0..c1-1`` is
        ``S[r1, c1] - S[r0, c1] - S[r1, c0] + S[r0, c0]``.
    """
    gray = _as_gray(gray)
    height, width = gray.shape
    table = np.zeros((height + 1, width + 1), dtype=np.float64)
    np.cumsum(gray, axis=0, out=table[1:, 1:])
    np.cumsum(table[1:, 1:], axis=1, out=table[1:, 1:])
    return table


def box_mean(gray: np.ndarray, window: int) -> np.ndarray:
    """Mean over a ``window`` x ``window`` square centred on every pixel.

    Computed in constant time per pixel from :func:`integral_image`. Near the
    image border the square is cut off by the border and the mean is taken
    over the pixels that remain, so no padding value leaks in.

    Args:
        gray: (H, W) float64.
        window: odd side length in pixels, at least 1.

    Returns:
        (H, W) float64.
    """
    gray = _as_gray(gray)
    _check_window(window)
    height, width = gray.shape
    half = window // 2
    table = integral_image(gray)

    rows = np.arange(height)
    columns = np.arange(width)
    top = np.clip(rows - half, 0, height)
    bottom = np.clip(rows + half + 1, 0, height)
    left = np.clip(columns - half, 0, width)
    right = np.clip(columns + half + 1, 0, width)

    total = table[np.ix_(bottom, right)]
    total -= table[np.ix_(top, right)]
    total -= table[np.ix_(bottom, left)]
    total += table[np.ix_(top, left)]
    count = np.outer(bottom - top, right - left)
    total /= count
    return total


def threshold_local_mean(gray: np.ndarray, window: int, offset: float) -> np.ndarray:
    """Local-mean threshold: dark if darker than the neighbourhood mean.

    A pixel is dark when ``gray < box_mean(gray, window) - offset``. Because
    the comparison is local, a slow change of illumination across the card
    does not move pixels between the classes, as it does for a global
    threshold.

    Args:
        gray: (H, W) float64 in [0, 255].
        window: odd side of the averaging square, in pixels. It should be
            clearly larger than the largest dark shape to keep, here a marker:
            a solid region larger than the window is hollowed out.
        offset: margin in gray levels. Positive values keep noise on blank
            paper out of the dark class.

    Returns:
        (H, W) bool, ``True`` for dark pixels.
    """
    gray = _as_gray(gray)
    return gray < box_mean(gray, window) - offset


def default_window(shape: tuple[int, ...]) -> int:
    """Default local-mean window for an image of this shape.

    The largest odd number not above 1/8 of the longer side, and at least
    ``MIN_WINDOW``. For a photograph in which the card fills a good part of
    the frame this is several times the marker size.
    """
    window = int(max(shape[:2]) * DEFAULT_WINDOW_FRACTION)
    window -= 1 - window % 2
    return max(window, MIN_WINDOW)


def _check_window(window: int) -> None:
    if int(window) != window or window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window}")
