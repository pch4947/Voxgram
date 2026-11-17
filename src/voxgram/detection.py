"""Marker detection: grayscale conversion, thresholding, connected components.

This is the first stage of the decoder. It turns a photograph into a grayscale
image, makes a binary working image from it and labels the dark regions of
that binary image.

The binary image exists only for detection. The grayscale image returned by
:func:`to_grayscale` is what rectification and extraction work on, unchanged.

Conventions
-----------
* Gray values are float64 in [0, 255], 0 black and 255 white.
* In binary images ``True`` means dark: marker, patch or spectrogram ink.
* Points are ``(x, y)`` with ``x`` the column and ``y`` the row, and pixel
  centres are at integer coordinates (docs/CARD_FORMAT.md, "Pixel
  conventions"). A component's centroid is the mean of its pixel centres.

Library boundary: everything here is written with NumPy arithmetic only. No
OpenCV, SciPy or scikit-image routine is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voxgram.format import GRAY_BLACK, GRAY_WHITE

__all__ = [
    "Components",
    "box_mean",
    "connected_components",
    "default_window",
    "integral_image",
    "label_components",
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


# ---------------------------------------------------------------------------
# Connected components (issue #5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Components:
    """Statistics of the connected components of a binary image.

    Component ``k`` is described by element ``k`` of every array. Components
    are numbered in raster order of their first pixel (top to bottom, then
    left to right); in a label image component ``k`` has label ``k + 1``.

    Attributes:
        area: (K,) int64, number of pixels.
        x_min, x_max, y_min, y_max: (K,) int64 bounding box. Both ends are
            inclusive pixel indices.
        centroid: (K, 2) float64 ``(x, y)``, the mean of the pixel centres.
    """

    area: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray
    y_min: np.ndarray
    y_max: np.ndarray
    centroid: np.ndarray

    @property
    def count(self) -> int:
        return int(self.area.size)

    @property
    def width(self) -> np.ndarray:
        """(K,) int64 bounding-box width in pixels."""
        return self.x_max - self.x_min + 1

    @property
    def height(self) -> np.ndarray:
        """(K,) int64 bounding-box height in pixels."""
        return self.y_max - self.y_min + 1

    @property
    def fill_ratio(self) -> np.ndarray:
        """(K,) float64 area divided by bounding-box area, in (0, 1]."""
        return self.area / (self.width * self.height)

    @property
    def aspect_ratio(self) -> np.ndarray:
        """(K,) float64 longer bounding-box side over shorter, at least 1."""
        width = self.width
        height = self.height
        return np.maximum(width, height) / np.minimum(width, height)


def _check_connectivity(connectivity: int) -> None:
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")


def _as_binary(binary: np.ndarray) -> np.ndarray:
    binary = np.asarray(binary, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"expected a (H, W) binary image, got shape {binary.shape}")
    return binary


def _find_runs(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Horizontal runs of ``True`` pixels.

    Returns:
        ``(rows, starts, ends)``, int64 arrays of equal length in raster order.
        Run ``i`` covers columns ``starts[i] .. ends[i] - 1`` of row ``rows[i]``.
    """
    height, width = binary.shape
    padded = np.zeros((height, width + 2), dtype=np.int8)
    padded[:, 1:-1] = binary
    step = np.diff(padded, axis=1)
    rows, starts = np.nonzero(step == 1)
    _, ends = np.nonzero(step == -1)
    return rows.astype(np.int64), starts.astype(np.int64), ends.astype(np.int64)


def _touching_run_pairs(
    rows: np.ndarray, starts: np.ndarray, ends: np.ndarray, width: int, connectivity: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of runs in consecutive rows that touch.

    Run ``a`` in row ``r - 1`` and run ``b`` in row ``r`` touch when their
    column ranges overlap (4-connectivity) or overlap after widening by one
    column (8-connectivity, diagonal neighbours).

    Every row's runs are sorted and disjoint, so the runs above ``b`` that
    touch it are a contiguous block, found by binary search. Keys
    ``row * (width + 2) + column`` make one sorted array of all rows.
    """
    stride = width + 2
    reach = 1 if connectivity == 8 else 0
    start_key = rows * stride + starts
    end_key = rows * stride + ends
    above = (rows - 1) * stride
    # a touches b iff  a.start <= b.end - 1 + reach  and  a.end >= b.start + 1 - reach.
    first = np.searchsorted(end_key, above + starts + 1 - reach, side="left")
    stop = np.searchsorted(start_key, above + ends - 1 + reach, side="right")
    counts = np.maximum(stop - first, 0)

    total = int(counts.sum())
    lower = np.repeat(np.arange(rows.size), counts)
    block_start = np.repeat(np.cumsum(counts) - counts, counts)
    upper = np.repeat(first, counts) + (np.arange(total) - block_start)
    return upper, lower


def _resolve_equivalences(count: int, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
    """Merge equivalent provisional labels with union-find.

    ``parent`` is the union-find forest over the provisional labels. Each
    round compresses every path to its root, then hooks the larger root of
    every still-separate equivalence pair onto the smaller one. Parents always
    have smaller indices than their children, so the forest stays acyclic.
    The rounds are vectorized over all pairs at once.

    Returns:
        ``(final, k)``: the final label 0..k-1 of every provisional label,
        numbered in increasing order of the smallest provisional label in each
        class, and the number of classes.
    """
    parent = np.arange(count)
    while True:
        while True:
            grandparent = parent[parent]
            if np.array_equal(grandparent, parent):
                break
            parent = grandparent
        root_a = parent[a]
        root_b = parent[b]
        separate = root_a != root_b
        if not separate.any():
            break
        a = a[separate]
        b = b[separate]
        root_a = root_a[separate]
        root_b = root_b[separate]
        np.minimum.at(parent, np.maximum(root_a, root_b), np.minimum(root_a, root_b))
    roots, final = np.unique(parent, return_inverse=True)
    return final.ravel(), int(roots.size)


def _label_runs(
    binary: np.ndarray, connectivity: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Two-pass labelling over runs. Returns runs, their labels and the count."""
    _check_connectivity(connectivity)
    rows, starts, ends = _find_runs(binary)
    # First pass: every run gets a provisional label (its index), and every
    # pair of touching runs is recorded as an equivalence.
    upper, lower = _touching_run_pairs(rows, starts, ends, binary.shape[1], connectivity)
    # Resolve the equivalences; the second pass is the lookup in `labels`.
    labels, count = _resolve_equivalences(rows.size, upper, lower)
    return rows, starts, ends, labels, count


def _run_statistics(
    rows: np.ndarray, starts: np.ndarray, ends: np.ndarray, labels: np.ndarray, count: int
) -> Components:
    lengths = ends - starts
    area = np.bincount(labels, weights=lengths, minlength=count)
    # Sum of the column indices starts..ends-1 is length * (start + end - 1) / 2.
    sum_x = np.bincount(labels, weights=lengths * (starts + ends - 1) / 2, minlength=count)
    sum_y = np.bincount(labels, weights=lengths * rows, minlength=count)

    big = np.iinfo(np.int64).max
    x_min = np.full(count, big, dtype=np.int64)
    y_min = np.full(count, big, dtype=np.int64)
    x_max = np.full(count, -1, dtype=np.int64)
    y_max = np.full(count, -1, dtype=np.int64)
    np.minimum.at(x_min, labels, starts)
    np.maximum.at(x_max, labels, ends - 1)
    np.minimum.at(y_min, labels, rows)
    np.maximum.at(y_max, labels, rows)

    if count:
        centroid = np.column_stack([sum_x / area, sum_y / area])
    else:
        centroid = np.zeros((0, 2), dtype=np.float64)
    return Components(
        area=area.astype(np.int64),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        centroid=centroid,
    )


def connected_components(binary: np.ndarray, connectivity: int = 8) -> Components:
    """Statistics of the connected regions of ``True`` pixels.

    Two-pass labelling over horizontal runs rather than single pixels. The
    first pass gives every run a provisional label and records which runs in
    consecutive rows touch. The equivalences are resolved with union-find,
    and the second pass looks up the final label of every run. Working on
    runs keeps the cost proportional to the number of runs, not of pixels.

    Args:
        binary: (H, W) bool, ``True`` for foreground.
        connectivity: 8 (default) joins pixels that share an edge or a
            corner; 4 joins only pixels that share an edge.

    Returns:
        :class:`Components`, in raster order of the first pixel.
    """
    binary = _as_binary(binary)
    rows, starts, ends, labels, count = _label_runs(binary, connectivity)
    return _run_statistics(rows, starts, ends, labels, count)


def label_components(binary: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, Components]:
    """Label image and statistics of the connected regions of ``True`` pixels.

    Same algorithm as :func:`connected_components`; this also paints the
    label image, which is useful for inspection and tests.

    Returns:
        ``(labels, components)``. ``labels`` is (H, W) int32 with 0 for
        background and ``k + 1`` for component ``k``.
    """
    binary = _as_binary(binary)
    rows, starts, ends, labels, count = _label_runs(binary, connectivity)
    height, width = binary.shape
    image = np.zeros(height * width, dtype=np.int32)
    lengths = ends - starts
    run_offset = np.repeat(np.cumsum(lengths) - lengths, lengths)
    pixels = np.repeat(rows * width + starts, lengths) + (np.arange(lengths.sum()) - run_offset)
    image[pixels] = np.repeat(labels + 1, lengths)
    return image.reshape(height, width), _run_statistics(rows, starts, ends, labels, count)
