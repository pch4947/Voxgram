"""Marker detection: grayscale conversion, thresholding, connected components.

This is the first stage of the decoder. It turns a photograph into a grayscale
image, makes a binary working image from it, labels the dark regions of that
binary image and finds the four fiducial markers among them.

The binary image exists only for detection. The grayscale image returned by
:func:`to_grayscale` is what rectification and extraction work on, unchanged.

Conventions
-----------
* Gray values are float64 in [0, 255], 0 black and 255 white.
* In binary images ``True`` means dark: marker, patch or spectrogram ink.
* Points are ``(x, y)`` with ``x`` the column and ``y`` the row, and pixel
  centres are at integer coordinates (docs/CARD_FORMAT.md, "Pixel
  conventions"). A component's centroid is the mean of its pixel centres.

Markers
-------
A marker is three concentric squares: a black frame, a white gap and a black
core. After thresholding and labelling it is two dark components, a frame and
a core, whose centroids coincide. The orientation marker has a smaller core,
so its frame-to-core area ratio is several times larger than that of the
three finder markers. Detection relies on these properties and on the four
markers forming a convex quadrilateral. From ``voxgram.format`` it takes only
the marker proportions, which give the expected area ratios, and the index of
the orientation marker. The marker size in pixels is free, so the photograph
can be taken at any distance.

Marker order follows the card: top-left, top-right, bottom-right (the
orientation marker), bottom-left. That order is clockwise on the card, and a
photograph of the front of a flat card preserves the sense of rotation, so the
order is recovered for any in-plane rotation, including upside down.

Library boundary: everything here is written with NumPy arithmetic only. No
OpenCV, SciPy or scikit-image routine is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from voxgram.format import (
    CUE_MARKER_INDEX,
    GRAY_BLACK,
    GRAY_WHITE,
    MARKER_CENTERS_MM,
    MARKER_CORE_MM,
    MARKER_CUE_CORE_MM,
    MARKER_GAP_MM,
    MARKER_SIZE_MM,
)

__all__ = [
    "Components",
    "MarkerCandidate",
    "MarkerDetectionError",
    "box_mean",
    "connected_components",
    "default_window",
    "detect_markers",
    "find_marker_candidates",
    "integral_image",
    "label_components",
    "otsu_threshold",
    "refine_center",
    "select_markers",
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


# ---------------------------------------------------------------------------
# Marker detection (issue #6)
# ---------------------------------------------------------------------------

# Component filters. Areas and sizes are in pixels.
MIN_COMPONENT_AREA = 9
MAX_ASPECT_RATIO = 3.0
# A square frame fills 49 % of its bounding box upright and 24.5 % at 45 degrees.
FRAME_FILL_RANGE = (0.15, 0.75)
# A solid square fills 100 % upright and 50 % at 45 degrees.
MIN_CORE_FILL = 0.35
# Frame and core centroids may differ by this fraction of the frame size.
CENTRE_TOLERANCE = 0.15

# What the card format prints, as ratios that do not depend on the scale of
# the photograph. Format 0.3-pilot: frame 24 mm2, so the frame-to-core area
# ratio is 24 / 9 = 2.67 for a finder and 24 / 2.25 = 10.67 for the
# orientation marker; core side over frame side is 3 / 7 and 1.5 / 7.
FINDER = "finder"
CUE = "cue"
_FRAME_AREA_MM2 = MARKER_SIZE_MM**2 - MARKER_GAP_MM**2
FINDER_AREA_RATIO = _FRAME_AREA_MM2 / MARKER_CORE_MM**2
CUE_AREA_RATIO = _FRAME_AREA_MM2 / MARKER_CUE_CORE_MM**2
FINDER_SIDE_RATIO = MARKER_CORE_MM / MARKER_SIZE_MM
CUE_SIDE_RATIO = MARKER_CUE_CORE_MM / MARKER_SIZE_MM
# A measured ratio matches a kind when it is within this factor of the printed
# value either way. Thresholding and blur move the edge of the small core by
# about a pixel, which changes its area by tens of percent.
RATIO_TOLERANCE = 1.6
# All four markers must be in the photograph, so the shortest distance between
# neighbouring markers (84 mm in 0.3-pilot) fits in the longer image side, and
# a marker (7 mm) is at most 1/12 of that side. Frames larger than twice this
# fraction are not markers, whatever their shape; perspective and a rotated
# bounding box account for the factor of two.
_MARKER_SPACING_MM = min(
    float(np.hypot(*np.subtract(MARKER_CENTERS_MM[i], MARKER_CENTERS_MM[(i + 1) % 4])))
    for i in range(4)
)
MAX_MARKER_FRACTION = 2 * MARKER_SIZE_MM / _MARKER_SPACING_MM

# Arrangement filters for a set of four markers.
# Candidates considered, largest frames first. 12 gives 495 sets of four.
MAX_CANDIDATES = 12
# Largest over smallest frame area; allows for perspective.
MAX_FRAME_AREA_SPREAD = 4.0
# The distance between neighbouring markers, converted to millimetres with the
# scale of the two markers themselves, must be within this factor of the
# distance in MARKER_CENTERS_MM. A tilt of 60 degrees gives about 1.41.
EDGE_LENGTH_TOLERANCE = 1.5


def _within(measured: float, printed: float, tolerance: float) -> bool:
    return printed / tolerance <= measured <= printed * tolerance


class MarkerDetectionError(ValueError):
    """The four markers could not be found unambiguously."""


@dataclass(frozen=True)
class MarkerCandidate:
    """A frame component with a core component at its centre.

    Attributes:
        center: ``(x, y)`` in pixels, the centroid of frame and core pixels
            together in the binary image.
        frame_area, core_area: pixel counts of the two components.
        size: frame size in pixels, the square root of its bounding-box area.
        bbox: frame bounding box ``(x_min, y_min, x_max, y_max)``, inclusive.
    """

    center: tuple[float, float]
    frame_area: int
    core_area: int
    size: float
    bbox: tuple[int, int, int, int] = (0, 0, -1, -1)

    @property
    def area_ratio(self) -> float:
        """Frame area over core area. Larger for the orientation marker."""
        return self.frame_area / self.core_area

    @property
    def kind(self) -> str | None:
        """``FINDER``, ``CUE`` or ``None``, from the frame-to-core area ratio."""
        if _within(self.area_ratio, FINDER_AREA_RATIO, RATIO_TOLERANCE):
            return FINDER
        if _within(self.area_ratio, CUE_AREA_RATIO, RATIO_TOLERANCE):
            return CUE
        return None

    @property
    def pixels_per_mm(self) -> float:
        """Local scale of the photograph, from the printed frame area."""
        return float(np.sqrt(self.frame_area / _FRAME_AREA_MM2))


def find_marker_candidates(binary: np.ndarray) -> list[MarkerCandidate]:
    """Find every frame-and-core pair in a binary image that looks like a marker.

    A frame is a hollow, roughly square component. A core is a solid,
    roughly square component inside the frame's bounding box whose centroid
    is close to the frame's. The pair is kept when its frame-to-core area
    ratio matches a finder or the orientation marker, its core-to-frame size
    ratio matches the same kind, and the frame is not too large to be a
    marker (``MAX_MARKER_FRACTION`` of the longer image side). Of several possible cores the closest
    is taken.

    Args:
        binary: (H, W) bool, ``True`` for dark pixels.

    Returns:
        Candidates, largest frame first. A busy image can give more than four;
        :func:`select_markers` chooses among them.
    """
    components = connected_components(binary, connectivity=8)
    if components.count == 0:
        return []
    fill = components.fill_ratio
    plausible = (components.area >= MIN_COMPONENT_AREA) & (
        components.aspect_ratio <= MAX_ASPECT_RATIO
    )
    size = np.sqrt(components.width * components.height)
    small_enough = size <= MAX_MARKER_FRACTION * max(binary.shape)
    frames = np.flatnonzero(
        plausible & small_enough & (fill >= FRAME_FILL_RANGE[0]) & (fill <= FRAME_FILL_RANGE[1])
    )
    cores = np.flatnonzero(plausible & (fill >= MIN_CORE_FILL))
    if frames.size == 0 or cores.size == 0:
        return []

    area = components.area
    centroid = components.centroid
    core_x_min = components.x_min[cores]
    core_x_max = components.x_max[cores]
    core_y_min = components.y_min[cores]
    core_y_max = components.y_max[cores]
    core_centroid = centroid[cores]
    core_area = area[cores]
    core_size = size[cores]

    candidates = []
    for frame in frames:
        inside = (
            (core_x_min > components.x_min[frame])
            & (core_x_max < components.x_max[frame])
            & (core_y_min > components.y_min[frame])
            & (core_y_max < components.y_max[frame])
        )
        if not inside.any():
            continue
        distance = np.hypot(*(core_centroid - centroid[frame]).T)
        match = inside & (distance <= CENTRE_TOLERANCE * size[frame])
        for best in np.flatnonzero(match)[np.argsort(distance[match])]:
            candidate = MarkerCandidate(
                center=_combined_centroid(
                    area[frame], centroid[frame], core_area[best], core_centroid[best]
                ),
                frame_area=int(area[frame]),
                core_area=int(core_area[best]),
                size=float(size[frame]),
                bbox=(
                    int(components.x_min[frame]),
                    int(components.y_min[frame]),
                    int(components.x_max[frame]),
                    int(components.y_max[frame]),
                ),
            )
            side_ratio = core_size[best] / size[frame]
            printed = {FINDER: FINDER_SIDE_RATIO, CUE: CUE_SIDE_RATIO}.get(candidate.kind)
            if printed is not None and _within(side_ratio, printed, RATIO_TOLERANCE):
                candidates.append(candidate)
                break
    candidates.sort(key=lambda candidate: candidate.frame_area, reverse=True)
    return candidates


def _combined_centroid(
    area_a: float, centroid_a: np.ndarray, area_b: float, centroid_b: np.ndarray
) -> tuple[float, float]:
    center = (area_a * centroid_a + area_b * centroid_b) / (area_a + area_b)
    return float(center[0]), float(center[1])


def _clockwise_order(points: np.ndarray) -> np.ndarray:
    """Indices that sort points clockwise as seen in the image (y down)."""
    offset = points - points.mean(axis=0)
    return np.argsort(np.arctan2(offset[:, 1], offset[:, 0]))


def _order_markers(group: tuple[MarkerCandidate, ...]) -> tuple[MarkerCandidate, ...]:
    """The four markers in card order. ``group`` has exactly one ``CUE``.

    ``MARKER_CENTERS_MM`` lists the markers clockwise on the card, and a
    photograph keeps that sense of rotation, so the clockwise order in the
    image only has to be rotated until the orientation marker sits at
    ``CUE_MARKER_INDEX``.
    """
    points = np.array([c.center for c in group], dtype=np.float64)
    clockwise = list(_clockwise_order(points))
    cue_position = next(k for k, i in enumerate(clockwise) if group[i].kind == CUE)
    shift = cue_position - CUE_MARKER_INDEX
    return tuple(group[i] for i in np.roll(clockwise, -shift))


def _is_consistent(group: tuple[MarkerCandidate, ...]) -> bool:
    """Whether four candidates can be the four markers of one card."""
    kinds = [c.kind for c in group]
    if kinds.count(CUE) != 1 or kinds.count(FINDER) != 3:
        return False
    frame_area = np.array([c.frame_area for c in group], dtype=np.float64)
    if frame_area.max() > MAX_FRAME_AREA_SPREAD * frame_area.min():
        return False

    ordered = _order_markers(group)
    points = np.array([c.center for c in ordered])

    # Convex, with no three points in a line: consecutive edges turn the
    # same way all the way round.
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not np.all(turns > 0):
        return False

    # Neighbouring markers as far apart as the card prints them.
    printed = np.array(MARKER_CENTERS_MM, dtype=np.float64)
    for i in range(4):
        j = (i + 1) % 4
        scale = (ordered[i].pixels_per_mm + ordered[j].pixels_per_mm) / 2
        measured = np.hypot(*(points[j] - points[i])) / scale
        if not _within(measured, np.hypot(*(printed[j] - printed[i])), EDGE_LENGTH_TOLERANCE):
            return False
    return True


def select_markers(candidates: list[MarkerCandidate]) -> tuple[MarkerCandidate, ...]:
    """Choose the four markers among candidates and put them in card order.

    Every set of four among the ``MAX_CANDIDATES`` largest candidates is
    checked for: three finders and one orientation marker, similar frame
    sizes, a convex quadrilateral, and distances between neighbouring markers
    that match ``MARKER_CENTERS_MM`` at the scale of the markers.

    Returns:
        The four candidates in the order of ``MARKER_CENTERS_MM``: top-left,
        top-right, bottom-right, bottom-left.

    Raises:
        MarkerDetectionError: if no set or more than one set passes.
    """
    if len(candidates) < 4:
        raise MarkerDetectionError(
            f"found {len(candidates)} marker candidate(s), need 4; "
            "a marker may be covered, cut off by the image edge, blurred or too small"
        )
    pool = candidates[:MAX_CANDIDATES]
    consistent = [group for group in combinations(pool, 4) if _is_consistent(group)]
    if not consistent:
        cues = sum(c.kind == CUE for c in pool)
        raise MarkerDetectionError(
            f"no four of the {len(pool)} marker candidates ({cues} orientation) form a "
            "card: a marker is missing, or the orientation marker cannot be told apart"
        )
    if len(consistent) > 1:
        raise MarkerDetectionError(
            f"{len(consistent)} different sets of four candidates could be the markers; "
            "refusing to guess"
        )
    return _order_markers(consistent[0])


# Centre refinement. Lengths are fractions of the frame size, with a floor in
# pixels. The card leaves at least 3.5 mm of white paper around every marker
# (docs/CARD_FORMAT.md), which is what the paper ring below relies on.
# Width of the partly inked fringe outside the frame pixels.
REFINE_FRINGE = (0.03, 2)
# Width of the paper ring outside the fringe, used as the white reference.
REFINE_RING = (0.10, 3)
# Percentile of the reflectance inside the marker taken as the ink reference.
REFINE_INK_PERCENTILE = 5.0
# Smallest paper-to-ink contrast, as a reflectance difference, worth refining.
REFINE_MIN_CONTRAST = 0.1


def _scaled(size: float, rule: tuple[float, int]) -> int:
    fraction, floor = rule
    return max(floor, int(np.ceil(fraction * size)))


def refine_center(
    gray: np.ndarray, binary: np.ndarray, candidate: MarkerCandidate
) -> tuple[float, float]:
    """Sub-pixel marker centre from the grayscale image.

    The binary centroid inherits any bias of the threshold: where the local
    mean differs on two sides of a marker, its edges are cut at different
    gray levels and the centroid moves. This instead weights pixels by how
    much ink covers them, so partly covered edge pixels count in proportion
    and the result does not depend on where the threshold cut.

    1. The frame component is found again in a window of ``binary``. Its
       farthest pixel from the binary centre gives the marker radius ``R``.
    2. The paper brightness is modelled as a plane, fitted by least squares
       to the light pixels of a ring just outside ``R``. Dividing ``gray`` by
       it removes the illumination and leaves reflectance, so a brightness
       gradient across the marker does not tilt the weights.
    3. Inside radius ``R`` plus a small fringe, each pixel is weighted by
       ``(1 - reflectance) / (1 - ink)`` clipped to [0, 1], where ``ink`` is a
       low percentile of the reflectance there. The centre is the weighted
       mean of the pixel centres.

    The disc rather than the bounding box limits the sum to the marker: the
    bounding box of a rotated marker reaches towards the nearby patches.

    Args:
        gray: (H, W) float64 in [0, 255].
        binary: (H, W) bool, the image the candidate was found in.
        candidate: a candidate from :func:`find_marker_candidates`.

    Returns:
        ``(x, y)`` in pixels. The binary centre if the frame cannot be found
        again, the paper ring is too small or the marker has no contrast.
    """
    gray = _as_gray(gray)
    binary = _as_binary(binary)
    height, width = gray.shape
    x_min, y_min, x_max, y_max = candidate.bbox
    if x_max < x_min or y_max < y_min:
        return candidate.center
    fringe = _scaled(candidate.size, REFINE_FRINGE)
    ring = _scaled(candidate.size, REFINE_RING)
    reach = int(np.ceil(0.5 * np.hypot(x_max - x_min + 1, y_max - y_min + 1))) + fringe + ring
    cx, cy = candidate.center
    x0, x1 = max(int(cx) - reach, 0), min(int(cx) + reach + 1, width)
    y0, y1 = max(int(cy) - reach, 0), min(int(cy) + reach + 1, height)

    # 1. The frame, by its bounding box, among the components of the window.
    dark = binary[y0:y1, x0:x1]
    labels, components = label_components(dark)
    match = np.flatnonzero(
        (components.x_min == x_min - x0)
        & (components.y_min == y_min - y0)
        & (components.x_max == x_max - x0)
        & (components.y_max == y_max - y0)
    )
    if match.size != 1:
        return candidate.center
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance = np.hypot(xx - cx, yy - cy)
    radius = distance[labels == match[0] + 1].max()

    # 2. Paper brightness from the ring outside the marker.
    paper = (distance > radius + fringe) & (distance <= radius + fringe + ring) & ~dark
    if np.count_nonzero(paper) < 3:
        return candidate.center
    window = gray[y0:y1, x0:x1]
    design = np.column_stack([np.ones(np.count_nonzero(paper)), xx[paper], yy[paper]])
    plane, *_ = np.linalg.lstsq(design, window[paper], rcond=None)

    # 3. Ink coverage inside the marker.
    inside = distance <= radius + fringe
    white = plane[0] + plane[1] * xx[inside] + plane[2] * yy[inside]
    if np.any(white <= 0):
        return candidate.center
    reflectance = window[inside] / white
    ink = np.percentile(reflectance, REFINE_INK_PERCENTILE)
    if 1.0 - ink < REFINE_MIN_CONTRAST:
        return candidate.center
    weight = np.clip((1.0 - reflectance) / (1.0 - ink), 0.0, 1.0)
    total = weight.sum()
    return float(weight @ xx[inside] / total), float(weight @ yy[inside] / total)


def detect_markers(
    gray: np.ndarray,
    *,
    method: str = "local",
    window: int | None = None,
    offset: float = DEFAULT_LOCAL_OFFSET,
    threshold: float | None = None,
) -> np.ndarray:
    """Find the four fiducial markers in a grayscale photograph.

    Args:
        gray: (H, W) float64 in [0, 255], from :func:`to_grayscale`.
        method: ``"local"`` for the local-mean threshold (default) or
            ``"global"`` for the global baseline.
        window: local-mean window, odd, in pixels. Defaults to
            :func:`default_window`.
        offset: local-mean darkness margin in gray levels.
        threshold: global threshold. Defaults to :func:`otsu_threshold`.

    Returns:
        (4, 2) float64 marker centres ``(x, y)`` in photograph pixels, pixel
        centres at integers, in the card order top-left, top-right,
        bottom-right, bottom-left, the order of ``MARKER_CENTERS_MM``; row
        ``CUE_MARKER_INDEX`` is the orientation marker. Pair row ``i`` with
        ``MARKER_CENTERS_MM[i]`` to estimate the homography.
        The binary image locates and identifies the markers; the centres are
        then refined on ``gray`` with :func:`refine_center`.

    Raises:
        MarkerDetectionError: if the markers are missing or ambiguous. The
            decoder must stop here rather than produce audio.
    """
    gray = _as_gray(gray)
    if method == "local":
        binary = threshold_local_mean(
            gray, default_window(gray.shape) if window is None else window, offset
        )
    elif method == "global":
        binary = threshold_global(gray, otsu_threshold(gray) if threshold is None else threshold)
    else:
        raise ValueError(f"method must be 'local' or 'global', got {method!r}")
    markers = select_markers(find_marker_candidates(binary))
    return np.array([refine_center(gray, binary, m) for m in markers], dtype=np.float64)
