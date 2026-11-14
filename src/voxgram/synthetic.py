"""Synthetic photographs of a card, with known ground truth.

A synthetic photograph is the card function from voxgram.card evaluated
through a known homography, optionally followed by a known illumination
gradient and tone curve. Because every step is known, detection, geometry,
calibration and extraction can be tested against exact answers without
printing or photographing anything.

This is test support. It does not replace real captures: there is no blur,
halftone pattern, sensor noise or lens distortion here.

Conventions shared with the rest of the pipeline:

* A homography ``H`` maps card millimetres to photograph pixels,
  ``[x_px, y_px, 1]^T ~ H [x_mm, y_mm, 1]^T``.
* Points are (x, y). In a photograph x is the column and y is the row, and
  pixel centres are at integer coordinates.
"""

from __future__ import annotations

import numpy as np

from voxgram.card import render_view
from voxgram.format import CARD_HEIGHT_MM, CARD_WIDTH_MM, GRAY_WHITE

__all__ = [
    "apply_tone_curve",
    "as_bgr",
    "illumination_gradient",
    "make_homography",
    "project_points",
    "render_view",
]


def make_homography(
    *,
    scale: float,
    center: tuple[float, float],
    rotation_deg: float = 0.0,
    perspective: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Build a card-to-photograph homography from viewing parameters.

    Args:
        scale: pixels per millimetre at the card centre.
        center: photograph pixel (x, y) where the card centre lands.
        rotation_deg: in-plane rotation, clockwise as seen in the image.
        perspective: (px, py) per millimetre. The projective divisor is
            ``1 + px * u + py * v`` with (u, v) measured from the card
            centre, so 0.001 makes the far edge of the 150 mm side look
            about 15 % smaller than the near edge. Zero means no tilt.

    Returns:
        (3, 3) float64 matrix, normalized so that ``H[2, 2]`` is 1.
    """
    to_center = np.array(
        [[1.0, 0.0, -CARD_WIDTH_MM / 2], [0.0, 1.0, -CARD_HEIGHT_MM / 2], [0.0, 0.0, 1.0]]
    )
    tilt = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [perspective[0], perspective[1], 1.0]])
    angle = np.deg2rad(rotation_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    place = np.array(
        [[scale * cos, -scale * sin, center[0]], [scale * sin, scale * cos, center[1]], [0, 0, 1.0]]
    )
    homography = place @ tilt @ to_center
    return homography / homography[2, 2]


def project_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homography to an (N, 2) array of (x, y) points."""
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ np.asarray(homography).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def illumination_gradient(
    shape: tuple[int, int], *, low: float, high: float, angle_deg: float = 0.0
) -> np.ndarray:
    """A linear brightness ramp across the photograph.

    Args:
        shape: (height, width).
        low, high: gain at the darkest and brightest corner, for example
            0.6 and 1.0.
        angle_deg: direction in which brightness increases. 0 is left to
            right, 90 is top to bottom.

    Returns:
        (height, width) float64 gain field. Multiply a photograph by it.
    """
    height, width = shape
    angle = np.deg2rad(angle_deg)
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ramp = x * np.cos(angle) + y * np.sin(angle)
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())
    return low + (high - low) * ramp


def apply_tone_curve(photo: np.ndarray, gamma: float) -> np.ndarray:
    """Apply ``255 * (value / 255) ** gamma``. Monotonic; fixes 0 and 255."""
    normalized = np.clip(np.asarray(photo, dtype=np.float64) / GRAY_WHITE, 0.0, 1.0)
    return GRAY_WHITE * normalized**gamma


def as_bgr(photo: np.ndarray) -> np.ndarray:
    """Round to uint8 and repeat over three channels.

    The result has the layout that ``voxgram.imageio.read_image`` returns for
    a colour photograph: (height, width, 3) in BGR order.
    """
    gray = np.rint(np.clip(photo, 0, GRAY_WHITE)).astype(np.uint8)
    return np.repeat(gray[:, :, np.newaxis], 3, axis=2)
