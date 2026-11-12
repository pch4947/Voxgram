"""Image file input and output.

This is the only place OpenCV is used. Files are read and written through
NumPy buffers because cv2.imread and cv2.imwrite do not handle non-ASCII
Windows paths reliably.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    """Read an image file without converting it.

    Returns:
        (H, W) for a single-channel file, or (H, W, 3) / (H, W, 4) in OpenCV's
        BGR / BGRA channel order. Grayscale conversion of a colour photograph
        is the detection module's job, not this function's.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file cannot be decoded as an image.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    buffer = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write an image file. The format follows the file extension.

    Use ``.png`` for anything that must survive unchanged; it is lossless.

    Raises:
        ValueError: if the image cannot be encoded for that extension.
    """
    path = Path(path)
    ok, buffer = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"could not encode image as {path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer.tofile(path)
