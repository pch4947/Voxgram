"""Image file input and output.

This is the only place OpenCV is used. Files are read and written through
NumPy buffers because cv2.imread and cv2.imwrite do not handle non-ASCII
Windows paths reliably.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import cv2
import numpy as np

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Signature, then the IHDR chunk: length, type, 13 bytes of data, CRC.
_PNG_IHDR_END = len(_PNG_SIGNATURE) + 4 + 4 + 13 + 4


def _png_with_dpi(data: bytes, dpi: float) -> bytes:
    """Insert a pHYs chunk stating the print resolution into PNG bytes."""
    if not data.startswith(_PNG_SIGNATURE) or data[12:16] != b"IHDR":
        raise ValueError("not a PNG stream")
    if b"pHYs" in data[: _PNG_IHDR_END + 64]:
        raise ValueError("PNG stream already states a resolution")
    pixels_per_metre = round(dpi / 0.0254)
    body = b"pHYs" + struct.pack(">IIB", pixels_per_metre, pixels_per_metre, 1)
    chunk = struct.pack(">I", 9) + body + struct.pack(">I", zlib.crc32(body))
    return data[:_PNG_IHDR_END] + chunk + data[_PNG_IHDR_END:]


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


def write_image(path: str | Path, image: np.ndarray, *, dpi: float | None = None) -> None:
    """Write an image file. The format follows the file extension.

    Use ``.png`` for anything that must survive unchanged; it is lossless.

    Args:
        path: output file.
        image: (H, W) or (H, W, 3) uint8 array.
        dpi: print resolution to record in the file, PNG only. With it, a
            print dialog set to actual size prints the image at its intended
            physical size.

    Raises:
        ValueError: if the image cannot be encoded for that extension, or
            ``dpi`` is given for a format other than PNG.
    """
    path = Path(path)
    if dpi is not None and path.suffix.lower() != ".png":
        raise ValueError("dpi can only be recorded in PNG files")
    ok, buffer = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"could not encode image as {path.suffix}")
    data = buffer.tobytes()
    if dpi is not None:
        data = _png_with_dpi(data, dpi)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
