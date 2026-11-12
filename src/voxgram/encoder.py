"""Encoder: voice recording to grayscale spectrogram image.

This covers the pipeline up to the grayscale mapping. Placing the image on a
card with markers and calibration patches comes later, once the canonical
card coordinates are specified.

Command line:
    python -m voxgram.encoder recording.wav -o spectrogram.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from voxgram.audio_backend import fit_duration, load_audio, stft_magnitude
from voxgram.imageio import write_image
from voxgram.intensity import grid_to_image, magnitude_to_gray


def encode_samples(samples: np.ndarray, *, truncate: bool = False) -> np.ndarray:
    """Encode a mono signal at SAMPLE_RATE as a spectrogram image.

    Args:
        samples: (N,) signal, at most NUM_SAMPLES long unless ``truncate``.
        truncate: keep only the first 3.0 s of a longer recording.

    Returns:
        (257, 376) uint8 image. Time runs left to right, low frequency is at
        the bottom, and darker means larger magnitude.

    Raises:
        ValueError: if the recording is silent, empty, or too long.
    """
    samples = fit_duration(samples, truncate=truncate)
    gray = magnitude_to_gray(stft_magnitude(samples))
    return grid_to_image(gray)


def encode_file(path: str | Path, *, truncate: bool = False) -> np.ndarray:
    """Encode an audio file as a spectrogram image. See :func:`encode_samples`."""
    return encode_samples(load_audio(path), truncate=truncate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m voxgram.encoder",
        description="Encode a voice recording as a grayscale spectrogram image.",
    )
    parser.add_argument("audio", type=Path, help="input audio file")
    parser.add_argument(
        "-o", "--output", type=Path, help="output PNG (default: <audio name>.png next to the input)"
    )
    parser.add_argument(
        "--truncate", action="store_true", help="keep only the first 3.0 s of a longer recording"
    )
    args = parser.parse_args(argv)

    output = args.output or args.audio.with_suffix(".png")
    try:
        image = encode_file(args.audio, truncate=args.truncate)
    except ValueError as exc:
        parser.exit(1, f"error: {exc}\n")
    write_image(output, image)
    print(f"wrote {output} ({image.shape[1]} x {image.shape[0]} px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
