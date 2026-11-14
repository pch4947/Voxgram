"""Encoder: voice recording to grayscale spectrogram image.

The spectrogram image is the 257 x 376 grid itself. The printable card, with
markers and calibration patches around that image, is drawn by voxgram.card.

Command line:
    python -m voxgram.encoder recording.wav -o spectrogram.png
    python -m voxgram.encoder recording.wav --card card.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from voxgram.audio_backend import fit_duration, load_audio, stft_magnitude
from voxgram.card import render_card
from voxgram.format import CARD_HEIGHT_MM, CARD_WIDTH_MM, TARGET_PRINT_DPI
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
    parser.add_argument(
        "--card", type=Path, metavar="PNG", help="also write the print-ready card to this PNG"
    )
    parser.add_argument(
        "--dpi",
        type=float,
        default=TARGET_PRINT_DPI,
        help=f"print resolution of the card (default: {TARGET_PRINT_DPI})",
    )
    args = parser.parse_args(argv)

    if args.card is not None and args.card.suffix.lower() != ".png":
        parser.error("--card must be a .png file")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    output = args.output or args.audio.with_suffix(".png")
    try:
        image = encode_file(args.audio, truncate=args.truncate)
    except ValueError as exc:
        parser.exit(1, f"error: {exc}\n")
    write_image(output, image)
    print(f"wrote {output} ({image.shape[1]} x {image.shape[0]} px)")

    if args.card is not None:
        card = render_card(image, dpi=args.dpi)
        write_image(args.card, card, dpi=args.dpi)
        print(
            f"wrote {args.card} ({card.shape[1]} x {card.shape[0]} px at {args.dpi:g} dpi, "
            f"prints at {CARD_WIDTH_MM:g} x {CARD_HEIGHT_MM:g} mm at actual size)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
