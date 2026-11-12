"""Digital baseline: recording to grayscale image and back, with no printing.

The round trip writes three audio files so the sources of loss can be heard
separately:

    1_original.wav          the padded input, as the encoder sees it
    2_griffinlim_only.wav   Griffin-Lim from the exact STFT magnitude:
                            phase loss only
    3_digital_baseline.wav  Griffin-Lim from the magnitude decoded from the
                            8-bit image: phase loss, the -60 dB floor and
                            quantization

A photographed card cannot sound better than file 3, so later experiments are
compared against it. If file 3 is clearly worse than file 2, the dynamic range
or bit depth is the problem; if file 2 is already poor, the STFT settings or
the duration are.

Command line:
    python -m voxgram.baseline recording.wav -o outputs/baseline
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voxgram.audio_backend import (
    fit_duration,
    griffin_lim,
    load_audio,
    save_audio,
    stft_magnitude,
)
from voxgram.format import DB_FLOOR
from voxgram.imageio import read_image, write_image
from voxgram.intensity import (
    gray_to_magnitude,
    grid_to_image,
    image_to_grid,
    magnitude_to_gray,
)

IMAGE_NAME = "spectrogram.png"
ORIGINAL_NAME = "1_original.wav"
GRIFFIN_LIM_NAME = "2_griffinlim_only.wav"
BASELINE_NAME = "3_digital_baseline.wav"


@dataclass(frozen=True)
class BaselineReport:
    """Numbers from one digital round trip.

    Attributes:
        image_exact: the image read back from disk equals the encoded one.
        floor_fraction: share of grid cells clipped at the dB floor.
        quantization_error_db: largest magnitude error, in dB, among cells
            above the floor. Half a gray level at most.
        convergence_griffinlim: spectral convergence of file 2 against the
            exact magnitude. 0 is perfect.
        convergence_baseline: spectral convergence of file 3 against the
            decoded magnitude.
    """

    image_exact: bool
    floor_fraction: float
    quantization_error_db: float
    convergence_griffinlim: float
    convergence_baseline: float

    def lines(self) -> list[str]:
        return [
            f"image read back exactly:        {self.image_exact}",
            f"cells clipped at {DB_FLOOR:.0f} dB floor:   {self.floor_fraction:.1%}",
            f"max quantization error:         {self.quantization_error_db:.3f} dB",
            f"spectral convergence, GL only:  {self.convergence_griffinlim:.4f}",
            f"spectral convergence, baseline: {self.convergence_baseline:.4f}",
        ]


def spectral_convergence(samples: np.ndarray, target_magnitude: np.ndarray) -> float:
    """Relative distance between a signal's STFT magnitude and a target.

    ``||target - |STFT(samples)|| / ||target||`` in the Frobenius norm.
    """
    achieved = stft_magnitude(samples).astype(np.float64)
    target = np.asarray(target_magnitude, dtype=np.float64)
    return float(np.linalg.norm(target - achieved) / np.linalg.norm(target))


def run_baseline(
    samples: np.ndarray, output_dir: str | Path, *, truncate: bool = False
) -> BaselineReport:
    """Run the digital round trip and write the image and audio files.

    Args:
        samples: (N,) mono signal at SAMPLE_RATE.
        output_dir: directory for the outputs; created if missing.
        truncate: keep only the first 3.0 s of a longer recording.
    """
    output_dir = Path(output_dir)
    samples = fit_duration(samples, truncate=truncate)

    # Encode, write the image, and decode from the file that was written.
    magnitude = stft_magnitude(samples).astype(np.float64)
    encoded_image = grid_to_image(magnitude_to_gray(magnitude))
    image_path = output_dir / IMAGE_NAME
    write_image(image_path, encoded_image)
    decoded_image = read_image(image_path)
    image_exact = decoded_image.shape == encoded_image.shape and bool(
        np.array_equal(decoded_image, encoded_image)
    )
    decoded_magnitude = gray_to_magnitude(image_to_grid(decoded_image))

    from_exact = griffin_lim(magnitude)
    from_image = griffin_lim(decoded_magnitude)

    save_audio(output_dir / ORIGINAL_NAME, samples)
    save_audio(output_dir / GRIFFIN_LIM_NAME, from_exact)
    save_audio(output_dir / BASELINE_NAME, from_image)

    normalized = magnitude / magnitude.max()
    above_floor = normalized > 10.0 ** (DB_FLOOR / 20.0)
    error_db = np.abs(
        20.0 * np.log10(normalized[above_floor]) - 20.0 * np.log10(decoded_magnitude[above_floor])
    )

    return BaselineReport(
        image_exact=image_exact,
        floor_fraction=float(1.0 - above_floor.mean()),
        quantization_error_db=float(error_db.max()) if error_db.size else 0.0,
        convergence_griffinlim=spectral_convergence(from_exact, magnitude),
        convergence_baseline=spectral_convergence(from_image, decoded_magnitude),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m voxgram.baseline",
        description="Round-trip a recording through the grayscale image without printing.",
    )
    parser.add_argument("audio", type=Path, help="input audio file")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="output directory (default: outputs/baseline/<audio name>)",
    )
    parser.add_argument(
        "--truncate", action="store_true", help="keep only the first 3.0 s of a longer recording"
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir or Path("outputs") / "baseline" / args.audio.stem
    try:
        report = run_baseline(load_audio(args.audio), output_dir, truncate=args.truncate)
    except ValueError as exc:
        parser.exit(1, f"error: {exc}\n")

    print(f"wrote {output_dir}")
    for line in report.lines():
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
