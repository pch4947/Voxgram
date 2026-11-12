"""Audio I/O and the fixed reconstruction backend.

Every Librosa call in the project goes through this module. The STFT and
Griffin-Lim settings come from voxgram.format and are the same for every
experimental condition, so differences between conditions come from the
image-processing front end and not from the audio side.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from voxgram.format import (
    CENTER,
    GRID_SHAPE,
    HOP_LENGTH,
    N_FFT,
    NUM_SAMPLES,
    PAD_MODE,
    SAMPLE_RATE,
    WINDOW,
)

# Griffin-Lim settings. Not part of the card format, but held fixed.
GRIFFIN_LIM_ITERATIONS = 100
GRIFFIN_LIM_MOMENTUM = 0.99
GRIFFIN_LIM_SEED = 0

# Peak level of written WAV files. The card does not keep absolute level.
OUTPUT_PEAK = 0.9


def load_audio(path: str | Path) -> np.ndarray:
    """Read an audio file as mono float32 at SAMPLE_RATE.

    The length is not changed. Use :func:`fit_duration` for that.
    """
    samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if rate != SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=rate, target_sr=SAMPLE_RATE)
    return np.ascontiguousarray(mono, dtype=np.float32)


def fit_duration(samples: np.ndarray, *, truncate: bool = False) -> np.ndarray:
    """Pad a recording with trailing zeros to exactly NUM_SAMPLES.

    Args:
        samples: (N,) mono signal at SAMPLE_RATE.
        truncate: keep only the first NUM_SAMPLES of a longer recording
            instead of raising.

    Raises:
        ValueError: if the signal is not 1-D, is empty, or is longer than
            NUM_SAMPLES while ``truncate`` is False.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError(f"expected a mono signal, got shape {samples.shape}")
    if samples.size == 0:
        raise ValueError("recording is empty")
    if samples.size > NUM_SAMPLES:
        if not truncate:
            seconds = samples.size / SAMPLE_RATE
            limit = NUM_SAMPLES / SAMPLE_RATE
            raise ValueError(
                f"recording is {seconds:.2f} s, longer than the {limit:.1f} s format; "
                "trim it or allow truncation"
            )
        return samples[:NUM_SAMPLES].copy()
    return np.pad(samples, (0, NUM_SAMPLES - samples.size))


def save_audio(path: str | Path, samples: np.ndarray, *, normalize: bool = True) -> None:
    """Write a mono 16-bit WAV file at SAMPLE_RATE.

    With ``normalize`` the signal is scaled so its peak is OUTPUT_PEAK.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if normalize:
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 0:
            samples = samples * (OUTPUT_PEAK / peak)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, SAMPLE_RATE, subtype="PCM_16")


def stft_magnitude(samples: np.ndarray) -> np.ndarray:
    """STFT magnitude of a NUM_SAMPLES signal.

    Returns:
        (257, 376) float32 grid, row 0 is the lowest frequency bin.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if samples.shape != (NUM_SAMPLES,):
        raise ValueError(f"expected shape ({NUM_SAMPLES},), got {samples.shape}")
    spectrum = librosa.stft(
        samples,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=WINDOW,
        center=CENTER,
        pad_mode=PAD_MODE,
    )
    return np.abs(spectrum)


def griffin_lim(magnitude: np.ndarray, *, n_iter: int = GRIFFIN_LIM_ITERATIONS) -> np.ndarray:
    """Estimate a NUM_SAMPLES signal whose STFT magnitude matches the grid.

    Deterministic: the random phase initialization uses a fixed seed.

    Args:
        magnitude: (257, 376) non-negative grid, row 0 is the lowest bin.
        n_iter: number of iterations. Leave at the default for experiments.

    Returns:
        (NUM_SAMPLES,) float32 signal.
    """
    magnitude = np.asarray(magnitude, dtype=np.float32)
    if magnitude.shape != GRID_SHAPE:
        raise ValueError(f"expected shape {GRID_SHAPE}, got {magnitude.shape}")
    samples = librosa.griffinlim(
        magnitude,
        n_iter=n_iter,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        window=WINDOW,
        center=CENTER,
        pad_mode=PAD_MODE,
        momentum=GRIFFIN_LIM_MOMENTUM,
        init="random",
        random_state=GRIFFIN_LIM_SEED,
        length=NUM_SAMPLES,
    )
    return samples.astype(np.float32)
