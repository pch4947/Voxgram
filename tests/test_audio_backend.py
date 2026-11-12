"""Tests for audio I/O and the Librosa wrapper."""

import numpy as np
import pytest
import soundfile as sf

from voxgram.audio_backend import (
    OUTPUT_PEAK,
    fit_duration,
    griffin_lim,
    load_audio,
    save_audio,
    stft_magnitude,
)
from voxgram.format import GRID_SHAPE, N_FFT, NUM_SAMPLES, SAMPLE_RATE


def tone(frequency: float, seconds: float, rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(round(seconds * rate)) / rate
    return (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def test_fit_duration_pads_with_trailing_zeros():
    fitted = fit_duration(tone(440, 1.0))
    assert fitted.shape == (NUM_SAMPLES,)
    assert np.all(fitted[SAMPLE_RATE:] == 0)
    assert np.array_equal(fitted[:SAMPLE_RATE], tone(440, 1.0))


def test_fit_duration_rejects_long_recordings_unless_truncating():
    long = tone(440, 3.5)
    with pytest.raises(ValueError, match="longer"):
        fit_duration(long)
    truncated = fit_duration(long, truncate=True)
    assert np.array_equal(truncated, long[:NUM_SAMPLES])


def test_fit_duration_rejects_empty_and_multichannel():
    with pytest.raises(ValueError, match="empty"):
        fit_duration(np.zeros(0))
    with pytest.raises(ValueError, match="mono"):
        fit_duration(np.zeros((100, 2)))


def test_stft_magnitude_shape_and_peak_bin():
    frequency = 1000.0
    magnitude = stft_magnitude(fit_duration(tone(frequency, 3.0)))
    assert magnitude.shape == GRID_SHAPE
    expected_bin = round(frequency * N_FFT / SAMPLE_RATE)  # 32
    assert int(magnitude[:, GRID_SHAPE[1] // 2].argmax()) == expected_bin


def test_stft_magnitude_requires_exact_length():
    with pytest.raises(ValueError, match="shape"):
        stft_magnitude(np.zeros(1000, dtype=np.float32))


def test_griffin_lim_is_deterministic_and_matches_the_magnitude():
    magnitude = stft_magnitude(fit_duration(tone(1000.0, 3.0)))
    first = griffin_lim(magnitude, n_iter=32)
    second = griffin_lim(magnitude, n_iter=32)
    assert first.shape == (NUM_SAMPLES,)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)

    rebuilt = stft_magnitude(first)
    relative_error = np.linalg.norm(rebuilt - magnitude) / np.linalg.norm(magnitude)
    assert relative_error < 0.15  # measured 0.10 at 32 iterations, 0.04 at 100


def test_griffin_lim_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        griffin_lim(np.ones((10, 10)))


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "sub" / "tone.wav"
    save_audio(path, tone(440, 0.5))
    loaded = load_audio(path)
    info = sf.info(str(path))
    assert info.samplerate == SAMPLE_RATE
    assert info.channels == 1
    assert loaded.shape == (SAMPLE_RATE // 2,)
    assert np.max(np.abs(loaded)) == pytest.approx(OUTPUT_PEAK, abs=1e-3)


def test_load_resamples_and_mixes_to_mono(tmp_path):
    rate = 44_100
    left = tone(1000.0, 1.0, rate)
    stereo = np.stack([left, left], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(str(path), stereo, rate)

    loaded = load_audio(path)
    assert loaded.ndim == 1
    assert abs(loaded.size - SAMPLE_RATE) <= 1

    magnitude = stft_magnitude(fit_duration(loaded))
    assert int(magnitude[:, 20].argmax()) == round(1000.0 * N_FFT / SAMPLE_RATE)
