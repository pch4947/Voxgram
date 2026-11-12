"""Tests for the encoder, image I/O and the digital baseline."""

import numpy as np
import pytest
import soundfile as sf

from voxgram import baseline, encoder
from voxgram.format import GRID_SHAPE, HOP_LENGTH, N_FFT, NUM_FREQ_BINS, SAMPLE_RATE
from voxgram.imageio import read_image, write_image


def tone(frequency: float, seconds: float) -> np.ndarray:
    t = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def voiced(seconds: float = 2.0) -> np.ndarray:
    """A harmonic signal with a gliding pitch, loosely like a vowel."""
    t = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    pitch = 120.0 + 60.0 * t / seconds
    phase = 2 * np.pi * np.cumsum(pitch) / SAMPLE_RATE
    signal = sum(np.sin(k * phase) / k for k in range(1, 20))
    return (0.2 * signal * np.hanning(t.size)).astype(np.float32)


def test_encoded_image_layout():
    frequency = 1000.0
    image = encoder.encode_samples(tone(frequency, 1.0))
    assert image.shape == GRID_SHAPE
    assert image.dtype == np.uint8

    # The tone is a dark horizontal line, counted from the bottom of the image.
    tone_bin = round(frequency * N_FFT / SAMPLE_RATE)
    tone_row = NUM_FREQ_BINS - 1 - tone_bin
    sounding = image[:, 10 : SAMPLE_RATE // HOP_LENGTH - 10]
    assert np.all(sounding.argmin(axis=0) == tone_row)
    assert sounding[tone_row].max() <= 2

    # The padded two seconds on the right are silent, so they are white.
    assert np.all(image[:, SAMPLE_RATE // HOP_LENGTH + 5 :] == 255)


def test_silent_recording_is_rejected():
    with pytest.raises(ValueError, match="silent"):
        encoder.encode_samples(np.zeros(SAMPLE_RATE, dtype=np.float32))


def test_long_recording_needs_truncate():
    with pytest.raises(ValueError, match="longer"):
        encoder.encode_samples(tone(440.0, 4.0))
    assert encoder.encode_samples(tone(440.0, 4.0), truncate=True).shape == GRID_SHAPE


def test_image_round_trip_on_a_non_ascii_path(tmp_path):
    image = np.arange(GRID_SHAPE[0] * GRID_SHAPE[1], dtype=np.uint32).reshape(GRID_SHAPE)
    image = (image % 256).astype(np.uint8)
    path = tmp_path / "카드" / "스펙트로그램.png"
    write_image(path, image)
    assert np.array_equal(read_image(path), image)


def test_read_image_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "missing.png")
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="decode"):
        read_image(broken)


def test_encoder_cli_writes_a_png(tmp_path, capsys):
    wav = tmp_path / "voice.wav"
    sf.write(str(wav), tone(440.0, 1.0), SAMPLE_RATE)
    assert encoder.main([str(wav)]) == 0
    assert "wrote" in capsys.readouterr().out
    assert read_image(tmp_path / "voice.png").shape == GRID_SHAPE


def test_baseline_round_trip(tmp_path):
    report = baseline.run_baseline(voiced(), tmp_path)

    for name in (
        baseline.IMAGE_NAME,
        baseline.ORIGINAL_NAME,
        baseline.GRIFFIN_LIM_NAME,
        baseline.BASELINE_NAME,
    ):
        assert (tmp_path / name).is_file(), name

    assert report.image_exact
    assert report.quantization_error_db <= 60.0 / 255.0 / 2 + 1e-6
    assert 0.0 <= report.floor_fraction < 1.0
    assert report.convergence_griffinlim < 0.2
    assert report.convergence_baseline < 0.2
    # The image path must not add noise where the original is quiet.
    assert report.quiet_level_baseline_db < report.quiet_level_griffinlim_db + 3.0
    assert sf.info(str(tmp_path / baseline.BASELINE_NAME)).frames == 48_000


def test_baseline_cli(tmp_path, capsys):
    wav = tmp_path / "voice.wav"
    sf.write(str(wav), voiced(1.0), SAMPLE_RATE)
    out = tmp_path / "out"
    assert baseline.main([str(wav), "-o", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "spectral convergence" in printed
    assert "quiet-frame level" in printed
    assert (out / baseline.BASELINE_NAME).is_file()
