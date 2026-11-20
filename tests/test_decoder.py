"""Tests for the decoder: photograph to WAV."""

import importlib.util

import numpy as np
import pytest
import soundfile as sf

from voxgram import card, decoder, encoder, synthetic
from voxgram.decoder import DecodeError, DecodeOptions, Stages, decode_file, decode_photo
from voxgram.format import GRID_SHAPE, MARKER_CENTERS_MM, NUM_SAMPLES, SAMPLE_RATE
from voxgram.imageio import read_image, write_image
from voxgram.intensity import gray_to_magnitude, image_to_grid

FAST = {"iterations": 2}  # Griffin-Lim quality is not under test here.
NO_MARKERS = frozenset({"detect"})
HAS_DETECTION = importlib.util.find_spec("voxgram.detection") is not None


def voiced_signal() -> np.ndarray:
    """A harmonic signal with a gliding pitch, loosely like a vowel."""
    t = np.arange(NUM_SAMPLES) / SAMPLE_RATE
    pitch = 120.0 + 60.0 * t / 3.0
    phase = 2 * np.pi * np.cumsum(pitch) / SAMPLE_RATE
    signal = sum(np.sin(k * phase) / k for k in range(1, 20))
    return (0.2 * signal * np.hanning(t.size)).astype(np.float32)


@pytest.fixture(scope="module")
def spectrogram() -> np.ndarray:
    """The encoded image, with real black cells and real white cells."""
    return encoder.encode_samples(voiced_signal())


@pytest.fixture(scope="module")
def undistorted_card(spectrogram) -> np.ndarray:
    return card.render_card(spectrogram, 300)


@pytest.fixture(scope="module")
def tilted(spectrogram):
    """A tilted, unevenly lit photograph, with the exact marker centres."""
    homography = synthetic.make_homography(
        scale=12.0, center=(1000, 750), rotation_deg=4, perspective=(0.0004, 0.0002)
    )
    shape = (1500, 2000)
    photo = card.render_view(spectrogram, homography, shape, supersample=2)
    gain = synthetic.illumination_gradient(shape, low=0.75, high=1.0, angle_deg=30)
    photo = synthetic.apply_tone_curve(photo * gain, 1.2)
    centres = synthetic.project_points(homography, np.array(MARKER_CENTERS_MM))
    return photo, centres


def exact_markers(centres: np.ndarray) -> Stages:
    """Stages whose detection returns the true marker centres."""
    return Stages(detect=lambda gray, method: centres)


# --- the undistorted card (milestone M3 check) ------------------------------


def test_undistorted_card_decodes_to_the_encoded_gray_values(spectrogram, undistorted_card):
    result = decode_photo(undistorted_card, DecodeOptions(skip=NO_MARKERS, **FAST))
    assert result.gray.shape == GRID_SHAPE
    assert np.allclose(result.gray, spectrogram, atol=1e-6)


def test_undistorted_card_magnitude_matches_the_digital_baseline(spectrogram, undistorted_card):
    result = decode_photo(undistorted_card, DecodeOptions(skip=NO_MARKERS, **FAST))
    expected = gray_to_magnitude(image_to_grid(spectrogram.astype(np.float64)))
    assert np.allclose(result.magnitude, expected)
    assert result.samples.shape == (NUM_SAMPLES,)


def test_skipping_detection_skips_the_geometry_and_the_calibration(undistorted_card):
    result = decode_photo(undistorted_card, DecodeOptions(skip=NO_MARKERS, **FAST))
    assert result.skipped == ["detect", "rectify", "calibrate"]
    assert result.homography is None


# --- a tilted photograph ----------------------------------------------------


def test_tilted_photograph_is_recovered_with_all_stages(spectrogram, tilted):
    photo, centres = tilted
    result = decode_photo(photo, DecodeOptions(**FAST), exact_markers(centres))
    assert result.skipped == []
    assert result.homography is not None
    assert np.mean(np.abs(result.gray - spectrogram)) < 2.0
    assert result.unmeasurable_fraction == 0.0


def test_each_correction_earns_its_place(spectrogram, tilted):
    photo, centres = tilted
    stages = exact_markers(centres)

    def error(skip):
        options = DecodeOptions(skip=frozenset(skip), **FAST)
        return np.mean(np.abs(decode_photo(photo, options, stages).gray - spectrogram))

    full = error(())
    no_calibration = error(("calibrate",))
    no_geometry = error(("rectify",))
    assert full < no_calibration / 5
    assert full < no_geometry / 5


def test_skipping_rectify_also_skips_the_calibration(tilted):
    photo, centres = tilted
    result = decode_photo(
        photo, DecodeOptions(skip=frozenset({"rectify"}), **FAST), exact_markers(centres)
    )
    assert result.skipped == ["rectify", "calibrate"]
    assert result.homography is not None  # detection and homography still ran


def test_skipping_calibrate_keeps_the_geometry(tilted):
    photo, centres = tilted
    result = decode_photo(
        photo, DecodeOptions(skip=frozenset({"calibrate"}), **FAST), exact_markers(centres)
    )
    assert result.skipped == ["calibrate"]
    assert [stage for stage, _ in result.timings if stage == "rectify"] == ["rectify"]


def test_black_cells_are_not_read_as_silence(spectrogram, tilted):
    # The loudest cells are photographed as exact black, which the calibration
    # flags as possibly clipped. They must decode as black, not as white.
    photo, centres = tilted
    assert (spectrogram == 0).any()
    result = decode_photo(photo, DecodeOptions(**FAST), exact_markers(centres))
    assert result.gray[spectrogram == 0].max() < 10.0


def test_stages_are_timed_in_pipeline_order(tilted):
    photo, centres = tilted
    result = decode_photo(photo, DecodeOptions(**FAST), exact_markers(centres))
    names = [stage for stage, _ in result.timings]
    assert names == [
        "grayscale",
        "detect",
        "homography",
        "rectify",
        "calibrate",
        "extract",
        "magnitude",
        "griffin-lim",
    ]
    assert all(seconds >= 0.0 for _, seconds in result.timings)
    assert set(names) <= set(decoder.STAGES)


def test_threshold_method_is_passed_to_detection(tilted):
    photo, centres = tilted
    seen = []

    def detect(gray, method):
        seen.append(method)
        return centres

    options = DecodeOptions(threshold="global", **FAST)
    decode_photo(photo, options, Stages(detect=detect))
    assert seen == ["global"]


def test_skipped_detection_is_never_called(undistorted_card):
    def detect(gray, method):
        raise AssertionError("detection must not run")

    decode_photo(undistorted_card, DecodeOptions(skip=NO_MARKERS, **FAST), Stages(detect=detect))


def test_smoothing_option_changes_the_recovered_grid(undistorted_card):
    plain = decode_photo(undistorted_card, DecodeOptions(skip=NO_MARKERS, **FAST))
    smoothed = decode_photo(
        undistorted_card, DecodeOptions(skip=NO_MARKERS, smoothing_size=3, **FAST)
    )
    assert not np.allclose(plain.gray, smoothed.gray)


# --- failures ---------------------------------------------------------------


def failing_detection(gray, method):
    raise ValueError("found 2 of the 4 markers")


def test_detection_failure_names_the_stage_and_produces_no_signal(tilted):
    photo, _ = tilted
    with pytest.raises(DecodeError, match=r"^detect: found 2 of the 4 markers"):
        decode_photo(photo, DecodeOptions(**FAST), Stages(detect=failing_detection))


def test_intermediates_survive_a_failure(tilted):
    photo, _ = tilted
    kept = {}
    with pytest.raises(DecodeError):
        decode_photo(
            photo, DecodeOptions(**FAST), Stages(detect=failing_detection), intermediates=kept
        )
    assert "grayscale" in kept
    assert np.array_equal(kept["grayscale"], photo)


def test_bad_marker_array_is_rejected(tilted):
    photo, _ = tilted
    stages = Stages(detect=lambda gray, method: np.zeros((3, 2)))
    with pytest.raises(DecodeError, match="detect"):
        decode_photo(photo, DecodeOptions(**FAST), stages)


def test_card_partly_out_of_frame_stops_instead_of_inventing_audio(spectrogram):
    homography = synthetic.make_homography(scale=12.0, center=(1500, 750))
    shape = (1500, 2000)  # the right part of the card falls outside the frame
    photo = card.render_view(spectrogram, homography, shape, supersample=1)
    centres = synthetic.project_points(homography, np.array(MARKER_CENTERS_MM))
    with pytest.raises(DecodeError) as excinfo:
        decode_photo(photo, DecodeOptions(**FAST), exact_markers(centres))
    assert str(excinfo.value).startswith(("calibrate:", "extract:", "rectify:"))


@pytest.fixture(scope="module")
def card_cut_by_the_frame(spectrogram):
    """The right part of the data region is outside the photograph."""
    homography = synthetic.make_homography(scale=12.0, center=(1500, 750))
    photo = card.render_view(spectrogram, homography, (1500, 2000), supersample=1)
    centres = synthetic.project_points(homography, np.array(MARKER_CENTERS_MM))
    return photo, centres


def test_too_many_unmeasurable_cells_stop_the_decoder(card_cut_by_the_frame):
    photo, centres = card_cut_by_the_frame
    options = DecodeOptions(skip=frozenset({"calibrate"}), max_unmeasurable=0.05, **FAST)
    with pytest.raises(DecodeError, match="could not be measured"):
        decode_photo(photo, options, exact_markers(centres))


def test_unmeasurable_cells_below_the_limit_are_read_as_silence(card_cut_by_the_frame, spectrogram):
    photo, centres = card_cut_by_the_frame
    options = DecodeOptions(skip=frozenset({"calibrate"}), max_unmeasurable=0.5, **FAST)
    result = decode_photo(photo, options, exact_markers(centres))
    assert 0.05 < result.unmeasurable_fraction < 0.5
    # The cells beyond the right edge of the frame are white, which is silence.
    assert np.all(result.gray[:, -10:] == 255.0)


def test_colour_photograph_without_the_detection_module_fails_clearly():
    if HAS_DETECTION:
        pytest.skip("voxgram.detection is present")
    colour = np.zeros((20, 30, 3), dtype=np.uint8)
    with pytest.raises(DecodeError, match="detection"):
        decode_photo(colour, DecodeOptions(skip=NO_MARKERS, **FAST))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"skip": frozenset({"griffin-lim"})},
        {"threshold": "adaptive"},
        {"card_dpi": 0.0},
        {"smoothing_size": 4},
        {"smoothing_size": 1},
        {"iterations": 0},
        {"max_unmeasurable": 1.5},
    ],
)
def test_invalid_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        DecodeOptions(**kwargs)


# --- files and the command line ---------------------------------------------


@pytest.fixture
def card_file(tmp_path, undistorted_card):
    path = tmp_path / "card.png"
    write_image(path, undistorted_card)
    return path


def test_decode_file_writes_a_wav_of_three_seconds(card_file, tmp_path):
    output = tmp_path / "out" / "voice.wav"
    result = decode_file(card_file, output, DecodeOptions(skip=NO_MARKERS, **FAST))
    info = sf.info(str(output))
    assert info.samplerate == SAMPLE_RATE
    assert info.frames == NUM_SAMPLES
    assert [stage for stage, _ in result.timings][0] == "read"
    assert [stage for stage, _ in result.timings][-1] == "write"


def test_command_line_decodes_and_saves_the_intermediates(card_file, tmp_path, spectrogram, capsys):
    output = tmp_path / "voice.wav"
    folder = tmp_path / "steps"
    code = decoder.main(
        [
            str(card_file),
            "-o",
            str(output),
            "--skip",
            "detect",
            "--iterations",
            "2",
            "--save-intermediates",
            str(folder),
            "-q",
        ]
    )
    assert code == 0
    assert output.is_file()
    assert (folder / "1_grayscale.png").is_file()
    recovered = read_image(folder / "5_spectrogram.png")
    assert np.array_equal(recovered, spectrogram)
    assert f"wrote {output}" in capsys.readouterr().out


def test_command_line_logs_the_time_of_each_stage(card_file, tmp_path, capsys):
    decoder.main(
        [str(card_file), "-o", str(tmp_path / "v.wav"), "--skip", "detect", "--iterations", "2"]
    )
    log_text = capsys.readouterr().err
    for stage in ("grayscale", "extract", "magnitude", "griffin-lim", "write"):
        assert stage in log_text
    assert "skipped" in log_text


def test_command_line_failure_exits_non_zero_and_writes_no_audio(tmp_path, capsys):
    blank = tmp_path / "blank.png"
    write_image(blank, np.full((300, 450), 200, dtype=np.uint8))  # no markers at all
    output = tmp_path / "voice.wav"
    folder = tmp_path / "steps"
    with pytest.raises(SystemExit) as excinfo:
        decoder.main([str(blank), "-o", str(output), "--save-intermediates", str(folder), "-q"])
    assert excinfo.value.code == 1
    assert not output.exists()
    error = capsys.readouterr().err
    assert "error: detect:" in error
    assert "no audio was written" in error
    assert (folder / "1_grayscale.png").is_file()  # kept to diagnose the failure


def test_command_line_missing_photograph_fails_cleanly(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        decoder.main([str(tmp_path / "absent.jpg"), "-o", str(tmp_path / "v.wav"), "-q"])
    assert excinfo.value.code == 1
    assert "not found" in capsys.readouterr().err
    assert not (tmp_path / "v.wav").exists()


def test_command_line_rejects_an_invalid_option(card_file, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        decoder.main([str(card_file), "-o", str(tmp_path / "v.wav"), "--smooth", "4"])
    assert excinfo.value.code == 2


def test_intermediate_images_show_unmeasurable_pixels_as_mid_gray(tmp_path):
    image = np.array([[0.0, 255.0], [np.nan, 100.4]])
    (path,) = decoder.save_intermediates(tmp_path, {"corrected": image})
    saved = read_image(path)
    assert saved.tolist() == [[0, 255], [decoder.INVALID_PIXEL, 100]]


def test_binary_image_is_saved_as_black_and_white(tmp_path):
    binary = np.array([[True, False], [False, True]])
    (path,) = decoder.save_intermediates(tmp_path, {"binary": binary})
    assert read_image(path).tolist() == [[255, 0], [0, 255]]


# --- with the real marker detection (runs once voxgram.detection is merged) ---


@pytest.mark.skipif(not HAS_DETECTION, reason="voxgram.detection is not merged yet")
def test_real_detection_decodes_a_tilted_colour_photograph(spectrogram, tilted):
    photo, _ = tilted
    colour = synthetic.as_bgr(photo)  # what imageio.read_image gives for a JPEG
    result = decode_photo(colour, DecodeOptions(**FAST))
    assert np.mean(np.abs(result.gray - spectrogram)) < 3.0
    assert result.skipped == []


@pytest.mark.skipif(not HAS_DETECTION, reason="voxgram.detection is not merged yet")
@pytest.mark.parametrize("threshold", ["local", "global"])
def test_real_detection_works_with_both_thresholds(spectrogram, tilted, threshold):
    photo, _ = tilted
    result = decode_photo(photo, DecodeOptions(threshold=threshold, **FAST))
    assert np.mean(np.abs(result.gray - spectrogram)) < 5.0


@pytest.mark.skipif(not HAS_DETECTION, reason="voxgram.detection is not merged yet")
def test_real_detection_binary_image_is_saved_with_the_intermediates(tilted, tmp_path):
    photo, _ = tilted
    kept = {}
    decode_photo(photo, DecodeOptions(keep_binary=True, **FAST), intermediates=kept)
    assert kept["binary"].dtype == np.bool_
    assert kept["binary"].shape == photo.shape
