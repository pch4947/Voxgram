"""Tests for the evaluation scripts and metrics."""

import csv

import numpy as np
import pytest
import soundfile as sf

from voxgram import card, encoder, evaluation, synthetic
from voxgram.decoder import DecodeError
from voxgram.detection import MarkerDetectionError
from voxgram.evaluation import (
    VARIANTS,
    Capture,
    SyntheticCondition,
    Variant,
    append_to_captures,
    capture_condition,
    cell_error_mm,
    detection_outcome,
    encoded_from_image,
    evaluate,
    gray_error,
    load_encoded,
    marker_error_mm,
    real_captures,
    spectral_error_db,
    summary_markdown,
    write_results_csv,
)
from voxgram.format import GRID_SHAPE, MARKER_CENTERS_MM, NUM_SAMPLES, SAMPLE_RATE
from voxgram.imageio import write_image

ITERATIONS = 2  # Griffin-Lim quality is not under test here.
SMALL_SHAPE = (900, 1200)
SMALL_SCALE = 6.5


def voiced_signal() -> np.ndarray:
    t = np.arange(NUM_SAMPLES) / SAMPLE_RATE
    pitch = 120.0 + 60.0 * t / 3.0
    phase = 2 * np.pi * np.cumsum(pitch) / SAMPLE_RATE
    signal = sum(np.sin(k * phase) / k for k in range(1, 20))
    return (0.2 * signal * np.hanning(t.size)).astype(np.float32)


@pytest.fixture(scope="module")
def encoded() -> np.ndarray:
    return encoder.encode_samples(voiced_signal())


# --- variants -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "stages", "notes"),
    [
        ("full", "all", ""),
        ("no-calibrate", "calibrate", ""),
        ("no-rectify", "rectify+calibrate", ""),
        ("no-detect", "detect+rectify+calibrate", ""),
        ("global-threshold", "all", "--threshold global"),
        ("smooth-3", "all", "--smooth 3"),
        ("samples-3", "all", "--samples 3"),
    ],
)
def test_variants_are_named_as_in_the_capture_table(name, stages, notes):
    variant = VARIANTS[name]
    assert variant.stages == stages
    assert variant.notes == notes
    variant.options(ITERATIONS)  # valid decoder options


# --- metrics ------------------------------------------------------------------


def test_gray_error_is_zero_for_a_perfect_recovery(encoded):
    assert gray_error(encoded.astype(float), encoded) == (0.0, 0.0)


def test_gray_error_in_db_follows_the_encoding():
    encoded = np.full(GRID_SHAPE, 100, dtype=np.uint8)
    recovered = np.full(GRID_SHAPE, 102.55)
    mae_gray, mae_db = gray_error(recovered, encoded)
    assert mae_gray == pytest.approx(2.55)
    assert mae_db == pytest.approx(0.6)  # 60 dB over 255 gray levels


def test_gray_error_rejects_a_wrong_shape(encoded):
    with pytest.raises(ValueError):
        gray_error(np.zeros((10, 10)), encoded)


def test_spectral_error_is_small_for_the_original_and_large_for_silence(encoded):
    assert spectral_error_db(voiced_signal(), encoded) < 1.0
    assert spectral_error_db(np.zeros(NUM_SAMPLES), encoded) == pytest.approx(60.0)


def test_geometric_errors_are_zero_for_the_true_homography():
    true = synthetic.make_homography(scale=10, center=(800, 600), perspective=(0.001, 0))
    assert marker_error_mm(true, true) == pytest.approx(0.0, abs=1e-9)
    assert cell_error_mm(true, true) == pytest.approx(0.0, abs=1e-9)


def test_geometric_errors_measure_a_shift_in_card_millimetres():
    true = synthetic.make_homography(scale=10, center=(800, 600), rotation_deg=30)
    shift = np.array([[1.0, 0.0, 0.3], [0.0, 1.0, -0.4], [0.0, 0.0, 1.0]])
    estimated = true @ shift  # samples 0.5 mm away from every intended point
    assert marker_error_mm(estimated, true) == pytest.approx(0.5)
    assert cell_error_mm(estimated, true) == pytest.approx(0.5)


def _decode_error(message: str, cause: Exception | None = None) -> DecodeError:
    error = DecodeError(message)
    error.__cause__ = cause
    return error


def test_detection_outcome():
    full = VARIANTS["full"]
    assert detection_outcome(None, full) == "yes"
    assert detection_outcome(None, VARIANTS["no-detect"]) == ""
    missing = MarkerDetectionError("found 3 marker candidate(s), need 4")
    assert detection_outcome(_decode_error(f"detect: {missing}", missing), full) == "no"
    twice = MarkerDetectionError("2 different sets of four candidates; refusing to guess")
    assert detection_outcome(_decode_error(f"detect: {twice}", twice), full) == "ambiguous"
    assert detection_outcome(_decode_error("homography: collinear"), full) == "no"
    # Detection worked; a later stage failed.
    assert detection_outcome(_decode_error("extract: 40% unmeasurable"), full) == "yes"


# --- inputs -------------------------------------------------------------------


def test_encoded_values_are_read_back_from_a_card_print(encoded):
    assert np.array_equal(encoded_from_image(card.render_card(encoded, 300)), encoded)
    assert np.array_equal(encoded_from_image(encoded), encoded)


def test_encoded_values_reject_an_image_of_another_shape():
    with pytest.raises(ValueError, match="neither"):
        encoded_from_image(np.zeros((500, 500), dtype=np.uint8))


def test_encoded_values_fall_back_to_the_recording(tmp_path, encoded):
    sf.write(tmp_path / "c07_source.wav", voiced_signal(), SAMPLE_RATE, subtype="FLOAT")
    assert np.array_equal(load_encoded("c07", tmp_path), encoded)
    with pytest.raises(FileNotFoundError):
        load_encoded("c08", tmp_path)


def test_capture_condition_comes_from_the_capture_id():
    assert capture_condition("20261117_c01_p1_t30p_l0_02") == "t30p_l0"
    assert capture_condition("odd-name") == "unknown"


# --- running ------------------------------------------------------------------


def _photo(encoded, condition: SyntheticCondition, *, cover_marker: bool = False):
    photo, homography = evaluation.render_synthetic(
        encoded, condition, shape=SMALL_SHAPE, scale=SMALL_SCALE
    )
    if cover_marker:
        x, y = synthetic.project_points(homography, np.array(MARKER_CENTERS_MM))[0]
        photo[int(y) - 40 : int(y) + 40, int(x) - 40 : int(x) + 40] = 255
    return photo, homography


@pytest.fixture(scope="module")
def results(encoded, tmp_path_factory):
    folder = tmp_path_factory.mktemp("eval")
    captures = []
    conditions = {
        "front": SyntheticCondition("front", noise=0.0),
        "gradient": SyntheticCondition("gradient", gradient=0.5, gamma=1.3),
        "covered": SyntheticCondition("front"),
    }
    for name, condition in conditions.items():
        photo, homography = _photo(encoded, condition, cover_marker=name == "covered")
        path = folder / f"{name}.png"
        write_image(path, photo)
        captures.append(Capture(name, name, "voiced", path, encoded, homography))
    variants = [VARIANTS["full"], VARIANTS["no-calibrate"], VARIANTS["no-detect"]]
    return evaluate(
        captures, variants, iterations=ITERATIONS, audio_dir=folder / "audio", progress=False
    ), folder


def _run(results, capture_id, variant):
    return next(
        r for r in results if r.capture.capture_id == capture_id and r.variant.name == variant
    )


def test_full_pipeline_recovers_the_encoded_values(results):
    runs, _ = results
    front = _run(runs, "front", "full")
    assert front.detect_ok == "yes" and front.decoded
    assert front.mae_gray < 2.0  # about 2 photo pixels per cell at this small scale
    assert front.marker_error_mm < 0.05
    assert front.cell_error_mm < 0.05
    assert front.timings["calibrate"] > 0


def test_each_switched_off_stage_makes_the_result_worse(results):
    runs, _ = results
    full = _run(runs, "gradient", "full")
    no_calibrate = _run(runs, "gradient", "no-calibrate")
    no_detect = _run(runs, "gradient", "no-detect")
    assert full.mae_gray < no_calibrate.mae_gray < no_detect.mae_gray
    assert no_detect.detect_ok == ""
    assert np.isnan(no_detect.marker_error_mm)


def test_a_failed_detection_is_recorded_not_raised(results):
    runs, folder = results
    covered = _run(runs, "covered", "full")
    assert covered.detect_ok == "no"
    assert covered.error.startswith("detect:")
    assert np.isnan(covered.mae_gray)
    assert not (folder / "audio" / "covered" / "full.wav").exists()


def test_audio_is_written_for_the_runs_and_the_baseline(results):
    _, folder = results
    assert (folder / "audio" / "front" / "full.wav").is_file()
    assert (folder / "audio" / "baseline" / "voiced.wav").is_file()


def test_results_table_and_summary(results, tmp_path):
    runs, _ = results
    path = tmp_path / "results.csv"
    write_results_csv(runs, path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(runs)
    assert {row["stages"] for row in rows} == {"all", "calibrate", "detect+rectify+calibrate"}

    summary = summary_markdown(runs, source="synthetic", commit="abc1234")
    assert "## Stage-by-stage comparison" in summary
    assert "| digital baseline |" in summary
    assert "| covered | 1 | 0/1 |" in summary  # detection rate per condition
    assert "Marker error mm" in summary


# --- the capture table --------------------------------------------------------

HEADER = (
    "capture_id,date,session,card,format_version,phone,tilt_deg,tilt_axis,roll_deg,light,"
    "lux_min,lux_max,distance_cm,ev_comp,setup_ok,commit,stages,detect_ok,mae_gray,mae_db,"
    "intelligibility,notes"
)
CAPTURE = "20261121_c01_p1_t00_l0_01"
ROW = (
    f"{CAPTURE},2026-11-21,20261121_session-01,c01,0.3-pilot,p1,0,none,0,l0,,,30,0,yes,"
    ',,,,,,"card flat"'
)


@pytest.fixture
def data_dir(tmp_path, encoded):
    data = tmp_path / "data"
    (data / "photos" / "20261121").mkdir(parents=True)
    (data / "cards").mkdir()
    write_image(data / "cards" / "c01_source.png", card.render_card(encoded, 300))
    photo, _ = _photo(encoded, SyntheticCondition("front"))
    write_image(data / "photos" / "20261121" / f"{CAPTURE}.jpg", photo)
    table = tmp_path / "captures.csv"
    table.write_bytes(f"{HEADER}\r\n{ROW}\r\n".encode())
    return data, table


def test_real_captures_are_read_from_the_data_folder(data_dir, encoded, tmp_path):
    data, table = data_dir
    with open(table, "ab") as f:
        f.write(b"20261121_c02_p1_t00_l0_01,,,c02,,,,,,,,,,,no,,,,,,,\r\n")
    captures, problems = real_captures(table, data)
    assert [c.capture_id for c in captures] == [CAPTURE]
    assert captures[0].condition == "t00_l0"
    assert np.array_equal(captures[0].encoded, encoded)
    assert len(problems) == 1 and "c02" in problems[0]


def test_results_are_recorded_in_the_capture_table(data_dir):
    data, table = data_dir
    captures, _ = real_captures(table, data)
    variants = [VARIANTS["full"], VARIANTS["global-threshold"]]
    runs = evaluate(captures, variants, iterations=ITERATIONS, progress=False)

    assert append_to_captures(runs, table, "abc1234") == 2
    raw = table.read_bytes()
    assert raw.count(b"\r\n") == 3 and raw.count(b"\n") == 3  # line endings kept
    with open(table, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    first, second = rows
    # The unprocessed row receives the full pipeline.
    assert (first["commit"], first["stages"], first["detect_ok"]) == ("abc1234", "all", "yes")
    assert float(first["mae_gray"]) < 3.0
    assert first["notes"] == "card flat"
    # Other variants are copies, with their options in the notes.
    assert second["capture_id"] == CAPTURE and second["light"] == "l0"
    assert second["notes"] == "card flat; --threshold global"

    # Recording the same results again changes nothing.
    assert append_to_captures(runs, table, "abc1234") == 0
    assert len(list(csv.DictReader(open(table, newline="", encoding="utf-8")))) == 2


def test_command_line_on_the_capture_table(data_dir, tmp_path, capsys):
    data, table = data_dir
    out = tmp_path / "out"
    report = tmp_path / "experiments" / "report.md"
    argv = ["captures", "--captures", str(table), "--data", str(data), "-o", str(out)]
    argv += ["--variants", "full", "--iterations", "1", "--no-audio", "--report", str(report)]
    assert evaluation.main(argv) == 0
    assert (out / "results.csv").is_file()
    assert report.read_text(encoding="utf-8") == (out / "summary.md").read_text(encoding="utf-8")


def test_variant_options_reach_the_decoder():
    variant = Variant("x", skip=frozenset({"rectify"}), threshold="global", smoothing_size=5)
    options = variant.options(7)
    assert options.skip == {"rectify"}
    assert (options.threshold, options.smoothing_size, options.iterations) == ("global", 5, 7)
