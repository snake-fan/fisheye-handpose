from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from fisheye_handpose.cli import build_parser, main

SIZE = (32, 24)
FRAME_TIMESTAMPS_US = (1_000_000, 1_033_333, 1_066_666, 1_099_999)

CALIBRATION_YAML = """
calibration_info:
  reference_camera: cam_0
cameras:
  - id: cam_0
    name: IR_L
    distortion_model: KB
    image_width: 32
    image_height: 24
    intrinsics: {fx: 20, fy: 20, cx: 16, cy: 12}
    distortion: {k1: 0, k2: 0, k3: 0, k4: 0, k5: 0, k6: 0, p1: 0, p2: 0}
    extrinsics:
      rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      translation: [0, 0, 0]
  - id: cam_1
    name: IR_R
    distortion_model: KB
    image_width: 32
    image_height: 24
    intrinsics: {fx: 20, fy: 20, cx: 16, cy: 12}
    distortion: {k1: 0, k2: 0, k3: 0, k4: 0, k5: 0, k6: 0, p1: 0, p2: 0}
    extrinsics:
      rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      translation: [-120, 0, 0]
"""


def _write_video(path: Path, levels: tuple[int, ...] = (20, 60, 100, 140)) -> Path:
    width, height = SIZE
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for index, level in enumerate(levels):
            image = np.full((height, width, 3), level, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = index
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def _write_pts(path: Path, values_us: tuple[int, ...] = FRAME_TIMESTAMPS_US) -> Path:
    path.write_text(
        "timestamp_us\n" + "".join(f"{value}\n" for value in values_us),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def tiny_session(tmp_path: Path) -> Path:
    prefix = "capture"
    (tmp_path / f"{prefix}_calibration_camera.yaml").write_text(CALIBRATION_YAML, encoding="utf-8")
    for side, delta in (("left", 0), ("right", 5)):
        _write_video(tmp_path / f"{prefix}_camera_{side}_part0001.mp4")
        _write_pts(
            tmp_path / f"{prefix}_camera_{side}_part0001_pts.csv",
            tuple(value + delta for value in FRAME_TIMESTAMPS_US),
        )
    return tmp_path


def _audit_args(session: Path, output: Path) -> list[str]:
    return [
        "audit-session",
        str(session),
        "--left-id",
        "cam_0",
        "--right-id",
        "cam_1",
        "--translation-unit",
        "mm",
        "--extrinsics-convention",
        "reference_to_camera",
        "--max-skew-us",
        "1000",
        "--min-video-bytes",
        "1",
        "--min-overlap-match-rate",
        "0.99",
        "--output-width",
        str(SIZE[0]),
        "--output-height",
        str(SIZE[1]),
        "--balance",
        "0",
        "--min-common-valid-fraction",
        "0",
        "--skip-epipolar-qa",
        "--allow-short-session",
        "--output",
        str(output),
    ]


def _walk(value: Any) -> Iterator[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _video_reports(report: dict[str, Any]) -> list[dict[str, Any]]:
    required = {"codec_name", "decoded_frame_count", "expected_frame_count", "hard_failures"}
    return [
        value for value in _walk(report) if isinstance(value, dict) and required <= value.keys()
    ]


def test_console_main_schema_emits_machine_readable_contract(capsys):
    assert main(["schema"]) == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["version"] == "fhp21/v1"
    assert payload["landmark_count"] == 21
    assert captured.err == ""


def test_pair_pts_reports_clock_offset_and_raw_corrected_provenance(tmp_path, capsys):
    left = _write_pts(tmp_path / "left.csv")
    right = _write_pts(
        tmp_path / "right.csv",
        tuple(value + 200 for value in FRAME_TIMESTAMPS_US),
    )
    pairs_csv = tmp_path / "pairs.csv"

    code = main(
        [
            "pair-pts",
            str(left),
            str(right),
            "--max-skew-us",
            "100",
            "--clock-offset-us",
            "-200",
            "--output",
            str(pairs_csv),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sync"]["clock_offset_ns"] == -200_000
    with pairs_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(FRAME_TIMESTAMPS_US)
    assert int(rows[0]["right_timestamp_raw_ns"]) == 1_000_200_000
    assert int(rows[0]["right_timestamp_corrected_ns"]) == 1_000_000_000
    assert int(rows[0]["corrected_skew_ns"]) == 0


def test_audit_session_fully_decodes_both_videos_and_writes_report(tiny_session, tmp_path):
    output = tmp_path / "audit.json"

    assert main(_audit_args(tiny_session, output)) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] in {"PASS", "WARN"}
    assert report["hard_failures"] == []
    assert "stages" in report
    videos = _video_reports(report)
    assert len(videos) == 2
    assert all(video["status"] == "PASS" for video in videos)
    assert all(video["decoded_frame_count"] == len(FRAME_TIMESTAMPS_US) for video in videos)
    assert all(video["decoded_frame_count"] == video["expected_frame_count"] for video in videos)


@pytest.mark.parametrize("fault", ["empty_pts", "malformed_pts", "bad_video"])
def test_audit_failure_still_writes_structured_fail_report(tiny_session, tmp_path, fault, capsys):
    if fault == "empty_pts":
        (tiny_session / "capture_camera_left_part0001_pts.csv").write_text(
            "timestamp_us\n", encoding="utf-8"
        )
    elif fault == "malformed_pts":
        (tiny_session / "capture_camera_left_part0001_pts.csv").write_text(
            "timestamp_us\nnot-an-integer\n", encoding="utf-8"
        )
    else:
        (tiny_session / "capture_camera_left_part0001.mp4").write_bytes(b"not a video")
    output = tmp_path / f"audit-{fault}.json"

    assert main(_audit_args(tiny_session, output)) == 2

    assert output.is_file()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert "stages" in report
    assert report["hard_failures"]
    serialized = json.dumps(report).lower()
    if fault.endswith("pts"):
        assert "timestamp" in serialized
    else:
        assert "video" in serialized and ("decode" in serialized or "codec" in serialized)
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--max-skew-us", "0"),
        ("--min-video-bytes", "-1"),
        ("--min-overlap-match-rate", "-0.01"),
        ("--min-overlap-match-rate", "1.01"),
        ("--output-width", "0"),
        ("--output-height", "0"),
        ("--balance", "-0.01"),
        ("--balance", "1.01"),
        ("--balance", "nan"),
        ("--fov-scale", "0"),
        ("--min-common-valid-fraction", "-0.01"),
        ("--min-common-valid-fraction", "1.01"),
    ],
)
def test_audit_gate_ranges_are_rejected_during_argument_parsing(option, value):
    arguments = [
        "audit-session",
        "unused-session",
        "--left-id",
        "cam_0",
        "--right-id",
        "cam_1",
        "--translation-unit",
        "mm",
        "--extrinsics-convention",
        "reference_to_camera",
    ]
    if option != "--max-skew-us":
        arguments.extend(("--max-skew-us", "1000"))
    arguments.extend((option, value))

    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args(arguments)
    assert caught.value.code == 2
