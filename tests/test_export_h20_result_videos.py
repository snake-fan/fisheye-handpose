from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from fisheye_handpose.trace import RunArtifactWriter, RunStatus, TraceStage, TraceStatus
from scripts import export_h20_result_videos as exporter


def _write_run(root: Path, *, item_id: str = "capture-001") -> tuple[Path, bytes]:
    run_dir = root / item_id / "run-001"
    blob_bytes = b"synthetic-h264-container"
    writer = RunArtifactWriter.create(
        run_dir,
        run_id="run-001",
        pipeline_version="test",
        metadata={"item_id": item_id},
    )
    blob = writer.put_blob(
        blob_bytes,
        role=exporter.VIDEO_ROLE,
        media_type="video/mp4",
        suffix=".mp4",
    )
    writer.append(
        record_id="video-export",
        stage=TraceStage.EXPORT,
        status=TraceStatus.SUCCEEDED,
        event=exporter.EXPORT_EVENT,
        payload={
            "output_status": "PRODUCED",
            "codec": "h264",
            "container": "mp4",
            "pixel_format": "yuv420p",
            "width": 320,
            "height": 240,
            "frame_count": 3,
            "frame_rate": {"numerator": 30, "denominator": 1},
            "time_base": {"numerator": 1, "denominator": 30},
        },
        blobs=(blob,),
    )
    writer.finalize(
        status=RunStatus.COMPLETED,
        summary={"item_id": item_id, "output_status": "PRODUCED"},
    )
    return run_dir, blob_bytes


def _video_info() -> dict[str, object]:
    return {
        "codec": "h264",
        "pixel_format": "yuv420p",
        "width": 320,
        "height": 240,
        "frame_rate": "30/1",
        "time_base": "1/30",
        "frame_count": 3,
        "duration_seconds": 0.1,
        "container": "mov,mp4,m4a,3gp,3g2,mj2",
    }


def _ffprobe_payload() -> dict[str, Any]:
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "width": 320,
                "height": 240,
                "avg_frame_rate": "30/1",
                "r_frame_rate": "30/1",
                "time_base": "1/30",
                "nb_frames": "3",
            }
        ],
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "0.100000",
        },
    }


def _mock_ffprobe(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
) -> list[str]:
    captured_command: list[str] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        captured_command.extend(command)
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(exporter.subprocess, "run", run)
    return captured_command


def test_validated_video_record_rejects_a_tampered_hash_chain(tmp_path: Path) -> None:
    run_dir, _ = _write_run(tmp_path / "runs")
    trace = run_dir / "trace.jsonl"
    value = json.loads(trace.read_text(encoding="utf-8"))
    value["payload"]["width"] = 999
    trace.write_text(json.dumps(value) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="canonical run validation failed"):
        exporter.validated_video_record(run_dir)


def test_resolve_blob_rejects_a_symlink_that_escapes_the_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    (run_dir / "escape.mp4").symlink_to(outside)

    with pytest.raises(ValueError, match="escapes its run directory"):
        exporter.resolve_blob(run_dir, "escape.mp4")


def test_discover_run_dirs_rejects_a_symlinked_run_that_escapes_root(
    tmp_path: Path,
) -> None:
    outside_run, _ = _write_run(tmp_path / "outside")
    runs_root = tmp_path / "runs"
    item_dir = runs_root / "capture-001"
    item_dir.mkdir(parents=True)
    (item_dir / "run-001").symlink_to(outside_run, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked run directory"):
        exporter.discover_run_dirs(runs_root)


def test_discover_run_dirs_rejects_a_symlinked_item_that_escapes_root(
    tmp_path: Path,
) -> None:
    outside_run, _ = _write_run(tmp_path / "outside")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    (runs_root / "capture-001").symlink_to(outside_run.parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked item directory"):
        exporter.discover_run_dirs(runs_root)


def test_discover_run_dirs_rejects_symlinked_canonical_metadata(tmp_path: Path) -> None:
    run_dir, _ = _write_run(tmp_path / "runs")
    manifest = run_dir / "run_manifest.json"
    outside_manifest = tmp_path / "outside-manifest.json"
    manifest.replace(outside_manifest)
    manifest.symlink_to(outside_manifest)

    with pytest.raises(ValueError, match="symlinked canonical metadata"):
        exporter.discover_run_dirs(tmp_path / "runs")


def test_probe_video_inspects_every_stream_and_returns_exact_rationals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _mock_ffprobe(monkeypatch, _ffprobe_payload())

    assert exporter.probe_video(tmp_path / "video.mp4", "ffprobe") == _video_info()
    assert "-select_streams" not in command


def test_probe_video_rejects_an_additional_audio_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _ffprobe_payload()
    payload["streams"].append(
        {
            "codec_type": "audio",
            "codec_name": "aac",
            "sample_rate": "48000",
        }
    )
    _mock_ffprobe(monkeypatch, payload)

    with pytest.raises(ValueError, match="exactly one total media stream"):
        exporter.probe_video(tmp_path / "video.mp4", "ffprobe")


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("streams", "width", 0, "invalid width"),
        ("streams", "height", -1, "invalid height"),
        ("streams", "nb_frames", "0", "invalid frame_count"),
        ("format", "duration", "nan", "invalid duration"),
    ],
)
def test_probe_video_rejects_invalid_dimensions_frame_count_or_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    payload = _ffprobe_payload()
    if section == "streams":
        payload["streams"][0][key] = value
    else:
        payload["format"][key] = value
    _mock_ffprobe(monkeypatch, payload)

    with pytest.raises(ValueError, match=message):
        exporter.probe_video(tmp_path / "video.mp4", "ffprobe")


@pytest.mark.parametrize(
    ("key", "reported", "message"),
    [
        ("frame_rate", "30000/1001", "mismatch for frame_rate"),
        ("time_base", "1/90000", "mismatch for time_base"),
    ],
)
def test_verify_contract_rejects_rate_or_time_base_mismatch(
    tmp_path: Path,
    key: str,
    reported: str,
    message: str,
) -> None:
    run_dir, _ = _write_run(tmp_path / "runs")
    _, record, _ = exporter.validated_video_record(run_dir)
    video = _video_info()
    video[key] = reported

    with pytest.raises(ValueError, match=message):
        exporter.verify_contract(record, video, run_dir)


def test_export_one_verifies_and_materializes_the_referenced_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, expected_bytes = _write_run(tmp_path / "runs")
    output = tmp_path / "partial"
    output.mkdir()
    monkeypatch.setattr(exporter, "probe_video", lambda _path, _ffprobe: _video_info())

    entry = exporter.export_one(run_dir, output, "ffprobe")

    assert (output / "capture-001.mp4").read_bytes() == expected_bytes
    assert entry["item_id"] == "capture-001"
    assert entry["run_id"] == "run-001"
    assert entry["materialization"] == "copy"
    assert entry["video"] == _video_info()
    exported = output / "capture-001.mp4"
    exported.write_bytes(b"consumer mutation")
    source_blob = next((run_dir / "blobs").rglob("*.mp4"))
    assert source_blob.read_bytes() == expected_bytes


def test_main_publishes_the_directory_only_after_every_run_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "runs"
    _write_run(runs_root)
    output = tmp_path / "published"
    monkeypatch.setattr(exporter, "probe_video", lambda _path, _ffprobe: _video_info())
    monkeypatch.setattr(exporter.shutil, "which", lambda _value: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        exporter,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {"runs_root": runs_root, "output_dir": output, "ffprobe": "ffprobe"},
        )(),
    )

    assert exporter.main() == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == exporter.MP4_EXPORT_SCHEMA
    assert manifest["video_count"] == 1
    assert (output / "capture-001.mp4").is_file()
    assert not list(tmp_path.glob(".published.*.partial"))
