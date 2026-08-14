from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fisheye_handpose.trace import (
    RunArtifactReader,
    RunArtifactWriter,
    RunStatus,
    TraceStage,
    TraceStatus,
)


def _raw_points(length_m: float) -> tuple[list[list[float] | None], list[str]]:
    points: list[list[float] | None] = [[0.0, 0.0, 1.0], [length_m, 0.0, 1.0]]
    points.extend([None] * 19)
    return points, ["VALID", "VALID", *(["LOW_KEYPOINT_SCORE"] * 19)]


def _build_canonical_run(root: Path) -> Path:
    run_dir = root / "canonical-run"
    writer = RunArtifactWriter.create(
        run_dir,
        run_id="baseline-fixture",
        pipeline_version="fixture-revision",
        config={"audit": {"max_skew_ns": 250_000}},
        inputs=[{"kind": "stereo_session", "path": "/capture/session-a"}],
        metadata={"item_id": "session-a"},
    )
    worker_manifest = {
        "schema_version": "fisheye-handpose/h20-worker-manifest/v1",
        "request": {"sha256": "a" * 64},
        "configuration": {
            "thresholds": {"bbox_score": 0.3, "keypoint_score": 0.2},
            "tracking": {"method": "sequence_root_distance_one_to_one_v1"},
            "mano": {"iterations": 200, "max_fit_rmse_m": 0.02},
            "temporal": {"method": "causal_time_ema_v1"},
        },
        "models": {"detector": {"id": "rtmdet-nano-hand"}},
        "calibration": {"calibration_id": "cal-fixture"},
    }
    manifest_blob = writer.put_blob(
        (json.dumps(worker_manifest, sort_keys=True) + "\n").encode(),
        role="worker_manifest",
        media_type="application/json",
        suffix=".json",
    )
    previous: str | None = None

    def append(
        record_id: str,
        stage: TraceStage,
        event: str,
        payload: dict[str, Any],
        *,
        status: TraceStatus = TraceStatus.SUCCEEDED,
        blobs=(),
    ) -> None:
        nonlocal previous
        writer.append(
            record_id=record_id,
            stage=stage,
            status=status,
            event=event,
            payload=payload,
            parent_ids=(() if previous is None else (previous,)),
            blobs=blobs,
        )
        previous = record_id

    append(
        "h20:system:verified",
        TraceStage.SYSTEM,
        "worker_inputs_verified",
        {"request_sha256": "a" * 64},
        blobs=(manifest_blob,),
    )
    frame_specs = (
        {
            "index": 0,
            "detections": {"left": 2, "right": 2},
            "matches": 2,
            "hands": (("track-0000", "NEW", 0.03, True), ("track-0001", "NEW", 0.04, False)),
        },
        {
            "index": 1,
            "detections": {"left": 2, "right": 1},
            "matches": 1,
            "hands": (("track-0000", "MATCHED", 0.10, False),),
        },
    )
    for spec in frame_specs:
        index = spec["index"]
        frame_id = f"part0001/pair{index:06d}"
        timestamp_ns = 1_000_000_000 + index * 33_333_333
        common = {
            "frame_id": frame_id,
            "frame_index": index,
            "timestamp_ns": timestamp_ns,
        }
        append(
            f"h20:{index}:sync",
            TraceStage.SYNCHRONIZATION,
            "stereo_pair_selected",
            common,
        )
        for view_id, count in spec["detections"].items():
            detections = [
                {"candidate_id": f"{view_id}-{candidate}", "bbox_score": 0.9 - candidate * 0.1}
                for candidate in range(count)
            ]
            append(
                f"h20:{index}:detection:{view_id}",
                TraceStage.DETECTION,
                "hand_candidates_detected",
                {**common, "view_id": view_id, "detections": detections},
            )
        matches = [
            {
                "match_id": f"match-{match_index}",
                "left_index": match_index,
                "right_index": match_index,
            }
            for match_index in range(spec["matches"])
        ]
        append(
            f"h20:{index}:association",
            TraceStage.CROSS_VIEW_ASSOCIATION,
            "cross_view_hands_associated",
            {
                **common,
                "matches": matches,
                "unmatched_left_indices": ([1] if index == 1 else []),
                "unmatched_right_indices": [],
            },
        )
        assignments = [{"track_id": hand[0], "decision": hand[1]} for hand in spec["hands"]]
        append(
            f"h20:{index}:tracking",
            TraceStage.CROSS_VIEW_ASSOCIATION,
            "sequence_tracks_assigned",
            {**common, "assignments": assignments},
        )
        for hand_index, (track_id, decision, bone_length, mano_produced) in enumerate(
            spec["hands"]
        ):
            points, validity = _raw_points(bone_length)
            hand_common = {**common, "track_id": track_id}
            append(
                f"h20:{index}:raw:{hand_index}",
                TraceStage.RAW_FUSION,
                "raw_landmarks_triangulated",
                {
                    **hand_common,
                    "output_status": "PRODUCED",
                    "track_assignment": {"track_id": track_id, "decision": decision},
                    "landmarks_xyz_m": points,
                    "validity": validity,
                    "valid_landmark_count": 2,
                },
            )
            attempts = (
                [
                    {"side": "left", "status": "REJECTED", "rmse_m": 0.03},
                    {
                        "side": "right",
                        "status": "ACCEPTED" if mano_produced else "REJECTED",
                        "rmse_m": 0.009 if mano_produced else 0.04,
                    },
                ]
                if index == 0
                else [
                    {"side": "left", "status": "REJECTED", "rmse_m": 0.05},
                    {"side": "right", "status": "REJECTED", "rmse_m": 0.06},
                ]
            )
            selection = {
                "decision": "SELECTED" if mano_produced else "NO_HIGH_QUALITY_FIT",
                "attempts": attempts,
            }
            append(
                f"h20:{index}:mano:{hand_index}",
                TraceStage.KINEMATIC_REFINEMENT,
                "mano_frame_fitted" if mano_produced else "mano_frame_not_produced",
                {
                    **hand_common,
                    "output_status": "PRODUCED" if mano_produced else "NOT_PRODUCED",
                    "selection": selection,
                    **({"rmse_m": 0.009} if mano_produced else {}),
                },
                status=TraceStatus.SUCCEEDED if mano_produced else TraceStatus.WARNING,
            )
            temporal_input = "KINEMATIC_REFINEMENT" if mano_produced else "RAW_FUSION"
            append(
                f"h20:{index}:temporal:{hand_index}",
                TraceStage.TEMPORAL_REFINEMENT,
                "temporal_landmarks_refined",
                {
                    **hand_common,
                    "output_status": "PRODUCED",
                    "input_stage": temporal_input,
                    "method": "causal_time_ema_v1",
                },
            )
            append(
                f"h20:{index}:export:{hand_index}",
                TraceStage.EXPORT,
                "fhp21_record_exported",
                {**hand_common, "output_status": "PRODUCED"},
            )
    writer.finalize(status=RunStatus.COMPLETED, summary={"output_status": "PRODUCED"})
    return run_dir


def test_reader_extracts_versioned_frame_detection_association_and_configuration_baseline(
    tmp_path: Path,
) -> None:
    from fisheye_handpose.baseline import BASELINE_SCHEMA, extract_baseline_metrics

    reader = RunArtifactReader(_build_canonical_run(tmp_path))

    baseline = extract_baseline_metrics(reader)

    assert baseline["schema_version"] == BASELINE_SCHEMA
    assert baseline["run"] == {
        "run_id": "baseline-fixture",
        "status": "COMPLETED",
        "pipeline_version": "fixture-revision",
        "record_count": 23,
        "last_record_hash": reader.summary["last_record_hash"],
    }
    snapshot = baseline["configuration_snapshot"]
    assert snapshot["value"]["core"]["config"] == {"audit": {"max_skew_ns": 250_000}}
    assert snapshot["value"]["worker"]["configuration"]["thresholds"] == {
        "bbox_score": 0.3,
        "keypoint_score": 0.2,
    }
    assert len(snapshot["sha256"]) == 64
    assert baseline["frames"] == {
        "pair_count": 2,
        "unique_frame_count": 2,
        "frame_index_min": 0,
        "frame_index_max": 1,
    }
    assert baseline["detection"] == {
        "view_frame_count": 4,
        "candidate_count": 7,
        "candidate_count_histogram": {"1": 1, "2": 3},
        "per_view": {
            "left": {
                "view_frame_count": 2,
                "candidate_count": 4,
                "candidate_count_histogram": {"2": 2},
            },
            "right": {
                "view_frame_count": 2,
                "candidate_count": 3,
                "candidate_count_histogram": {"1": 1, "2": 1},
            },
        },
    }
    assert baseline["association"] == {
        "frame_count": 2,
        "match_count": 3,
        "match_count_histogram": {"1": 1, "2": 1},
        "two_hand_frame_count": 1,
        "unmatched_left_count": 1,
        "unmatched_right_count": 0,
    }


def test_reader_extracts_hand_raw_bone_track_mano_and_temporal_baseline(tmp_path: Path) -> None:
    from fisheye_handpose.baseline import extract_baseline_metrics

    baseline = extract_baseline_metrics(_build_canonical_run(tmp_path))

    assert baseline["hands"] == {
        "raw_hand_frame_count": 3,
        "exported_hand_frame_count": 3,
        "raw_hands_per_frame_histogram": {"1": 1, "2": 1},
        "frames_with_any_raw_hand": 2,
        "frames_with_two_raw_hands": 1,
    }
    assert baseline["raw_3d"] == {
        "hand_frame_count": 3,
        "joint_slot_count": 63,
        "valid_joint_count": 6,
        "valid_joint_rate": 0.095238095,
        "complete_hand_frame_count": 0,
        "invalid_reason_counts": {"LOW_KEYPOINT_SCORE": 57},
    }
    assert baseline["bone_lengths_m"]["overall"] == {
        "sample_count": 3,
        "min": 0.03,
        "median": 0.04,
        "p95": 0.094,
        "max": 0.1,
    }
    assert baseline["bone_lengths_m"]["threshold_counts"] == {
        "over_0_05_m": 1,
        "over_0_10_m": 0,
        "hand_frames_over_0_05_m": 1,
        "hand_frames_over_0_10_m": 0,
    }
    assert baseline["bone_lengths_m"]["by_edge"]["0-1"]["sample_count"] == 3
    assert baseline["bone_lengths_m"]["track_edge_cv"]["series_count"] == 1
    assert baseline["bone_lengths_m"]["track_edge_cv"]["median"] == 0.538461538
    assert baseline["tracks"] == {
        "track_count": 2,
        "new_assignment_count": 2,
        "matched_assignment_count": 1,
        "by_track": {
            "track-0000": {
                "hand_frame_count": 2,
                "first_frame_index": 0,
                "last_frame_index": 1,
                "new_assignment_count": 1,
                "matched_assignment_count": 1,
            },
            "track-0001": {
                "hand_frame_count": 1,
                "first_frame_index": 0,
                "last_frame_index": 0,
                "new_assignment_count": 1,
                "matched_assignment_count": 0,
            },
        },
    }
    assert baseline["mano"]["configured"] is True
    assert baseline["mano"]["hand_frame_count"] == 3
    assert baseline["mano"]["produced_count"] == 1
    assert baseline["mano"]["not_produced_count"] == 2
    assert baseline["mano"]["production_rate"] == 0.333333333
    assert baseline["mano"]["decision_counts"] == {
        "NO_HIGH_QUALITY_FIT": 2,
        "SELECTED": 1,
    }
    assert baseline["mano"]["attempt_status_counts"] == {"ACCEPTED": 1, "REJECTED": 5}
    assert baseline["mano"]["attempt_rmse_m"] == {
        "sample_count": 6,
        "min": 0.009,
        "median": 0.035,
        "p95": 0.0575,
        "max": 0.06,
    }
    assert baseline["mano"]["accepted_rmse_m"] == {
        "sample_count": 1,
        "min": 0.009,
        "median": 0.009,
        "p95": 0.009,
        "max": 0.009,
    }
    assert baseline["temporal"] == {
        "hand_frame_count": 3,
        "produced_count": 3,
        "not_produced_count": 0,
        "input_stage_counts": {"KINEMATIC_REFINEMENT": 1, "RAW_FUSION": 2},
        "method_counts": {"causal_time_ema_v1": 3},
    }


def test_trace_baseline_cli_writes_the_same_deterministic_public_contract(
    tmp_path: Path,
) -> None:
    from fisheye_handpose.baseline import extract_baseline_metrics
    from fisheye_handpose.cli import main

    run_dir = _build_canonical_run(tmp_path)
    output = tmp_path / "reports" / "baseline.json"

    code = main(["trace-baseline", str(run_dir), "--output", str(output)])

    assert code == 0
    first = output.read_bytes()
    assert json.loads(first) == extract_baseline_metrics(run_dir)
    assert main(["trace-baseline", str(run_dir), "--output", str(output)]) == 0
    assert output.read_bytes() == first


def test_hand_histogram_keeps_synchronized_zero_hand_frames(tmp_path: Path) -> None:
    from fisheye_handpose.baseline import extract_baseline_metrics

    run_dir = tmp_path / "zero-hand-run"
    writer = RunArtifactWriter.create(
        run_dir,
        run_id="zero-hand-fixture",
        pipeline_version="fixture-revision",
    )
    writer.append(
        record_id="h20:frame:sync",
        stage=TraceStage.SYNCHRONIZATION,
        status=TraceStatus.SUCCEEDED,
        event="stereo_pair_selected",
        payload={"frame_id": "part0001/pair000000", "frame_index": 0},
    )
    writer.finalize(status=RunStatus.COMPLETED)

    baseline = extract_baseline_metrics(run_dir)

    assert baseline["hands"] == {
        "raw_hand_frame_count": 0,
        "exported_hand_frame_count": 0,
        "raw_hands_per_frame_histogram": {"0": 1},
        "frames_with_any_raw_hand": 0,
        "frames_with_two_raw_hands": 0,
    }
    assert baseline["configuration_snapshot"]["value"]["worker"] is None
    assert baseline["raw_3d"]["valid_joint_rate"] is None
    assert baseline["mano"]["configured"] is None
