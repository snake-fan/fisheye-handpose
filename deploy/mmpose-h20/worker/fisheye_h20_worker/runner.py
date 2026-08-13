"""Complete H20 stereo worker composition and immutable process artifact export."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .artifacts import ResultWriter
from .assets import EXPECTED_CONFIGS, MMPOSE_COMMIT, verify_assets, verify_source_report
from .calibration import load_rectified_stereo
from .contracts import WorkerError, load_request
from .geometry import associate, normalize_instances, triangulate_match
from .mano import MANO_FHP21_MAPPING_ID, verify_mano_assets
from .output_contract import build_pose_estimate
from .runtime import OpenMMLabRuntime
from .session import discover_parts, match_part, selected_frames
from .temporal import CausalTemporalRefiner
from .tracking import SequenceTracker


def _serialized_assets(report: dict[str, Any]) -> dict[str, Any]:
    return {
        artifact_id: {
            key: str(value) if isinstance(value, Path) else value
            for key, value in artifact.items()
            if key != "config_path"
        }
        for artifact_id, artifact in report["artifacts"].items()
    }


def _should_save_source(policy: Any, global_pair_index: int) -> bool:
    return policy.source_frames == "ALL" or (
        policy.source_frames == "SAMPLED" and global_pair_index % policy.sample_every == 0
    )


def _resolved_configuration(request: Any) -> dict[str, Any]:
    return {
        "session": {
            "path": str(request.session.path),
            "timestamp_column": request.session.timestamp_column,
            "timestamp_unit": request.session.timestamp_unit,
            "max_skew_ns": request.session.max_skew_ns,
            "clock_offset_ns": request.session.clock_offset_ns,
            "max_pairs": request.session.max_pairs,
        },
        "calibration": {
            "path": str(request.calibration.path),
            "left_camera_id": request.calibration.left_camera_id,
            "right_camera_id": request.calibration.right_camera_id,
            "translation_unit": request.calibration.translation_unit,
            "extrinsics_convention": request.calibration.extrinsics_convention,
            "output_size": list(request.calibration.output_size),
            "balance": request.calibration.balance,
            "fov_scale": request.calibration.fov_scale,
        },
        "thresholds": {
            "bbox_score": request.thresholds.bbox_score,
            "keypoint_score": request.thresholds.keypoint_score,
            "association_epipolar_px": request.thresholds.association_epipolar_px,
            "max_reprojection_error_px": request.thresholds.max_reprojection_error_px,
            "min_ray_angle_deg": request.thresholds.min_ray_angle_deg,
        },
        "models": {
            "device": request.models.device,
            "detector_category_id": request.models.detector_category_id,
            "license_risk_acknowledged": request.models.license_risk_acknowledged,
        },
        "artifacts": {
            "source_frames": request.artifacts.source_frames,
            "sample_every": request.artifacts.sample_every,
            "image_format": request.artifacts.image_format,
        },
        "tracking": {
            "max_root_distance_m": request.tracking.max_root_distance_m,
            "max_gap_ms": request.tracking.max_gap_ms,
            "method": "sequence_root_distance_one_to_one_v1",
        },
        "mano": (
            None
            if request.mano is None
            else {
                "model_root": str(request.mano.model_root),
                "manifest": str(request.mano.manifest),
                "min_valid_landmarks": request.mano.min_valid_landmarks,
                "max_fit_rmse_m": request.mano.max_fit_rmse_m,
                "iterations": request.mano.iterations,
                "learning_rate": request.mano.learning_rate,
                "mapping_id": MANO_FHP21_MAPPING_ID,
            }
        ),
        "temporal": {
            "method": request.temporal.method,
            "time_constant_ms": request.temporal.time_constant_ms,
            "gap_reset_ms": request.temporal.gap_reset_ms,
        },
    }


def _json_blob(
    writer: ResultWriter,
    value: Any,
    *,
    role: str,
) -> dict[str, Any]:
    try:
        data = (
            json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WorkerError(f"stage artifact is not strict JSON: {exc}") from exc
    return writer.put_blob(
        data,
        role=role,
        media_type="application/json",
        suffix=".json",
    )


def _detection(instance: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": instance["candidate_id"],
        "bbox_xyxy": instance["bbox_xyxy"],
        "score": instance["bbox_score"],
        "bbox_score": instance["bbox_score"],
        "label": instance["label"],
    }


def _validate_mano_fit(value: Any, *, expected_side: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("side") != expected_side:
        raise WorkerError("MANO runtime returned an invalid side")
    if value.get("mapping_id") != MANO_FHP21_MAPPING_ID:
        raise WorkerError("MANO runtime returned an unexpected FHP21 mapping")
    fields = {
        "landmarks_xyz_m": 21,
        "validity": 21,
        "global_orient": 3,
        "hand_pose": 45,
        "transl": 3,
        "beta": 10,
    }
    for field, length in fields.items():
        item = value.get(field)
        if not isinstance(item, list) or len(item) != length:
            raise WorkerError(f"MANO runtime {field} must contain {length} values")
    for point in value["landmarks_xyz_m"]:
        if (
            not isinstance(point, list)
            or len(point) != 3
            or any(
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
                for number in point
            )
        ):
            raise WorkerError("MANO runtime landmarks_xyz_m contains an invalid point")
    for field in ("global_orient", "hand_pose", "transl", "beta"):
        if any(
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            for number in value[field]
        ):
            raise WorkerError(f"MANO runtime {field} contains a non-finite value")
    rmse = value.get("rmse_m")
    if (
        isinstance(rmse, bool)
        or not isinstance(rmse, (int, float))
        or not math.isfinite(float(rmse))
        or float(rmse) < 0
    ):
        raise WorkerError("MANO runtime rmse_m is invalid")
    return value


def _fit_mano_frame(
    *,
    runtime: Any,
    models: Any,
    request: Any,
    device: str,
    track_id: str,
    raw: dict[str, Any],
    track_states: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    valid_count = raw["valid_landmark_count"]
    if valid_count < request.min_valid_landmarks:
        return None, {
            "decision": "INSUFFICIENT_VALID_LANDMARKS",
            "valid_landmark_count": valid_count,
            "required_valid_landmark_count": request.min_valid_landmarks,
            "attempts": [],
        }
    state = track_states.get(track_id)
    sides = (state["handedness"],) if state is not None else ("left", "right")
    attempts: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for side in sides:
        try:
            fit = _validate_mano_fit(
                runtime.fit_mano(
                    models,
                    side=side,
                    target_xyz_m=raw["landmarks_xyz_m"],
                    validity=raw["validity"],
                    fixed_beta=(None if state is None else state["beta"]),
                    device=device,
                    iterations=request.iterations,
                    learning_rate=request.learning_rate,
                ),
                expected_side=side,
            )
        except Exception as exc:
            attempts.append(
                {
                    "side": side,
                    "status": "ERROR",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            continue
        accepted_fit = float(fit["rmse_m"]) <= request.max_fit_rmse_m
        attempts.append(
            {
                "side": side,
                "status": "ACCEPTED" if accepted_fit else "REJECTED",
                "rmse_m": fit["rmse_m"],
                "max_fit_rmse_m": request.max_fit_rmse_m,
            }
        )
        if accepted_fit:
            accepted.append(fit)
    if not accepted:
        return None, {
            "decision": "NO_HIGH_QUALITY_FIT",
            "valid_landmark_count": valid_count,
            "attempts": attempts,
        }
    selected = min(accepted, key=lambda fit: (float(fit["rmse_m"]), fit["side"]))
    first_selection = state is None
    if first_selection:
        track_states[track_id] = {
            "handedness": selected["side"],
            "beta": list(selected["beta"]),
        }
    return selected, {
        "decision": "SELECTED",
        "first_high_quality_frame": first_selection,
        "selected_side": selected["side"],
        "attempts": attempts,
    }


def run_worker(
    request_path: str | Path,
    result_dir: str | Path,
    *,
    runtime: Any | None = None,
) -> dict[str, Any]:
    request = load_request(request_path)
    destination = Path(result_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"worker result already exists: {destination}")
    runtime = OpenMMLabRuntime() if runtime is None else runtime
    assets = verify_assets(request.models)
    mano_assets = (
        None
        if request.mano is None
        else verify_mano_assets(request.mano.model_root, request.mano.manifest)
    )
    source_report = verify_source_report(
        runtime.verify_source(
            source=request.models.mmpose_source,
            expected_commit=MMPOSE_COMMIT,
            config_relative_paths=list(EXPECTED_CONFIGS.values()),
        )
    )
    rectification = load_rectified_stereo(request.calibration)
    parts = discover_parts(request.session.path)
    pairs_by_part = {part.part_number: match_part(part, request.session) for part in parts}
    manifest = {
        "request": {
            "path": str(request.source_path),
            "sha256": request.source_sha256,
        },
        "session": {"path": str(request.session.path), "part_count": len(parts)},
        "configuration": _resolved_configuration(request),
        "calibration": rectification.to_dict(),
        "models": {
            "manifest": str(assets["manifest"]),
            "manifest_sha256": assets["manifest_sha256"],
            "artifacts": _serialized_assets(assets),
            "mmpose_source": {
                "path": str(request.models.mmpose_source),
                **source_report,
            },
            "mano": mano_assets,
        },
    }
    writer = ResultWriter(destination, manifest)
    processed_pairs = 0
    matched_hands = 0
    valid_landmarks = 0
    mano_outputs = 0
    temporal_outputs = 0
    export_count = 0
    tracker = SequenceTracker(
        max_root_distance_m=request.tracking.max_root_distance_m,
        max_gap_ms=request.tracking.max_gap_ms,
    )
    temporal_refiner = CausalTemporalRefiner(
        time_constant_ms=request.temporal.time_constant_ms,
        gap_reset_ms=request.temporal.gap_reset_ms,
    )
    mano_track_states: dict[str, dict[str, Any]] = {}
    tracking_state_events: dict[str, tuple[str, int]] = {}
    mano_state_events: dict[str, str] = {}
    temporal_state_events: dict[str, str] = {}
    backend_provenance = {
        "producer": "fisheye_h20_worker",
        "producer_version": "h20-worker/v1",
        "worker_request_sha256": request.source_sha256,
        "model_manifest_sha256": assets["manifest_sha256"],
        "mmpose_commit": source_report["commit"],
        "detector": {
            "id": "rtmdet-nano-hand",
            "sha256": assets["artifacts"]["rtmdet-nano-hand"]["sha256"],
            "config": assets["artifacts"]["rtmdet-nano-hand"]["config"],
        },
        "pose": {
            "id": "rtmpose-m-hand5",
            "sha256": assets["artifacts"]["rtmpose-m-hand5"]["sha256"],
            "config": assets["artifacts"]["rtmpose-m-hand5"]["config"],
        },
        "fusion_method": "rectified_stereo_dlt_v1",
        "kinematic_method": ("mano_v1.2_fit_v1" if request.mano is not None else "NONE"),
        "temporal_method": request.temporal.method,
    }
    active_stage = "SYSTEM"
    active_frame_id: str | None = None
    try:
        writer.append(
            event_id="system:verified",
            stage="SYSTEM",
            status="SUCCEEDED",
            event="worker_inputs_verified",
            payload={
                "request_sha256": request.source_sha256,
                "model_manifest_sha256": assets["manifest_sha256"],
                "mmpose_commit": source_report["commit"],
                "mano_manifest_sha256": (
                    None if mano_assets is None else mano_assets["manifest_sha256"]
                ),
            },
        )
        writer.append(
            event_id="calibration:rectification",
            stage="RECTIFICATION",
            status="SUCCEEDED",
            event="worker_rectification_loaded",
            payload=rectification.to_dict(),
            parent_event_ids=("system:verified",),
        )
        detector = assets["artifacts"]["rtmdet-nano-hand"]
        pose = assets["artifacts"]["rtmpose-m-hand5"]
        models = runtime.load_models(
            det_config=detector["config_path"],
            det_checkpoint=detector["checkpoint"],
            pose_config=pose["config_path"],
            pose_checkpoint=pose["checkpoint"],
            device=request.models.device,
        )
        if request.mano is None:
            mano_models = None
            writer.append(
                event_id="mano:configuration",
                stage="KINEMATIC_REFINEMENT",
                status="SKIPPED",
                event="mano_not_configured",
                payload={
                    "reason": "request.mano is null or absent",
                    "output_status": "NOT_PRODUCED",
                    "mapping_id": MANO_FHP21_MAPPING_ID,
                },
                parent_event_ids=("calibration:rectification",),
            )
        else:
            assert mano_assets is not None
            mano_models = runtime.load_mano_models(
                model_root=request.mano.model_root,
                device=request.models.device,
            )
            writer.append(
                event_id="mano:configuration",
                stage="KINEMATIC_REFINEMENT",
                status="SUCCEEDED",
                event="mano_models_loaded",
                payload={
                    "output_status": "NOT_PRODUCED",
                    "model_status": "READY",
                    "manifest_sha256": mano_assets["manifest_sha256"],
                    "mapping_id": MANO_FHP21_MAPPING_ID,
                    "sides": ["left", "right"],
                },
                parent_event_ids=("calibration:rectification",),
            )

        for part in parts:
            remaining = request.session.max_pairs - processed_pairs
            if remaining <= 0:
                break
            pairs = pairs_by_part[part.part_number][:remaining]
            left_frames = selected_frames(
                runtime, part.left_video, (pair.left_index for pair in pairs)
            )
            right_frames = selected_frames(
                runtime, part.right_video, (pair.right_index for pair in pairs)
            )
            for pair in pairs:
                global_index = processed_pairs
                frame_id = f"part{part.part_number:04d}/pair{pair.pair_index:06d}"
                active_frame_id = frame_id
                event_prefix = f"part{part.part_number:04d}:pair{pair.pair_index:06d}"
                left_frame = left_frames[pair.left_index]
                right_frame = right_frames[pair.right_index]
                source_blobs: list[dict[str, Any]] = []
                if _should_save_source(request.artifacts, global_index):
                    suffix = f".{request.artifacts.image_format}"
                    media_type = (
                        "image/jpeg" if request.artifacts.image_format == "jpg" else "image/png"
                    )
                    for side, frame in (("left", left_frame), ("right", right_frame)):
                        source_blobs.append(
                            writer.put_blob(
                                runtime.encode_frame(frame, request.artifacts.image_format),
                                role=f"source_{side}",
                                media_type=media_type,
                                suffix=suffix,
                            )
                        )
                writer.append(
                    event_id=f"{event_prefix}:sync",
                    stage="SYNCHRONIZATION",
                    status="SUCCEEDED",
                    event="stereo_pair_selected",
                    payload={
                        "frame_id": frame_id,
                        "frame_index": global_index,
                        **pair.to_dict(),
                    },
                    blobs=source_blobs,
                    parent_event_ids=("calibration:rectification",),
                )
                views: dict[str, list[dict[str, Any]]] = {}
                for side, frame in (("left", left_frame), ("right", right_frame)):
                    active_stage = "DETECTION"
                    raw_instances = runtime.infer(
                        models,
                        frame,
                        bbox_threshold=request.thresholds.bbox_score,
                        category_id=request.models.detector_category_id,
                        max_instances=2,
                    )
                    instances = normalize_instances(
                        raw_instances,
                        side=side,
                        rectification=rectification,
                    )
                    active_stage = "POSE_2D"
                    views[side] = instances
                    common = {
                        "frame_id": frame_id,
                        "frame_index": global_index,
                        "timestamp_ns": pair.pair_timestamp_ns,
                        "view_id": side,
                        "image_width": rectification.image_size[0],
                        "image_height": rectification.image_size[1],
                    }
                    detections = [_detection(instance) for instance in instances]
                    writer.append(
                        event_id=f"{event_prefix}:detection:{side}",
                        stage="DETECTION",
                        status="SUCCEEDED" if detections else "WARNING",
                        event="hand_candidates_detected",
                        payload={
                            **common,
                            "output_status": ("PRODUCED" if detections else "NOT_PRODUCED"),
                            "detections": detections,
                            "instances": detections,
                        },
                        parent_event_ids=(f"{event_prefix}:sync",),
                    )
                    writer.append(
                        event_id=f"{event_prefix}:pose2d:{side}",
                        stage="POSE_2D",
                        status="SUCCEEDED" if instances else "WARNING",
                        event="view_keypoints_inferred",
                        payload={
                            **common,
                            "landmark_schema": "fhp21/v1",
                            "output_status": ("PRODUCED" if instances else "NOT_PRODUCED"),
                            "instances": instances,
                        },
                        parent_event_ids=(f"{event_prefix}:detection:{side}",),
                    )
                active_stage = "CROSS_VIEW_ASSOCIATION"
                association = associate(views["left"], views["right"], request.thresholds)
                writer.append(
                    event_id=f"{event_prefix}:association",
                    stage="CROSS_VIEW_ASSOCIATION",
                    status="SUCCEEDED" if association["matches"] else "WARNING",
                    event="cross_view_hands_associated",
                    payload={
                        "frame_id": frame_id,
                        "frame_index": global_index,
                        "timestamp_ns": pair.pair_timestamp_ns,
                        "output_status": ("PRODUCED" if association["matches"] else "NOT_PRODUCED"),
                        **association,
                    },
                    parent_event_ids=(
                        f"{event_prefix}:pose2d:left",
                        f"{event_prefix}:pose2d:right",
                    ),
                )
                observations: list[dict[str, Any]] = []
                for match in association["matches"]:
                    active_stage = "RAW_FUSION"
                    raw = triangulate_match(
                        views["left"][match["left_index"]],
                        views["right"][match["right_index"]],
                        rectification=rectification,
                        thresholds=request.thresholds,
                    )
                    observations.append(
                        {
                            "observation_id": f"{event_prefix}:{match['match_id']}",
                            "match": match,
                            **raw,
                        }
                    )
                active_stage = "CROSS_VIEW_ASSOCIATION"
                assignments = tracker.assign(
                    observations,
                    timestamp_ns=pair.pair_timestamp_ns,
                )
                tracking_max_gap_ns = int(request.tracking.max_gap_ms * 1_000_000)
                tracking_predecessors = sorted(
                    {
                        event_id
                        for event_id, state_timestamp_ns in tracking_state_events.values()
                        if 0 <= pair.pair_timestamp_ns - state_timestamp_ns <= tracking_max_gap_ns
                    }
                )
                tracking_event_id = f"{event_prefix}:tracking"
                writer.append(
                    event_id=tracking_event_id,
                    stage="CROSS_VIEW_ASSOCIATION",
                    status="SUCCEEDED" if assignments else "WARNING",
                    event="sequence_tracks_assigned",
                    payload={
                        "frame_id": frame_id,
                        "frame_index": global_index,
                        "timestamp_ns": pair.pair_timestamp_ns,
                        "method": "sequence_root_distance_one_to_one_v1",
                        "output_status": ("PRODUCED" if assignments else "NOT_PRODUCED"),
                        "state_predecessor_event_ids": tracking_predecessors,
                        "assignments": assignments,
                    },
                    parent_event_ids=(
                        f"{event_prefix}:association",
                        *tracking_predecessors,
                    ),
                )
                for assignment in assignments:
                    tracking_state_events[assignment["track_id"]] = (
                        tracking_event_id,
                        pair.pair_timestamp_ns,
                    )
                if not assignments:
                    empty_payload = {
                        "frame_id": frame_id,
                        "frame_index": global_index,
                        "timestamp_ns": pair.pair_timestamp_ns,
                        "track_id": None,
                        "output_status": "NOT_PRODUCED",
                        "reason": "NO_CROSS_VIEW_MATCH",
                    }
                    writer.append(
                        event_id=f"{event_prefix}:raw:none",
                        stage="RAW_FUSION",
                        status="WARNING",
                        event="raw_landmarks_not_produced",
                        payload=empty_payload,
                        parent_event_ids=(f"{event_prefix}:tracking",),
                    )
                    writer.append(
                        event_id=f"{event_prefix}:mano:none",
                        stage="KINEMATIC_REFINEMENT",
                        status="SKIPPED",
                        event="kinematic_refinement_not_produced",
                        payload={**empty_payload, "reason": "NO_RAW_FUSION_OBSERVATION"},
                        parent_event_ids=(f"{event_prefix}:raw:none",),
                    )
                    writer.append(
                        event_id=f"{event_prefix}:temporal:none",
                        stage="TEMPORAL_REFINEMENT",
                        status="SKIPPED",
                        event="temporal_refinement_not_produced",
                        payload={**empty_payload, "reason": "NO_KINEMATIC_OR_RAW_OBSERVATION"},
                        parent_event_ids=(f"{event_prefix}:mano:none",),
                    )
                    writer.append(
                        event_id=f"{event_prefix}:export:none",
                        stage="EXPORT",
                        status="SKIPPED",
                        event="fhp21_record_not_produced",
                        payload={
                            **empty_payload,
                            "reason": "NO_TEMPORAL_POSE_ESTIMATE",
                            "output_file": None,
                        },
                        parent_event_ids=(f"{event_prefix}:temporal:none",),
                    )
                for observation, assignment in zip(observations, assignments, strict=True):
                    match = observation["match"]
                    track_id = assignment["track_id"]
                    tracked_pose_ids: list[str] = []
                    for side, candidate_index in (
                        ("left", match["left_index"]),
                        ("right", match["right_index"]),
                    ):
                        instance = views[side][candidate_index]
                        event_id = f"{event_prefix}:pose2d:tracked:{match['match_id']}:{side}"
                        writer.append(
                            event_id=event_id,
                            stage="POSE_2D",
                            status="SUCCEEDED",
                            event="tracked_view_keypoints_recorded",
                            payload={
                                "frame_id": frame_id,
                                "frame_index": global_index,
                                "timestamp_ns": pair.pair_timestamp_ns,
                                "view_id": side,
                                "track_id": track_id,
                                "candidate_id": instance["candidate_id"],
                                "landmark_schema": "fhp21/v1",
                                "output_status": "PRODUCED",
                                "detections": [_detection(instance)],
                                "keypoints_uv": instance["keypoints_uv"],
                                "keypoints_uv_rectified": instance["keypoints_uv_rectified"],
                                "keypoint_scores": instance["keypoint_scores"],
                                "image_width": rectification.image_size[0],
                                "image_height": rectification.image_size[1],
                            },
                            parent_event_ids=(f"{event_prefix}:tracking",),
                        )
                        tracked_pose_ids.append(event_id)

                    raw = {
                        key: value
                        for key, value in observation.items()
                        if key not in {"observation_id", "match"}
                    }
                    matched_hands += 1
                    valid_landmarks += raw["valid_landmark_count"]
                    raw_payload = {
                        "frame_id": frame_id,
                        "frame_index": global_index,
                        "timestamp_ns": pair.pair_timestamp_ns,
                        "track_id": track_id,
                        "observation_id": observation["observation_id"],
                        "match": match,
                        "track_assignment": assignment,
                        "output_status": (
                            "PRODUCED" if raw["valid_landmark_count"] else "NOT_PRODUCED"
                        ),
                        **raw,
                    }
                    raw_event_id = f"{event_prefix}:raw:{match['match_id']}"
                    writer.append(
                        event_id=raw_event_id,
                        stage="RAW_FUSION",
                        status=("SUCCEEDED" if raw["valid_landmark_count"] else "WARNING"),
                        event="raw_landmarks_triangulated",
                        payload=raw_payload,
                        blobs=[
                            _json_blob(
                                writer,
                                raw_payload,
                                role="raw_fhp21",
                            )
                        ],
                        parent_event_ids=tuple(tracked_pose_ids),
                    )

                    mano_fit: dict[str, Any] | None = None
                    temporal_parent = raw_event_id
                    mano_event_id = f"{event_prefix}:mano:{match['match_id']}"
                    if request.mano is not None:
                        active_stage = "KINEMATIC_REFINEMENT"
                        assert mano_models is not None
                        mano_fit, selection = _fit_mano_frame(
                            runtime=runtime,
                            models=mano_models,
                            request=request.mano,
                            device=request.models.device,
                            track_id=track_id,
                            raw=raw,
                            track_states=mano_track_states,
                        )
                        mano_predecessor = mano_state_events.get(track_id)
                        if mano_fit is None:
                            mano_payload = {
                                "frame_id": frame_id,
                                "frame_index": global_index,
                                "timestamp_ns": pair.pair_timestamp_ns,
                                "track_id": track_id,
                                "input_stage": "RAW_FUSION",
                                "output_status": "NOT_PRODUCED",
                                "state_predecessor_event_id": mano_predecessor,
                                "selection": selection,
                            }
                            writer.append(
                                event_id=mano_event_id,
                                stage="KINEMATIC_REFINEMENT",
                                status="WARNING",
                                event="mano_frame_not_produced",
                                payload=mano_payload,
                                parent_event_ids=(
                                    (raw_event_id,)
                                    if mano_predecessor is None
                                    else (raw_event_id, mano_predecessor)
                                ),
                            )
                        else:
                            state = mano_track_states[track_id]
                            mano_outputs += 1
                            mano_payload = {
                                "frame_id": frame_id,
                                "frame_index": global_index,
                                "timestamp_ns": pair.pair_timestamp_ns,
                                "track_id": track_id,
                                "input_stage": "RAW_FUSION",
                                "output_status": "PRODUCED",
                                "state_predecessor_event_id": mano_predecessor,
                                "coordinate_frame": "rectified_left_camera",
                                "length_unit": "m",
                                "landmark_schema": "fhp21/v1",
                                "handedness": state["handedness"],
                                "selection": selection,
                                "beta_frozen": not selection["first_high_quality_frame"],
                                "pose": mano_fit["hand_pose"],
                                "global_orient": mano_fit["global_orient"],
                                "transl": mano_fit["transl"],
                                "beta": state["beta"],
                                "mapping_id": mano_fit["mapping_id"],
                                "rmse_m": mano_fit["rmse_m"],
                                "loss": {
                                    "metric": "RMSE_M",
                                    "value": mano_fit["rmse_m"],
                                },
                                "landmarks_xyz_m": mano_fit["landmarks_xyz_m"],
                                "validity": mano_fit["validity"],
                            }
                            writer.append(
                                event_id=mano_event_id,
                                stage="KINEMATIC_REFINEMENT",
                                status="SUCCEEDED",
                                event="mano_frame_fitted",
                                payload=mano_payload,
                                blobs=[
                                    _json_blob(
                                        writer,
                                        mano_payload,
                                        role="mano_fhp21",
                                    )
                                ],
                                parent_event_ids=(
                                    (raw_event_id,)
                                    if mano_predecessor is None
                                    else (raw_event_id, mano_predecessor)
                                ),
                            )
                        mano_state_events[track_id] = mano_event_id
                        temporal_parent = mano_event_id
                    else:
                        writer.append(
                            event_id=mano_event_id,
                            stage="KINEMATIC_REFINEMENT",
                            status="SKIPPED",
                            event="mano_frame_not_configured",
                            payload={
                                "frame_id": frame_id,
                                "frame_index": global_index,
                                "timestamp_ns": pair.pair_timestamp_ns,
                                "track_id": track_id,
                                "input_stage": "RAW_FUSION",
                                "output_status": "NOT_PRODUCED",
                                "reason": "MANO_NOT_CONFIGURED",
                                "mapping_id": MANO_FHP21_MAPPING_ID,
                            },
                            parent_event_ids=(raw_event_id,),
                        )
                        temporal_parent = mano_event_id

                    temporal_input = mano_fit if mano_fit is not None else raw
                    temporal_input_stage = (
                        "KINEMATIC_REFINEMENT" if mano_fit is not None else "RAW_FUSION"
                    )
                    active_stage = "TEMPORAL_REFINEMENT"
                    temporal = temporal_refiner.refine(
                        track_id=track_id,
                        timestamp_ns=pair.pair_timestamp_ns,
                        landmarks_xyz_m=temporal_input["landmarks_xyz_m"],
                        validity=temporal_input["validity"],
                        input_stage=temporal_input_stage,
                    )
                    temporal_produced = temporal["valid_landmark_count"] > 0
                    if temporal_produced:
                        active_stage = "EXPORT"
                        temporal_outputs += 1
                    temporal_payload = {
                        "frame_id": frame_id,
                        "frame_index": global_index,
                        "timestamp_ns": pair.pair_timestamp_ns,
                        "track_id": track_id,
                        "input_stage": temporal_input_stage,
                        "state_predecessor_event_id": temporal_state_events.get(track_id),
                        "output_status": ("PRODUCED" if temporal_produced else "NOT_PRODUCED"),
                        "coordinate_frame": "rectified_left_camera",
                        "length_unit": "m",
                        "landmark_schema": "fhp21/v1",
                        **temporal,
                    }
                    temporal_event_id = f"{event_prefix}:temporal:{match['match_id']}"
                    temporal_blob = _json_blob(
                        writer,
                        temporal_payload,
                        role="temporal_fhp21",
                    )
                    writer.append(
                        event_id=temporal_event_id,
                        stage="TEMPORAL_REFINEMENT",
                        status="SUCCEEDED" if temporal_produced else "WARNING",
                        event=(
                            "temporal_landmarks_refined"
                            if temporal_produced
                            else "temporal_landmarks_not_produced"
                        ),
                        payload=temporal_payload,
                        blobs=[temporal_blob],
                        parent_event_ids=(
                            (temporal_parent,)
                            if temporal_state_events.get(track_id) is None
                            else (temporal_parent, temporal_state_events[track_id])
                        ),
                    )
                    temporal_state_events[track_id] = temporal_event_id
                    if temporal_produced:
                        mano_export = (
                            None
                            if mano_fit is None
                            else {
                                "side": mano_fit["side"],
                                "handedness": mano_fit["side"],
                                "mapping_id": mano_fit["mapping_id"],
                                "pose": mano_fit["hand_pose"],
                                "global_orient": mano_fit["global_orient"],
                                "transl": mano_fit["transl"],
                                "beta": mano_track_states[track_id]["beta"],
                                "rmse_m": mano_fit["rmse_m"],
                                "loss": {
                                    "metric": "RMSE_M",
                                    "value": mano_fit["rmse_m"],
                                },
                                "landmarks_xyz_m": mano_fit["landmarks_xyz_m"],
                                "validity": mano_fit["validity"],
                            }
                        )
                        export_value = build_pose_estimate(
                            sequence_id=request.session.path.name,
                            estimate_id=f"{event_prefix}:{match['match_id']}:temporal-estimate",
                            frame_id=frame_id,
                            frame_index=global_index,
                            timestamp_ns=pair.pair_timestamp_ns,
                            track_id=track_id,
                            source_observation_id=observation["observation_id"],
                            calibration_id=rectification.calibration_id,
                            raw=raw,
                            mano=mano_export,
                            temporal=temporal,
                            keypoint_score_threshold=request.thresholds.keypoint_score,
                            backend_provenance=backend_provenance,
                        )
                        writer.append_fhp21(export_value)
                        writer.append(
                            event_id=f"{event_prefix}:export:{match['match_id']}",
                            stage="EXPORT",
                            status=(
                                "SUCCEEDED" if temporal["valid_landmark_count"] == 21 else "WARNING"
                            ),
                            event="fhp21_record_exported",
                            payload={
                                **export_value,
                                "output_file": "fhp21.jsonl",
                            },
                            parent_event_ids=(temporal_event_id,),
                        )
                        export_count += 1
                    else:
                        active_stage = "EXPORT"
                        writer.append(
                            event_id=f"{event_prefix}:export:{match['match_id']}",
                            stage="EXPORT",
                            status="SKIPPED",
                            event="fhp21_record_not_produced",
                            payload={
                                "frame_id": frame_id,
                                "frame_index": global_index,
                                "timestamp_ns": pair.pair_timestamp_ns,
                                "track_id": track_id,
                                "source_observation_ids": [observation["observation_id"]],
                                "output_status": "NOT_PRODUCED",
                                "output_file": None,
                                "selected_output_stage": "TEMPORAL_REFINEMENT",
                                "reason": "NO_VALID_TEMPORAL_LANDMARK",
                                "landmarks_xyz_m": temporal["landmarks_xyz_m"],
                                "validity": temporal["validity"],
                            },
                            parent_event_ids=(temporal_event_id,),
                        )
                processed_pairs += 1
        if processed_pairs == 0:
            raise WorkerError("worker processed no synchronized pairs")
        summary = writer.finalize(
            status="COMPLETED",
            summary={
                "pair_count": processed_pairs,
                "matched_hand_count": matched_hands,
                "valid_landmark_count": valid_landmarks,
                "mano_output_count": mano_outputs,
                "temporal_output_count": temporal_outputs,
                "export_count": export_count,
                "output_status": "PRODUCED" if export_count else "NOT_PRODUCED",
                "output_file": "fhp21.jsonl" if export_count else None,
            },
        )
    except BaseException as exc:
        try:
            writer.append(
                event_id="system:failure",
                stage=active_stage,
                status="FAILED",
                event="worker_execution_failed",
                payload={
                    "output_status": "NOT_PRODUCED",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "pair_count": processed_pairs,
                    "export_count": export_count,
                    "failed_stage": active_stage,
                    "frame_id": active_frame_id,
                },
            )
        except BaseException:
            pass
        try:
            writer.finalize(
                status="FAILED",
                summary={
                    "pair_count": processed_pairs,
                    "matched_hand_count": matched_hands,
                    "valid_landmark_count": valid_landmarks,
                    "mano_output_count": mano_outputs,
                    "temporal_output_count": temporal_outputs,
                    "export_count": export_count,
                    "output_status": "NOT_PRODUCED",
                    "output_file": None,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
        except BaseException:
            pass
        raise
    return {"result_dir": str(destination), **summary}


__all__ = ["run_worker"]
