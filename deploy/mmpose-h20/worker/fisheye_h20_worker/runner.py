"""H20 worker lifecycle composition and immutable process artifact export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import ResultWriter
from .assets import EXPECTED_CONFIGS, MMPOSE_COMMIT, verify_assets, verify_source_report
from .calibration import load_rectified_stereo
from .candidates import CandidatePolicy
from .contracts import WorkerError, load_request
from .crop import VirtualPerspectiveCropper
from .frame_execution import (
    FrameExecutionContext,
    FrameExecutionState,
    FrameExecutor,
    FrameInput,
)
from .fusion import FUSION_METHOD_ID
from .mano import MANO_FHP21_MAPPING_ID, verify_mano_assets
from .mano_fitting import ManoTrackFitter
from .pose_adapter import VirtualCropPoseAdapter
from .runtime import OpenMMLabRuntime
from .scores import MODEL_SCORE_SEMANTICS, QUALITY_WEIGHT_METHOD, QUALITY_WEIGHT_STATUS
from .session import discover_parts, match_part, selected_frames
from .temporal import CausalTemporalRefiner
from .tracking import MOTION_METHOD_ID, PALM_ANCHOR_METHOD_ID, SequenceTracker
from .visualization import RawVsStableOverlayVideo


def _serialized_assets(report: dict[str, Any]) -> dict[str, Any]:
    return {
        artifact_id: {
            key: str(value) if isinstance(value, Path) else value
            for key, value in artifact.items()
            if key != "config_path"
        }
        for artifact_id, artifact in report["artifacts"].items()
    }


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
            "min_depth_m": request.thresholds.min_depth_m,
            "max_depth_m": request.thresholds.max_depth_m,
        },
        "perception": {
            "pose_input": request.perception.pose_input,
            "crop_output_size": list(request.perception.crop_output_size),
            "crop_bbox_scale": request.perception.crop_bbox_scale,
            "crop_min_valid_fraction": request.perception.crop_min_valid_fraction,
            "crop_policy_id": "virtual-perspective-kb4/v1",
            "recovery_bbox_score": request.perception.recovery_bbox_score,
            "max_candidates_per_view": request.perception.max_candidates_per_view,
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
            "overlay_video": request.artifacts.overlay_video,
        },
        "tracking": {
            "max_root_distance_m": request.tracking.max_root_distance_m,
            "max_gap_ms": request.tracking.max_gap_ms,
            "method": MOTION_METHOD_ID,
            "anchor_method": PALM_ANCHOR_METHOD_ID,
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


def _selected_pair_timestamps(
    *,
    parts: list[Any],
    pairs_by_part: dict[int, list[Any]],
    max_pairs: int,
) -> list[int]:
    timestamps: list[int] = []
    for part in parts:
        remaining = max_pairs - len(timestamps)
        if remaining <= 0:
            break
        timestamps.extend(
            pair.pair_timestamp_ns for pair in pairs_by_part[part.part_number][:remaining]
        )
    return timestamps


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
    virtual_pose_adapter = (
        VirtualCropPoseAdapter(
            cropper=VirtualPerspectiveCropper(
                output_size=request.perception.crop_output_size,
                bbox_scale=request.perception.crop_bbox_scale,
            ),
            min_valid_fraction=request.perception.crop_min_valid_fraction,
        )
        if request.perception.pose_input == "virtual_perspective_kb4_v1"
        else None
    )
    candidate_policy = CandidatePolicy(
        seed_threshold=request.thresholds.bbox_score,
        recovery_threshold=request.perception.recovery_bbox_score,
        max_candidates=request.perception.max_candidates_per_view,
    )
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
    frame_state = FrameExecutionState()
    overlay_video_output_count = 0
    tracker = SequenceTracker(
        max_root_distance_m=request.tracking.max_root_distance_m,
        max_gap_ms=request.tracking.max_gap_ms,
    )
    temporal_refiner = CausalTemporalRefiner(
        time_constant_ms=request.temporal.time_constant_ms,
        gap_reset_ms=request.temporal.gap_reset_ms,
    )
    mano_fitter: ManoTrackFitter | None = None
    overlay_video: RawVsStableOverlayVideo | None = None
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
            "input_space": request.perception.pose_input,
        },
        "fusion_method": FUSION_METHOD_ID,
        "model_keypoint_score_semantics": MODEL_SCORE_SEMANTICS,
        "keypoint_quality_weight_method": QUALITY_WEIGHT_METHOD,
        "keypoint_quality_weight_status": QUALITY_WEIGHT_STATUS,
        "kinematic_method": (
            "mano_v1.2_full45_robust_weighted_v3" if request.mano is not None else "NONE"
        ),
        "temporal_method": request.temporal.method,
    }
    try:
        if request.artifacts.overlay_video:
            overlay_video = RawVsStableOverlayVideo(
                output_path=destination / ".raw-vs-stable-stereo-rectified.mp4",
                image_size=rectification.output_size,
                timestamps_ns=_selected_pair_timestamps(
                    parts=parts,
                    pairs_by_part=pairs_by_part,
                    max_pairs=request.session.max_pairs,
                ),
                temporal_method=request.temporal.method,
            )
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
            mano_fitter = ManoTrackFitter(
                runtime=runtime,
                models=mano_models,
                cold_start_seeds={"mano_mean": None},
                rmse_gate_m=request.mano.max_fit_rmse_m,
                max_gap_ms=request.tracking.max_gap_ms,
                device=request.models.device,
                iterations=request.mano.iterations,
                learning_rate=request.mano.learning_rate,
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

        executor = FrameExecutor(
            FrameExecutionContext(
                request=request,
                runtime=runtime,
                models=models,
                rectification=rectification,
                writer=writer,
                candidate_policy=candidate_policy,
                virtual_pose_adapter=virtual_pose_adapter,
                tracker=tracker,
                temporal_refiner=temporal_refiner,
                mano_models=mano_models,
                mano_fitter=mano_fitter,
                backend_provenance=backend_provenance,
                overlay_video=overlay_video,
            ),
            state=frame_state,
        )
        for part in parts:
            remaining = request.session.max_pairs - frame_state.processed_pairs
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
                executor.execute(
                    FrameInput(
                        part_number=part.part_number,
                        global_index=frame_state.processed_pairs,
                        pair=pair,
                        left_frame=left_frames[pair.left_index],
                        right_frame=right_frames[pair.right_index],
                    )
                )
        if frame_state.processed_pairs == 0:
            raise WorkerError("worker processed no synchronized pairs")
        if overlay_video is not None:
            frame_state.active_stage = "EXPORT"
            overlay_result = overlay_video.close()
            overlay_path = Path(overlay_result["path"])
            try:
                overlay_blob = writer.put_blob_file(
                    overlay_path,
                    role="overlay_video_raw_vs_stable_stereo_rectified",
                    media_type="video/mp4",
                    suffix=".mp4",
                )
            finally:
                overlay_path.unlink(missing_ok=True)
            timeline_blob = _json_blob(
                writer,
                overlay_result["timeline"],
                role="overlay_video_timeline",
            )
            metadata = overlay_result["metadata"]
            if not isinstance(metadata, dict):
                raise WorkerError("overlay video metadata is invalid")
            writer.append(
                event_id="overlay:raw-vs-stable:export",
                stage="EXPORT",
                status="SUCCEEDED",
                event="raw_vs_stable_overlay_video_exported",
                payload={
                    **metadata,
                    "output_status": "PRODUCED",
                    "calibration_id": rectification.calibration_id,
                    "temporal_method": request.temporal.method,
                },
                blobs=[overlay_blob, timeline_blob],
                parent_event_ids=tuple(frame_state.overlay_parent_event_ids),
            )
            overlay_video_output_count = 1
        summary = writer.finalize(
            status="COMPLETED",
            summary={
                "pair_count": frame_state.processed_pairs,
                "matched_hand_count": frame_state.matched_hands,
                "valid_landmark_count": frame_state.valid_landmarks,
                "mano_output_count": frame_state.mano_outputs,
                "temporal_output_count": frame_state.temporal_outputs,
                "overlay_video_output_count": overlay_video_output_count,
                "export_count": frame_state.export_count,
                "output_status": ("PRODUCED" if frame_state.export_count else "NOT_PRODUCED"),
                "output_file": "fhp21.jsonl" if frame_state.export_count else None,
            },
        )
    except BaseException as exc:
        if overlay_video is not None:
            try:
                overlay_video.abort()
            except BaseException:
                pass
        try:
            writer.append(
                event_id="system:failure",
                stage=frame_state.active_stage,
                status="FAILED",
                event="worker_execution_failed",
                payload={
                    "output_status": "NOT_PRODUCED",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "pair_count": frame_state.processed_pairs,
                    "export_count": frame_state.export_count,
                    "failed_stage": frame_state.active_stage,
                    "frame_id": frame_state.active_frame_id,
                },
            )
        except BaseException:
            pass
        try:
            writer.finalize(
                status="FAILED",
                summary={
                    "pair_count": frame_state.processed_pairs,
                    "matched_hand_count": frame_state.matched_hands,
                    "valid_landmark_count": frame_state.valid_landmarks,
                    "mano_output_count": frame_state.mano_outputs,
                    "temporal_output_count": frame_state.temporal_outputs,
                    "overlay_video_output_count": overlay_video_output_count,
                    "export_count": frame_state.export_count,
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
