"""Typed, frame-local execution and trace projection for the H20 worker.

The worker runner owns run lifecycle and process-level resources.  This module owns the
single-frame transaction: source evidence, per-view inference, cross-view fusion, tracking,
optional MANO refinement, temporal refinement, export, and the corresponding trace DAG.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from ._generated_project_contract import FHP21_SCHEMA_ID, MANO_FHP21_MAPPING_ID
from .artifacts import ResultWriter
from .calibration import RectifiedStereo, project_rectified_keypoints
from .candidates import CandidateBatch, CandidateDecision, CandidatePolicy
from .contracts import WorkerError, WorkerRequest
from .geometry import associate, normalize_instances, triangulate_match
from .mano_fitting import ROBUST_GATE_METHOD, ROBUST_GATE_STATUS, ManoTrackFitter
from .output_contract import build_pose_estimate
from .pose_adapter import VirtualCropPoseAdapter
from .session import FramePair
from .temporal import CausalTemporalRefiner
from .tracking import MOTION_METHOD_ID, PALM_ANCHOR_METHOD_ID, SequenceTracker
from .visualization import RawVsStableOverlayVideo

Side = Literal["left", "right"]


@dataclass(frozen=True)
class FrameInput:
    """One decoded, synchronized stereo pair ready for frame-local execution."""

    part_number: int
    global_index: int
    pair: FramePair
    left_frame: Any
    right_frame: Any


@dataclass(frozen=True)
class FrameOutcome:
    """Observable delta produced by one completed frame transaction."""

    frame_id: str
    matched_hand_count: int
    valid_landmark_count: int
    mano_output_count: int
    temporal_output_count: int
    export_count: int
    export_event_ids: tuple[str, ...]


@dataclass
class FrameExecutionState:
    """Cross-frame state and cumulative counters owned by ``FrameExecutor``."""

    active_stage: str = "SYSTEM"
    active_frame_id: str | None = None
    processed_pairs: int = 0
    matched_hands: int = 0
    valid_landmarks: int = 0
    mano_outputs: int = 0
    temporal_outputs: int = 0
    export_count: int = 0
    tracking_events: dict[str, tuple[str, int]] = field(default_factory=dict)
    mano_events: dict[str, str] = field(default_factory=dict)
    temporal_events: dict[str, str] = field(default_factory=dict)
    overlay_parent_event_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FrameExecutionContext:
    """Stable dependencies behind the narrow per-frame execution interface."""

    request: WorkerRequest
    runtime: Any
    models: Any
    rectification: RectifiedStereo
    writer: ResultWriter
    candidate_policy: CandidatePolicy
    virtual_pose_adapter: VirtualCropPoseAdapter | None
    tracker: SequenceTracker
    temporal_refiner: CausalTemporalRefiner
    mano_models: Any | None
    mano_fitter: ManoTrackFitter | None
    backend_provenance: dict[str, Any]
    overlay_video: RawVsStableOverlayVideo | None


@dataclass(frozen=True)
class _RejectedObservation:
    event_id: str
    match_id: str
    observation_id: str


@dataclass(frozen=True)
class _PreparedAssignment:
    observation: dict[str, Any]
    assignment: dict[str, Any]
    raw: dict[str, Any]
    raw_event_id: str


@dataclass(frozen=True)
class _FusionBatch:
    observations: tuple[dict[str, Any], ...]
    rejected: tuple[_RejectedObservation, ...]
    empty_payload: dict[str, Any] | None
    empty_raw_event_id: str | None


@dataclass(frozen=True)
class _TrackingBatch:
    assignments: tuple[dict[str, Any], ...]
    prepared: tuple[_PreparedAssignment, ...]
    tracking_event_id: str
    matched_hand_count: int
    valid_landmark_count: int


def _should_save_source(policy: Any, global_pair_index: int) -> bool:
    return policy.source_frames == "ALL" or (
        policy.source_frames == "SAMPLED" and global_pair_index % policy.sample_every == 0
    )


def _json_blob(writer: ResultWriter, value: Any, *, role: str) -> dict[str, Any]:
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


def _projection_fields(
    rectification: RectifiedStereo,
    landmarks_xyz_m: Any = None,
    validity: Any = None,
) -> dict[str, Any]:
    projected = (
        {"left": [None] * 21, "right": [None] * 21}
        if landmarks_xyz_m is None or validity is None
        else project_rectified_keypoints(rectification, landmarks_xyz_m, validity)
    )
    return {
        "projected_keypoints_space": "rectified",
        "projected_keypoints_uv": projected,
    }


def _detection(instance: dict[str, Any]) -> dict[str, Any]:
    value = {
        "candidate_id": instance["candidate_id"],
        "bbox_xyxy": instance["bbox_xyxy"],
        "score": instance["bbox_score"],
        "bbox_score": instance["bbox_score"],
        "label": instance["label"],
    }
    for name in (
        "source_index",
        "classification",
        "reason",
        "eligible_for_association",
        "final_selection",
    ):
        if name in instance:
            value[name] = instance[name]
    return value


def _candidate_payload(decision: CandidateDecision) -> dict[str, Any]:
    return {
        "candidate_id": decision.candidate_id,
        "source_index": decision.source_index,
        "bbox_xyxy": list(decision.bbox_xyxy),
        "score": decision.score,
        "bbox_score": decision.score,
        "label": decision.label,
        "classification": decision.classification,
        "reason": decision.reason,
        "eligible_for_association": decision.eligible_for_association,
        "final_selection": decision.final_selection,
    }


def _legacy_candidate_batch(
    raw_instances: Any,
    *,
    policy: CandidatePolicy,
    category_id: int,
    view_id: str,
) -> CandidateBatch:
    if not isinstance(raw_instances, list):
        raise WorkerError("legacy runtime inference must return a list")
    return policy.classify(
        bboxes=[instance.get("bbox_xyxy") for instance in raw_instances],
        scores=[instance.get("bbox_score") for instance in raw_instances],
        labels=[instance.get("label") for instance in raw_instances],
        category_id=category_id,
        view_id=view_id,
    )


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
    for name, length in fields.items():
        item = value.get(name)
        if not isinstance(item, list) or len(item) != length:
            raise WorkerError(f"MANO runtime {name} must contain {length} values")
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
    for name in ("global_orient", "hand_pose", "transl", "beta"):
        if any(
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            for number in value[name]
        ):
            raise WorkerError(f"MANO runtime {name} contains a non-finite value")
    rmse = value.get("rmse_m")
    if (
        isinstance(rmse, bool)
        or not isinstance(rmse, (int, float))
        or not math.isfinite(float(rmse))
        or float(rmse) < 0
    ):
        raise WorkerError("MANO runtime rmse_m is invalid")
    return value


def _mano_selection_payload(
    decision: dict[str, Any],
    *,
    valid_landmark_count: int,
) -> dict[str, Any]:
    status = decision["status"]
    selected = decision["fit"]
    if status == "ACCEPTED":
        selection_decision = "SELECTED"
    elif status == "REJECTED":
        selection_decision = "NO_HIGH_QUALITY_FIT"
    else:
        selection_decision = "OPTIMIZER_ERROR"
    return {
        "decision": selection_decision,
        "status": status,
        "valid_landmark_count": valid_landmark_count,
        "init_source": decision["init_source"],
        "predecessor_timestamp_ns": decision["predecessor_timestamp_ns"],
        "reset_reason": decision["reset_reason"],
        "selected_attempt_index": decision["selected_attempt_index"],
        "selected_side": None if selected is None else selected["side"],
        "gate": decision.get("gate"),
        "first_high_quality_frame": status == "ACCEPTED"
        and decision["init_source"] == "COLD_START",
        "best_attempt": decision["best_attempt"],
        "attempts": decision["attempts"],
    }


def _fit_mano_frame(
    *,
    fitter: ManoTrackFitter,
    request: Any,
    track_id: str,
    timestamp_ns: int,
    raw: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    valid_count = raw["valid_landmark_count"]
    if valid_count < request.min_valid_landmarks:
        return None, {
            "decision": "INSUFFICIENT_VALID_LANDMARKS",
            "valid_landmark_count": valid_count,
            "required_valid_landmark_count": request.min_valid_landmarks,
            "attempts": [],
        }
    decision = fitter.fit_frame(
        track_id=track_id,
        target_xyz_m=raw["landmarks_xyz_m"],
        validity=raw["validity"],
        timestamp_ns=timestamp_ns,
    )
    selected = decision["fit"]
    if selected is not None:
        selected = _validate_mano_fit(selected, expected_side=selected["side"])
    return selected, _mano_selection_payload(
        decision,
        valid_landmark_count=valid_count,
    )


class FrameExecutor:
    """Execute one stereo frame transaction behind a narrow typed interface."""

    def __init__(
        self,
        context: FrameExecutionContext,
        *,
        state: FrameExecutionState | None = None,
    ) -> None:
        if not isinstance(context, FrameExecutionContext):
            raise TypeError("context must be a FrameExecutionContext")
        self.context = context
        self.state = FrameExecutionState() if state is None else state

    def execute(self, frame: FrameInput) -> FrameOutcome:
        """Compute and persist exactly one frame, returning its observable delta."""

        if not isinstance(frame, FrameInput):
            raise TypeError("frame must be a FrameInput")
        frame_id, event_prefix = self._frame_ids(frame)
        self.state.active_frame_id = frame_id
        save_source, rendered_views = self._record_input_evidence(
            frame,
            frame_id=frame_id,
            event_prefix=event_prefix,
        )
        views = {
            side: self._infer_view(
                frame,
                side=side,
                image=image,
                frame_id=frame_id,
                event_prefix=event_prefix,
                save_source=save_source,
            )
            for side, image in (
                ("left", frame.left_frame),
                ("right", frame.right_frame),
            )
        }
        association = self._record_association(
            frame,
            frame_id=frame_id,
            event_prefix=event_prefix,
            views=views,
        )
        fusion = self._fuse_matches(
            frame,
            frame_id=frame_id,
            event_prefix=event_prefix,
            views=views,
            association=association,
        )
        tracking = self._assign_tracks(
            frame,
            frame_id=frame_id,
            event_prefix=event_prefix,
            fusion=fusion,
        )
        export_event_ids = self._record_absent_downstream(
            frame,
            frame_id=frame_id,
            event_prefix=event_prefix,
            fusion=fusion,
            tracking_event_id=tracking.tracking_event_id,
        )
        mano_before = self.state.mano_outputs
        temporal_before = self.state.temporal_outputs
        export_before = self.state.export_count
        overlay_tracks: list[dict[str, Any]] = []
        for prepared in tracking.prepared:
            hand_export_ids, overlay_track = self._execute_hand(
                frame,
                frame_id=frame_id,
                event_prefix=event_prefix,
                views=views,
                prepared=prepared,
                tracking_event_id=tracking.tracking_event_id,
            )
            export_event_ids.extend(hand_export_ids)
            overlay_tracks.append(overlay_track)
        self._append_overlay_frame(
            frame,
            frame_id=frame_id,
            rendered_views=rendered_views,
            tracks=overlay_tracks,
            export_event_ids=export_event_ids,
        )
        self.state.processed_pairs += 1
        return FrameOutcome(
            frame_id=frame_id,
            matched_hand_count=tracking.matched_hand_count,
            valid_landmark_count=tracking.valid_landmark_count,
            mano_output_count=self.state.mano_outputs - mano_before,
            temporal_output_count=self.state.temporal_outputs - temporal_before,
            export_count=self.state.export_count - export_before,
            export_event_ids=tuple(export_event_ids),
        )

    @staticmethod
    def _frame_ids(frame: FrameInput) -> tuple[str, str]:
        frame_id = f"part{frame.part_number:04d}/pair{frame.pair.pair_index:06d}"
        event_prefix = f"part{frame.part_number:04d}:pair{frame.pair.pair_index:06d}"
        return frame_id, event_prefix

    def _record_input_evidence(
        self,
        frame: FrameInput,
        *,
        frame_id: str,
        event_prefix: str,
    ) -> tuple[bool, dict[str, dict[str, Any]] | None]:
        context = self.context
        request = context.request
        writer = context.writer
        runtime = context.runtime
        rectification = context.rectification
        source_blobs: list[dict[str, Any]] = []
        save_source = _should_save_source(request.artifacts, frame.global_index)
        if save_source:
            suffix = f".{request.artifacts.image_format}"
            media_type = "image/jpeg" if request.artifacts.image_format == "jpg" else "image/png"
            for side, image in (("left", frame.left_frame), ("right", frame.right_frame)):
                source_blobs.append(
                    writer.put_blob(
                        runtime.encode_frame(image, request.artifacts.image_format),
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
                "frame_index": frame.global_index,
                **frame.pair.to_dict(),
            },
            blobs=source_blobs,
            parent_event_ids=("calibration:rectification",),
        )
        rendered_views: dict[str, dict[str, Any]] | None = None
        if save_source or context.overlay_video is not None:
            self.state.active_stage = "RECTIFICATION"
            rendered_views = {
                side: runtime.render_rectification(rectification, side, image)
                for side, image in (
                    ("left", frame.left_frame),
                    ("right", frame.right_frame),
                )
            }
        if save_source:
            assert rendered_views is not None
            rendered_blobs: list[dict[str, Any]] = []
            suffix = f".{request.artifacts.image_format}"
            media_type = "image/jpeg" if request.artifacts.image_format == "jpg" else "image/png"
            for side in ("left", "right"):
                for image_space in ("undistorted", "rectified"):
                    rendered_blobs.append(
                        writer.put_blob(
                            runtime.encode_frame(
                                rendered_views[side][image_space],
                                request.artifacts.image_format,
                            ),
                            role=f"{image_space}_{side}",
                            media_type=media_type,
                            suffix=suffix,
                        )
                    )
            writer.append(
                event_id=f"{event_prefix}:rectification",
                stage="RECTIFICATION",
                status="SUCCEEDED",
                event="stereo_pair_rectification_rendered",
                payload={
                    "frame_id": frame_id,
                    "frame_index": frame.global_index,
                    "timestamp_ns": frame.pair.pair_timestamp_ns,
                    "calibration_id": rectification.calibration_id,
                    "output_status": "PRODUCED",
                    "image_width": rectification.output_size[0],
                    "image_height": rectification.output_size[1],
                },
                blobs=rendered_blobs,
                parent_event_ids=(f"{event_prefix}:sync",),
            )
        return save_source, rendered_views

    def _infer_view(
        self,
        frame: FrameInput,
        *,
        side: Side,
        image: Any,
        frame_id: str,
        event_prefix: str,
        save_source: bool,
    ) -> list[dict[str, Any]]:
        context = self.context
        request = context.request
        runtime = context.runtime
        writer = context.writer
        rectification = context.rectification
        self.state.active_stage = "DETECTION"
        crop_results: tuple[Any, ...] = ()
        if hasattr(runtime, "detect_candidates"):
            candidate_batch = runtime.detect_candidates(
                context.models,
                image,
                policy=context.candidate_policy,
                category_id=request.models.detector_category_id,
                view_id=side,
            )
            if not isinstance(candidate_batch, CandidateBatch):
                raise WorkerError("runtime detect_candidates must return a CandidateBatch")
            pool_detections = [
                _candidate_payload(decision) for decision in candidate_batch.candidate_pool
            ]
            legacy_pose_instances = None
        elif context.virtual_pose_adapter is None:
            legacy_pose_instances = runtime.infer(
                context.models,
                image,
                bbox_threshold=request.perception.recovery_bbox_score,
                category_id=request.models.detector_category_id,
                max_instances=request.perception.max_candidates_per_view,
            )
            candidate_batch = _legacy_candidate_batch(
                legacy_pose_instances,
                policy=context.candidate_policy,
                category_id=request.models.detector_category_id,
                view_id=side,
            )
            pool_detections = [
                _candidate_payload(decision) for decision in candidate_batch.candidate_pool
            ]
        else:
            legacy_pose_instances = None
            detector_candidates = runtime.detect(
                context.models,
                image,
                bbox_threshold=request.perception.recovery_bbox_score,
                category_id=request.models.detector_category_id,
                max_instances=request.perception.max_candidates_per_view,
            )
            candidate_batch = _legacy_candidate_batch(
                detector_candidates,
                policy=context.candidate_policy,
                category_id=request.models.detector_category_id,
                view_id=side,
            )
            pool_detections = [
                _candidate_payload(decision) for decision in candidate_batch.candidate_pool
            ]
        candidate_decisions = [
            _candidate_payload(decision) for decision in candidate_batch.decisions
        ]
        detected_candidates = [_detection(detection) for detection in pool_detections]
        common = {
            "frame_id": frame_id,
            "frame_index": frame.global_index,
            "timestamp_ns": frame.pair.pair_timestamp_ns,
            "view_id": side,
            "image_width": rectification.image_size[0],
            "image_height": rectification.image_size[1],
        }
        writer.append(
            event_id=f"{event_prefix}:detection:{side}",
            stage="DETECTION",
            status="SUCCEEDED" if detected_candidates else "WARNING",
            event="hand_candidates_detected",
            payload={
                **common,
                "output_status": "PRODUCED" if detected_candidates else "NOT_PRODUCED",
                "detections": detected_candidates,
                "instances": detected_candidates,
                "candidate_decisions": candidate_decisions,
                "candidate_pool": pool_detections,
            },
            parent_event_ids=(f"{event_prefix}:sync",),
        )
        self.state.active_stage = "POSE_2D"
        if context.virtual_pose_adapter is None:
            if legacy_pose_instances is None:
                pose_results = runtime.infer_pose(
                    context.models,
                    image,
                    bboxes=[item["bbox_xyxy"] for item in pool_detections],
                )
                raw_instances = [
                    {**detection, **pose}
                    for detection, pose in zip(pool_detections, pose_results, strict=True)
                ]
            else:
                raw_instances = [
                    {
                        **pool_detections[pool_index],
                        **legacy_pose_instances[decision.source_index],
                        **pool_detections[pool_index],
                    }
                    for pool_index, decision in enumerate(candidate_batch.candidate_pool)
                ]
            instances = normalize_instances(
                raw_instances,
                side=side,
                rectification=rectification,
            )
        else:
            crop_batch = context.virtual_pose_adapter.infer(
                runtime=runtime,
                models=context.models,
                frame=image,
                side=side,
                detections=pool_detections,
                rectification=rectification,
            )
            crop_results = crop_batch.results
            instances = normalize_instances(
                list(crop_batch.produced_instances),
                side=side,
                rectification=rectification,
            )
        crop_event_ids: list[str] = []
        for crop_index, crop_result in enumerate(crop_results):
            crop_event_id = f"{event_prefix}:crop:{side}:{crop_index}"
            crop_blobs: list[dict[str, Any]] = []
            if save_source and crop_result.crop is not None:
                suffix = f".{request.artifacts.image_format}"
                media_type = (
                    "image/jpeg" if request.artifacts.image_format == "jpg" else "image/png"
                )
                crop_blobs.extend(
                    (
                        writer.put_blob(
                            runtime.encode_frame(
                                crop_result.crop.image,
                                request.artifacts.image_format,
                            ),
                            role="virtual_crop",
                            media_type=media_type,
                            suffix=suffix,
                        ),
                        writer.put_blob(
                            runtime.encode_frame(
                                crop_result.crop.valid_mask.astype("uint8") * 255,
                                "png",
                            ),
                            role="virtual_crop_valid_mask",
                            media_type="image/png",
                            suffix=".png",
                        ),
                    )
                )
            produced = crop_result.status == "PRODUCED"
            writer.append(
                event_id=crop_event_id,
                stage="POSE_2D",
                status="SUCCEEDED" if produced else "WARNING",
                event=(
                    "virtual_crop_pose_inferred" if produced else "virtual_crop_pose_not_produced"
                ),
                payload={**common, **crop_result.trace_payload()},
                blobs=crop_blobs,
                parent_event_ids=(f"{event_prefix}:detection:{side}",),
            )
            crop_event_ids.append(crop_event_id)
        writer.append(
            event_id=f"{event_prefix}:pose2d:{side}",
            stage="POSE_2D",
            status="SUCCEEDED" if instances else "WARNING",
            event="view_keypoints_inferred",
            payload={
                **common,
                "landmark_schema": FHP21_SCHEMA_ID,
                "output_status": "PRODUCED" if instances else "NOT_PRODUCED",
                "instances": instances,
            },
            parent_event_ids=(
                tuple(crop_event_ids) if crop_event_ids else (f"{event_prefix}:detection:{side}",)
            ),
        )
        return instances

    def _record_association(
        self,
        frame: FrameInput,
        *,
        frame_id: str,
        event_prefix: str,
        views: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        self.state.active_stage = "CROSS_VIEW_ASSOCIATION"
        association = associate(
            views["left"],
            views["right"],
            self.context.request.thresholds,
        )
        self.context.writer.append(
            event_id=f"{event_prefix}:association",
            stage="CROSS_VIEW_ASSOCIATION",
            status="SUCCEEDED" if association["matches"] else "WARNING",
            event="cross_view_hands_associated",
            payload={
                "frame_id": frame_id,
                "frame_index": frame.global_index,
                "timestamp_ns": frame.pair.pair_timestamp_ns,
                "output_status": "PRODUCED" if association["matches"] else "NOT_PRODUCED",
                **association,
            },
            parent_event_ids=(
                f"{event_prefix}:pose2d:left",
                f"{event_prefix}:pose2d:right",
            ),
        )
        return association

    def _fuse_matches(
        self,
        frame: FrameInput,
        *,
        frame_id: str,
        event_prefix: str,
        views: dict[str, list[dict[str, Any]]],
        association: dict[str, Any],
    ) -> _FusionBatch:
        context = self.context
        observations: list[dict[str, Any]] = []
        rejected: list[_RejectedObservation] = []
        for match in association["matches"]:
            self.state.active_stage = "RAW_FUSION"
            raw = triangulate_match(
                views["left"][match["left_index"]],
                views["right"][match["right_index"]],
                rectification=context.rectification,
                thresholds=context.request.thresholds,
            )
            observation = {
                "observation_id": f"{event_prefix}:{match['match_id']}",
                "match": match,
                **raw,
            }
            if raw["hand_validity"] == "VALID":
                observations.append(observation)
                continue
            rejected_event_id = f"{event_prefix}:raw-gate:{match['match_id']}"
            rejected_payload = {
                "frame_id": frame_id,
                "frame_index": frame.global_index,
                "timestamp_ns": frame.pair.pair_timestamp_ns,
                "track_id": None,
                "observation_id": observation["observation_id"],
                "match": match,
                "output_status": "NOT_PRODUCED",
                "projected_keypoints_space": "rectified",
                "projected_keypoints_uv": project_rectified_keypoints(
                    context.rectification,
                    raw["landmarks_xyz_m"],
                    raw["validity"],
                ),
                **raw,
            }
            context.writer.append(
                event_id=rejected_event_id,
                stage="RAW_FUSION",
                status="WARNING",
                event="raw_hand_gate_not_produced",
                payload=rejected_payload,
                blobs=[
                    _json_blob(
                        context.writer,
                        rejected_payload,
                        role="raw_fhp21_rejected_hand",
                    )
                ],
                parent_event_ids=(f"{event_prefix}:association",),
            )
            rejected.append(
                _RejectedObservation(
                    event_id=rejected_event_id,
                    match_id=match["match_id"],
                    observation_id=observation["observation_id"],
                )
            )
        empty_payload: dict[str, Any] | None = None
        empty_raw_event_id: str | None = None
        if not observations and not rejected:
            empty_payload = {
                "frame_id": frame_id,
                "frame_index": frame.global_index,
                "timestamp_ns": frame.pair.pair_timestamp_ns,
                "track_id": None,
                "output_status": "NOT_PRODUCED",
                "reason": ("NO_FUSABLE_HAND" if association["matches"] else "NO_CROSS_VIEW_MATCH"),
                **_projection_fields(context.rectification),
            }
            empty_raw_event_id = f"{event_prefix}:raw:none"
            context.writer.append(
                event_id=empty_raw_event_id,
                stage="RAW_FUSION",
                status="WARNING",
                event="raw_landmarks_not_produced",
                payload=empty_payload,
                parent_event_ids=(f"{event_prefix}:association",),
            )
        return _FusionBatch(
            observations=tuple(observations),
            rejected=tuple(rejected),
            empty_payload=empty_payload,
            empty_raw_event_id=empty_raw_event_id,
        )

    def _assign_tracks(
        self,
        frame: FrameInput,
        *,
        frame_id: str,
        event_prefix: str,
        fusion: _FusionBatch,
    ) -> _TrackingBatch:
        context = self.context
        timestamp_ns = frame.pair.pair_timestamp_ns
        self.state.active_stage = "CROSS_VIEW_ASSOCIATION"
        assignments = context.tracker.assign(
            list(fusion.observations),
            timestamp_ns=timestamp_ns,
        )
        prepared: list[_PreparedAssignment] = []
        accepted_raw_event_ids: list[str] = []
        matched_hand_count = 0
        valid_landmark_count = 0
        for observation, assignment in zip(fusion.observations, assignments, strict=True):
            match = observation["match"]
            track_id = assignment["track_id"]
            raw = {
                key: value
                for key, value in observation.items()
                if key not in {"observation_id", "match"}
            }
            self.state.matched_hands += 1
            self.state.valid_landmarks += raw["valid_landmark_count"]
            matched_hand_count += 1
            valid_landmark_count += raw["valid_landmark_count"]
            raw_projection = project_rectified_keypoints(
                context.rectification,
                raw["landmarks_xyz_m"],
                raw["validity"],
            )
            raw_payload = {
                "frame_id": frame_id,
                "frame_index": frame.global_index,
                "timestamp_ns": timestamp_ns,
                "track_id": track_id,
                "observation_id": observation["observation_id"],
                "match": match,
                "track_assignment": assignment,
                "output_status": ("PRODUCED" if raw["valid_landmark_count"] else "NOT_PRODUCED"),
                "projected_keypoints_space": "rectified",
                "projected_keypoints_uv": raw_projection,
                **raw,
            }
            raw_event_id = f"{event_prefix}:raw:{match['match_id']}"
            context.writer.append(
                event_id=raw_event_id,
                stage="RAW_FUSION",
                status="SUCCEEDED" if raw["valid_landmark_count"] else "WARNING",
                event="raw_landmarks_triangulated",
                payload=raw_payload,
                blobs=[_json_blob(context.writer, raw_payload, role="raw_fhp21")],
                parent_event_ids=(f"{event_prefix}:association",),
            )
            accepted_raw_event_ids.append(raw_event_id)
            prepared.append(
                _PreparedAssignment(
                    observation=observation,
                    assignment=assignment,
                    raw=raw,
                    raw_event_id=raw_event_id,
                )
            )
        tracking_max_gap_ns = int(context.request.tracking.max_gap_ms * 1_000_000)
        active_tracking_states = {
            track_id: (event_id, state_timestamp_ns)
            for track_id, (event_id, state_timestamp_ns) in self.state.tracking_events.items()
            if 0 <= timestamp_ns - state_timestamp_ns <= tracking_max_gap_ns
        }
        tracking_predecessors = sorted(
            {event_id for event_id, _state_timestamp_ns in active_tracking_states.values()}
        )
        tracking_event_id = f"{event_prefix}:tracking"
        context.writer.append(
            event_id=tracking_event_id,
            stage="CROSS_VIEW_ASSOCIATION",
            status="SUCCEEDED" if assignments else "WARNING",
            event="sequence_tracks_assigned",
            payload={
                "frame_id": frame_id,
                "frame_index": frame.global_index,
                "timestamp_ns": timestamp_ns,
                "method": MOTION_METHOD_ID,
                "anchor_method": PALM_ANCHOR_METHOD_ID,
                "output_status": "PRODUCED" if assignments else "NOT_PRODUCED",
                "state_predecessor_event_ids": tracking_predecessors,
                "assignments": assignments,
            },
            parent_event_ids=(
                *(
                    accepted_raw_event_ids
                    + [record.event_id for record in fusion.rejected]
                    + ([] if fusion.empty_raw_event_id is None else [fusion.empty_raw_event_id])
                    or [f"{event_prefix}:association"]
                ),
                *tracking_predecessors,
            ),
        )
        for track_id, (_event_id, last_seen_timestamp_ns) in active_tracking_states.items():
            self.state.tracking_events[track_id] = (
                tracking_event_id,
                last_seen_timestamp_ns,
            )
        for assignment in assignments:
            self.state.tracking_events[assignment["track_id"]] = (
                tracking_event_id,
                timestamp_ns,
            )
        return _TrackingBatch(
            assignments=tuple(assignments),
            prepared=tuple(prepared),
            tracking_event_id=tracking_event_id,
            matched_hand_count=matched_hand_count,
            valid_landmark_count=valid_landmark_count,
        )

    def _record_absent_downstream(
        self,
        frame: FrameInput,
        *,
        frame_id: str,
        event_prefix: str,
        fusion: _FusionBatch,
        tracking_event_id: str,
    ) -> list[str]:
        writer = self.context.writer
        rectification = self.context.rectification
        timestamp_ns = frame.pair.pair_timestamp_ns
        export_event_ids: list[str] = []
        for rejected in fusion.rejected:
            rejected_base_payload = {
                "frame_id": frame_id,
                "frame_index": frame.global_index,
                "timestamp_ns": timestamp_ns,
                "track_id": None,
                "source_observation_ids": [rejected.observation_id],
                "output_status": "NOT_PRODUCED",
                "reason": "RAW_HAND_GATE_REJECTED",
                **_projection_fields(rectification),
            }
            rejected_mano_event_id = f"{event_prefix}:mano-rejected:{rejected.match_id}"
            rejected_temporal_event_id = f"{event_prefix}:temporal-rejected:{rejected.match_id}"
            rejected_export_event_id = f"{event_prefix}:export-rejected:{rejected.match_id}"
            writer.append(
                event_id=rejected_mano_event_id,
                stage="KINEMATIC_REFINEMENT",
                status="SKIPPED",
                event="kinematic_refinement_not_produced",
                payload=rejected_base_payload,
                parent_event_ids=(tracking_event_id,),
            )
            writer.append(
                event_id=rejected_temporal_event_id,
                stage="TEMPORAL_REFINEMENT",
                status="SKIPPED",
                event="temporal_refinement_not_produced",
                payload=rejected_base_payload,
                parent_event_ids=(rejected_mano_event_id,),
            )
            writer.append(
                event_id=rejected_export_event_id,
                stage="EXPORT",
                status="SKIPPED",
                event="fhp21_record_not_produced",
                payload={**rejected_base_payload, "output_file": None},
                parent_event_ids=(rejected_temporal_event_id,),
            )
            export_event_ids.append(rejected_export_event_id)
        if fusion.empty_raw_event_id is not None:
            assert fusion.empty_payload is not None
            writer.append(
                event_id=f"{event_prefix}:mano:none",
                stage="KINEMATIC_REFINEMENT",
                status="SKIPPED",
                event="kinematic_refinement_not_produced",
                payload={
                    **fusion.empty_payload,
                    "reason": "NO_RAW_FUSION_OBSERVATION",
                },
                parent_event_ids=(tracking_event_id,),
            )
            writer.append(
                event_id=f"{event_prefix}:temporal:none",
                stage="TEMPORAL_REFINEMENT",
                status="SKIPPED",
                event="temporal_refinement_not_produced",
                payload={
                    **fusion.empty_payload,
                    "reason": "NO_KINEMATIC_OR_RAW_OBSERVATION",
                },
                parent_event_ids=(f"{event_prefix}:mano:none",),
            )
            writer.append(
                event_id=f"{event_prefix}:export:none",
                stage="EXPORT",
                status="SKIPPED",
                event="fhp21_record_not_produced",
                payload={
                    **fusion.empty_payload,
                    "reason": "NO_TEMPORAL_POSE_ESTIMATE",
                    "output_file": None,
                },
                parent_event_ids=(f"{event_prefix}:temporal:none",),
            )
            export_event_ids.append(f"{event_prefix}:export:none")
        return export_event_ids

    def _execute_hand(
        self,
        frame: FrameInput,
        *,
        frame_id: str,
        event_prefix: str,
        views: dict[str, list[dict[str, Any]]],
        prepared: _PreparedAssignment,
        tracking_event_id: str,
    ) -> tuple[list[str], dict[str, Any]]:
        context = self.context
        request = context.request
        writer = context.writer
        rectification = context.rectification
        observation = prepared.observation
        assignment = prepared.assignment
        raw = prepared.raw
        match = observation["match"]
        track_id = assignment["track_id"]
        timestamp_ns = frame.pair.pair_timestamp_ns

        for side, candidate_index in (
            ("left", match["left_index"]),
            ("right", match["right_index"]),
        ):
            instance = views[side][candidate_index]
            writer.append(
                event_id=(f"{event_prefix}:pose2d:tracked:{match['match_id']}:{side}"),
                stage="POSE_2D",
                status="SUCCEEDED",
                event="tracked_view_keypoints_recorded",
                payload={
                    "frame_id": frame_id,
                    "frame_index": frame.global_index,
                    "timestamp_ns": timestamp_ns,
                    "view_id": side,
                    "track_id": track_id,
                    "candidate_id": instance["candidate_id"],
                    "landmark_schema": FHP21_SCHEMA_ID,
                    "output_status": "PRODUCED",
                    "detections": [_detection(instance)],
                    "keypoints_uv": instance["keypoints_uv"],
                    "keypoints_uv_rectified": instance["keypoints_uv_rectified"],
                    "keypoint_scores": instance["keypoint_scores"],
                    "model_keypoint_scores": instance["model_keypoint_scores"],
                    "keypoint_score_semantics": instance["keypoint_score_semantics"],
                    "keypoint_quality_weight_method": instance["keypoint_quality_weight_method"],
                    "keypoint_quality_weight_status": instance["keypoint_quality_weight_status"],
                    "image_width": rectification.image_size[0],
                    "image_height": rectification.image_size[1],
                },
                parent_event_ids=(tracking_event_id,),
            )

        raw_projection = project_rectified_keypoints(
            rectification,
            raw["landmarks_xyz_m"],
            raw["validity"],
        )
        mano_fit: dict[str, Any] | None = None
        temporal_parent = prepared.raw_event_id
        mano_event_id = f"{event_prefix}:mano:{match['match_id']}"
        if request.mano is not None:
            self.state.active_stage = "KINEMATIC_REFINEMENT"
            assert context.mano_models is not None
            assert context.mano_fitter is not None
            mano_fit, selection = _fit_mano_frame(
                fitter=context.mano_fitter,
                request=request.mano,
                track_id=track_id,
                timestamp_ns=timestamp_ns,
                raw=raw,
            )
            mano_predecessor = self.state.mano_events.get(track_id)
            if mano_fit is None:
                mano_payload = {
                    "frame_id": frame_id,
                    "frame_index": frame.global_index,
                    "timestamp_ns": timestamp_ns,
                    "track_id": track_id,
                    "input_stage": "RAW_FUSION",
                    "output_status": "NOT_PRODUCED",
                    "state_predecessor_event_id": mano_predecessor,
                    "selection": selection,
                    "fit_quality": selection.get("gate"),
                    **_projection_fields(rectification),
                }
                writer.append(
                    event_id=mano_event_id,
                    stage="KINEMATIC_REFINEMENT",
                    status="WARNING",
                    event="mano_frame_not_produced",
                    payload=mano_payload,
                    parent_event_ids=(
                        (tracking_event_id,)
                        if mano_predecessor is None
                        else (tracking_event_id, mano_predecessor)
                    ),
                )
            else:
                self.state.mano_outputs += 1
                mano_projection = project_rectified_keypoints(
                    rectification,
                    mano_fit["landmarks_xyz_m"],
                    mano_fit["validity"],
                )
                mano_payload = {
                    "frame_id": frame_id,
                    "frame_index": frame.global_index,
                    "timestamp_ns": timestamp_ns,
                    "track_id": track_id,
                    "input_stage": "RAW_FUSION",
                    "output_status": "PRODUCED",
                    "state_predecessor_event_id": mano_predecessor,
                    "coordinate_frame": "rectified_left_camera",
                    "length_unit": "m",
                    "landmark_schema": FHP21_SCHEMA_ID,
                    "handedness": mano_fit["side"],
                    "selection": selection,
                    "fit_quality": selection["gate"],
                    "beta_frozen": not selection["first_high_quality_frame"],
                    "pose": mano_fit["hand_pose"],
                    "global_orient": mano_fit["global_orient"],
                    "transl": mano_fit["transl"],
                    "beta": mano_fit["beta"],
                    "mapping_id": mano_fit["mapping_id"],
                    "rmse_m": mano_fit["rmse_m"],
                    "raw_rmse_m": selection["gate"]["raw_rmse_m"],
                    "full_rmse_m": selection["gate"]["full_rmse_m"],
                    "weighted_rmse_m": selection["gate"]["weighted_rmse_m"],
                    "inlier_rmse_m": selection["gate"]["inlier_rmse_m"],
                    "joint_weights": selection["gate"]["joint_weights"],
                    "inlier_mask": selection["gate"]["inlier_mask"],
                    "effective_joint_count": selection["gate"]["effective_joint_count"],
                    "robust_gate_method": ROBUST_GATE_METHOD,
                    "robust_gate_status": ROBUST_GATE_STATUS,
                    "loss": {"metric": "RMSE_M", "value": mano_fit["rmse_m"]},
                    "optimizer": {
                        "iterations_run": mano_fit["iterations_run"],
                        "best_loss": mano_fit["best_loss"],
                        "final_loss": mano_fit["final_loss"],
                        "joint_residuals_m": mano_fit["joint_residuals_m"],
                        "converged": mano_fit["converged"],
                    },
                    "landmarks_xyz_m": mano_fit["landmarks_xyz_m"],
                    "validity": mano_fit["validity"],
                    "projected_keypoints_space": "rectified",
                    "projected_keypoints_uv": mano_projection,
                }
                writer.append(
                    event_id=mano_event_id,
                    stage="KINEMATIC_REFINEMENT",
                    status="SUCCEEDED",
                    event="mano_frame_fitted",
                    payload=mano_payload,
                    blobs=[_json_blob(writer, mano_payload, role="mano_fhp21")],
                    parent_event_ids=(
                        (tracking_event_id,)
                        if mano_predecessor is None
                        else (tracking_event_id, mano_predecessor)
                    ),
                )
                self.state.mano_events[track_id] = mano_event_id
            temporal_parent = mano_event_id
        else:
            writer.append(
                event_id=mano_event_id,
                stage="KINEMATIC_REFINEMENT",
                status="SKIPPED",
                event="mano_frame_not_configured",
                payload={
                    "frame_id": frame_id,
                    "frame_index": frame.global_index,
                    "timestamp_ns": timestamp_ns,
                    "track_id": track_id,
                    "input_stage": "RAW_FUSION",
                    "output_status": "NOT_PRODUCED",
                    "reason": "MANO_NOT_CONFIGURED",
                    "mapping_id": MANO_FHP21_MAPPING_ID,
                    **_projection_fields(rectification),
                },
                parent_event_ids=(tracking_event_id,),
            )
            temporal_parent = mano_event_id

        temporal_input = mano_fit if mano_fit is not None else raw
        temporal_input_stage = "KINEMATIC_REFINEMENT" if mano_fit is not None else "RAW_FUSION"
        self.state.active_stage = "TEMPORAL_REFINEMENT"
        temporal = context.temporal_refiner.refine(
            track_id=track_id,
            timestamp_ns=timestamp_ns,
            landmarks_xyz_m=temporal_input["landmarks_xyz_m"],
            validity=temporal_input["validity"],
            input_stage=temporal_input_stage,
        )
        temporal_produced = temporal["valid_landmark_count"] > 0
        if temporal_produced:
            self.state.active_stage = "EXPORT"
            self.state.temporal_outputs += 1
        temporal_projection = project_rectified_keypoints(
            rectification,
            temporal["landmarks_xyz_m"],
            temporal["validity"],
        )
        temporal_predecessor = self.state.temporal_events.get(track_id)
        temporal_payload = {
            "frame_id": frame_id,
            "frame_index": frame.global_index,
            "timestamp_ns": timestamp_ns,
            "track_id": track_id,
            "input_stage": temporal_input_stage,
            "state_predecessor_event_id": temporal_predecessor,
            "output_status": "PRODUCED" if temporal_produced else "NOT_PRODUCED",
            "coordinate_frame": "rectified_left_camera",
            "length_unit": "m",
            "landmark_schema": FHP21_SCHEMA_ID,
            "projected_keypoints_space": "rectified",
            "projected_keypoints_uv": temporal_projection,
            **temporal,
        }
        temporal_event_id = f"{event_prefix}:temporal:{match['match_id']}"
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
            blobs=[_json_blob(writer, temporal_payload, role="temporal_fhp21")],
            parent_event_ids=(
                (temporal_parent,)
                if temporal_predecessor is None
                else (temporal_parent, temporal_predecessor)
            ),
        )
        self.state.temporal_events[track_id] = temporal_event_id

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
                    "beta": mano_fit["beta"],
                    "rmse_m": mano_fit["rmse_m"],
                    "loss": {"metric": "RMSE_M", "value": mano_fit["rmse_m"]},
                    "landmarks_xyz_m": mano_fit["landmarks_xyz_m"],
                    "validity": mano_fit["validity"],
                }
            )
            export_value = build_pose_estimate(
                sequence_id=request.session.path.name,
                estimate_id=(f"{event_prefix}:{match['match_id']}:temporal-estimate"),
                frame_id=frame_id,
                frame_index=frame.global_index,
                timestamp_ns=timestamp_ns,
                track_id=track_id,
                source_observation_id=observation["observation_id"],
                calibration_id=rectification.calibration_id,
                raw=raw,
                mano=mano_export,
                temporal=temporal,
                keypoint_score_threshold=request.thresholds.keypoint_score,
                backend_provenance=context.backend_provenance,
            )
            writer.append_fhp21(export_value)
            export_event_id = f"{event_prefix}:export:{match['match_id']}"
            writer.append(
                event_id=export_event_id,
                stage="EXPORT",
                status=("SUCCEEDED" if temporal["valid_landmark_count"] == 21 else "WARNING"),
                event="fhp21_record_exported",
                payload={
                    **export_value,
                    "output_file": "fhp21.jsonl",
                    "projected_keypoints_space": "rectified",
                    "projected_keypoints_uv": temporal_projection,
                },
                parent_event_ids=(temporal_event_id,),
            )
            self.state.export_count += 1
        else:
            self.state.active_stage = "EXPORT"
            export_event_id = f"{event_prefix}:export:{match['match_id']}"
            writer.append(
                event_id=export_event_id,
                stage="EXPORT",
                status="SKIPPED",
                event="fhp21_record_not_produced",
                payload={
                    "frame_id": frame_id,
                    "frame_index": frame.global_index,
                    "timestamp_ns": timestamp_ns,
                    "track_id": track_id,
                    "source_observation_ids": [observation["observation_id"]],
                    "output_status": "NOT_PRODUCED",
                    "output_file": None,
                    "selected_output_stage": "TEMPORAL_REFINEMENT",
                    "reason": "NO_VALID_TEMPORAL_LANDMARK",
                    "landmarks_xyz_m": temporal["landmarks_xyz_m"],
                    "validity": temporal["validity"],
                    "projected_keypoints_space": "rectified",
                    "projected_keypoints_uv": temporal_projection,
                },
                parent_event_ids=(temporal_event_id,),
            )
        return [export_event_id], {
            "track_id": track_id,
            "raw": raw_projection,
            "stable": temporal_projection,
            "stable_input_stage": temporal_input_stage,
        }

    def _append_overlay_frame(
        self,
        frame: FrameInput,
        *,
        frame_id: str,
        rendered_views: dict[str, dict[str, Any]] | None,
        tracks: list[dict[str, Any]],
        export_event_ids: list[str],
    ) -> None:
        overlay_video = self.context.overlay_video
        if overlay_video is None:
            return
        assert rendered_views is not None
        overlay_video.append_frame(
            left_frame=rendered_views["left"]["rectified"],
            right_frame=rendered_views["right"]["rectified"],
            frame_id=frame_id,
            frame_index=frame.global_index,
            timestamp_ns=frame.pair.pair_timestamp_ns,
            tracks=tracks,
        )
        self.state.overlay_parent_event_ids.extend(export_event_ids)


__all__ = [
    "FrameExecutionContext",
    "FrameExecutionState",
    "FrameExecutor",
    "FrameInput",
    "FrameOutcome",
]
