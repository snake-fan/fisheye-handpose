"""Deterministic V0 metrics extracted from one immutable canonical run trace."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ._generated_project_contract import BASELINE_METRICS_SCHEMA
from .errors import FisheyeHandposeError
from .joints import FHP21
from .trace import RunArtifactReader, TraceRecord

BASELINE_SCHEMA = BASELINE_METRICS_SCHEMA


class BaselineExtractionError(FisheyeHandposeError):
    """A canonical run cannot be represented by the V0 baseline contract."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload(record: TraceRecord) -> dict[str, Any]:
    if not isinstance(record.payload, dict):
        raise BaselineExtractionError(f"record {record.record_id} payload must be an object")
    return record.payload


def _list_field(record: TraceRecord, name: str) -> list[Any]:
    value = _payload(record).get(name)
    if not isinstance(value, list):
        raise BaselineExtractionError(f"record {record.record_id} {name} must be an array")
    return value


def _histogram(values: list[int]) -> dict[str, int]:
    counts = Counter(values)
    return {str(value): counts[value] for value in sorted(counts)}


def _rounded(value: float) -> float:
    return round(float(value), 9)


def _ratio(numerator: int, denominator: int) -> float | None:
    return _rounded(numerator / denominator) if denominator else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _distribution(values: list[float]) -> dict[str, int | float | None]:
    return {
        "sample_count": len(values),
        "min": None if not values else _rounded(min(values)),
        "median": None if not values else _rounded(statistics.median(values)),
        "p95": None if not values else _rounded(_percentile(values, 0.95)),
        "max": None if not values else _rounded(max(values)),
    }


def _text_field(record: TraceRecord, name: str) -> str:
    value = _payload(record).get(name)
    if not isinstance(value, str) or not value:
        raise BaselineExtractionError(f"record {record.record_id} {name} must be non-empty")
    return value


def _integer_field(record: TraceRecord, name: str) -> int:
    value = _payload(record).get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BaselineExtractionError(f"record {record.record_id} {name} must be an integer")
    return value


def _raw_observations(
    records: tuple[TraceRecord, ...],
) -> tuple[dict[str, Any], ...]:
    observations: list[dict[str, Any]] = []
    for record in records:
        if record.event != "raw_landmarks_triangulated":
            continue
        payload = _payload(record)
        points = _list_field(record, "landmarks_xyz_m")
        validity = _list_field(record, "validity")
        if len(points) != 21 or len(validity) != 21:
            raise BaselineExtractionError(
                f"record {record.record_id} raw landmark arrays must contain 21 entries"
            )
        valid_count = 0
        normalized_points: list[tuple[float, float, float] | None] = []
        for index, (point, flag) in enumerate(zip(points, validity, strict=True)):
            if not isinstance(flag, str) or not flag:
                raise BaselineExtractionError(
                    f"record {record.record_id} validity[{index}] must be non-empty"
                )
            if flag != "VALID":
                normalized_points.append(None)
                continue
            if not isinstance(point, list) or len(point) != 3:
                raise BaselineExtractionError(
                    f"record {record.record_id} valid landmark {index} must contain xyz"
                )
            xyz = tuple(float(value) for value in point)
            if not all(math.isfinite(value) for value in xyz):
                raise BaselineExtractionError(
                    f"record {record.record_id} valid landmark {index} must be finite"
                )
            normalized_points.append(xyz)
            valid_count += 1
        declared = payload.get("valid_landmark_count")
        if declared != valid_count:
            raise BaselineExtractionError(
                f"record {record.record_id} valid_landmark_count disagrees with validity"
            )
        observations.append(
            {
                "record_id": record.record_id,
                "frame_id": _text_field(record, "frame_id"),
                "frame_index": _integer_field(record, "frame_index"),
                "track_id": _text_field(record, "track_id"),
                "points": normalized_points,
                "validity": validity,
                "valid_count": valid_count,
                "track_assignment": payload.get("track_assignment"),
            }
        )
    return tuple(observations)


def _raw_metrics(observations: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    invalid_reasons: Counter[str] = Counter()
    for observation in observations:
        invalid_reasons.update(flag for flag in observation["validity"] if flag != "VALID")
    joint_slots = 21 * len(observations)
    valid_count = sum(observation["valid_count"] for observation in observations)
    return {
        "hand_frame_count": len(observations),
        "joint_slot_count": joint_slots,
        "valid_joint_count": valid_count,
        "valid_joint_rate": _ratio(valid_count, joint_slots),
        "complete_hand_frame_count": sum(
            observation["valid_count"] == 21 for observation in observations
        ),
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
    }


def _bone_metrics(observations: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    by_edge: dict[tuple[int, int], list[float]] = {edge: [] for edge in FHP21.edges}
    track_edge: dict[tuple[str, tuple[int, int]], list[float]] = defaultdict(list)
    all_lengths: list[float] = []
    hand_frame_maxima: list[float] = []
    for observation in observations:
        frame_lengths: list[float] = []
        for edge in FHP21.edges:
            parent, child = edge
            parent_point = observation["points"][parent]
            child_point = observation["points"][child]
            if parent_point is None or child_point is None:
                continue
            length = round(
                math.sqrt(
                    sum(
                        (left - right) ** 2
                        for left, right in zip(parent_point, child_point, strict=True)
                    )
                ),
                12,
            )
            by_edge[edge].append(length)
            track_edge[(observation["track_id"], edge)].append(length)
            all_lengths.append(length)
            frame_lengths.append(length)
        if frame_lengths:
            hand_frame_maxima.append(max(frame_lengths))
    cvs: list[float] = []
    for values in track_edge.values():
        if len(values) < 2:
            continue
        mean = statistics.fmean(values)
        if mean <= 0:
            continue
        variance = statistics.fmean((value - mean) ** 2 for value in values)
        cvs.append(math.sqrt(variance) / mean)
    return {
        "quantile_method": "linear_interpolation_r7",
        "overall": _distribution(all_lengths),
        "threshold_counts": {
            "over_0_05_m": sum(value > 0.05 for value in all_lengths),
            "over_0_10_m": sum(value > 0.10 for value in all_lengths),
            "hand_frames_over_0_05_m": sum(value > 0.05 for value in hand_frame_maxima),
            "hand_frames_over_0_10_m": sum(value > 0.10 for value in hand_frame_maxima),
        },
        "by_edge": {
            f"{parent}-{child}": {
                "parent_name": FHP21.names[parent],
                "child_name": FHP21.names[child],
                **_distribution(by_edge[(parent, child)]),
            }
            for parent, child in FHP21.edges
        },
        "track_edge_cv": {
            "minimum_samples_per_series": 2,
            "series_count": len(cvs),
            "min": None if not cvs else _rounded(min(cvs)),
            "median": None if not cvs else _rounded(statistics.median(cvs)),
            "p95": None if not cvs else _rounded(_percentile(cvs, 0.95)),
            "max": None if not cvs else _rounded(max(cvs)),
        },
    }


def _hand_and_track_metrics(
    observations: tuple[dict[str, Any], ...],
    records: tuple[TraceRecord, ...],
    synchronized_frame_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_frame = Counter({frame_id: 0 for frame_id in synchronized_frame_ids})
    by_frame.update(observation["frame_id"] for observation in observations)
    export_count = sum(record.event == "fhp21_record_exported" for record in records)
    tracks: dict[str, dict[str, Any]] = {}
    for observation in observations:
        track = tracks.setdefault(
            observation["track_id"],
            {"indices": [], "decisions": Counter()},
        )
        track["indices"].append(observation["frame_index"])
        assignment = observation["track_assignment"]
        if isinstance(assignment, dict) and isinstance(assignment.get("decision"), str):
            track["decisions"][assignment["decision"]] += 1
    by_track = {
        track_id: {
            "hand_frame_count": len(value["indices"]),
            "first_frame_index": min(value["indices"]),
            "last_frame_index": max(value["indices"]),
            "new_assignment_count": value["decisions"]["NEW"],
            "matched_assignment_count": value["decisions"]["MATCHED"],
        }
        for track_id, value in sorted(tracks.items())
    }
    return (
        {
            "raw_hand_frame_count": len(observations),
            "exported_hand_frame_count": export_count,
            "raw_hands_per_frame_histogram": _histogram(list(by_frame.values())),
            "frames_with_any_raw_hand": sum(count > 0 for count in by_frame.values()),
            "frames_with_two_raw_hands": sum(count == 2 for count in by_frame.values()),
        },
        {
            "track_count": len(by_track),
            "new_assignment_count": sum(
                value["new_assignment_count"] for value in by_track.values()
            ),
            "matched_assignment_count": sum(
                value["matched_assignment_count"] for value in by_track.values()
            ),
            "by_track": by_track,
        },
    )


def _mano_metrics(records: tuple[TraceRecord, ...], *, configured: bool | None) -> dict[str, Any]:
    mano_records = [
        record
        for record in records
        if record.event
        in {"mano_frame_fitted", "mano_frame_not_produced", "mano_frame_not_configured"}
        and isinstance(record.payload, dict)
        and record.payload.get("track_id") is not None
    ]
    decisions: Counter[str] = Counter()
    attempt_statuses: Counter[str] = Counter()
    attempt_rmse: list[float] = []
    accepted_rmse: list[float] = []
    produced = 0
    for record in mano_records:
        payload = _payload(record)
        if payload.get("output_status") == "PRODUCED":
            produced += 1
            rmse = payload.get("rmse_m")
            if (
                isinstance(rmse, (int, float))
                and not isinstance(rmse, bool)
                and math.isfinite(rmse)
            ):
                accepted_rmse.append(float(rmse))
        selection = payload.get("selection")
        if not isinstance(selection, dict):
            continue
        decision = selection.get("decision")
        if isinstance(decision, str) and decision:
            decisions[decision] += 1
        attempts = selection.get("attempts")
        if not isinstance(attempts, list):
            raise BaselineExtractionError(
                f"record {record.record_id} selection.attempts must be an array"
            )
        for attempt in attempts:
            if not isinstance(attempt, dict):
                raise BaselineExtractionError(
                    f"record {record.record_id} MANO attempt must be an object"
                )
            status = attempt.get("status")
            if isinstance(status, str) and status:
                attempt_statuses[status] += 1
            rmse = attempt.get("rmse_m")
            if (
                isinstance(rmse, (int, float))
                and not isinstance(rmse, bool)
                and math.isfinite(rmse)
            ):
                attempt_rmse.append(float(rmse))
    return {
        "configured": configured,
        "hand_frame_count": len(mano_records),
        "produced_count": produced,
        "not_produced_count": len(mano_records) - produced,
        "production_rate": _ratio(produced, len(mano_records)),
        "decision_counts": dict(sorted(decisions.items())),
        "attempt_count": sum(attempt_statuses.values()),
        "attempt_status_counts": dict(sorted(attempt_statuses.items())),
        "attempt_rmse_m": _distribution(attempt_rmse),
        "accepted_rmse_m": _distribution(accepted_rmse),
    }


def _temporal_metrics(records: tuple[TraceRecord, ...]) -> dict[str, Any]:
    temporal_records = [
        record
        for record in records
        if record.event in {"temporal_landmarks_refined", "temporal_landmarks_not_produced"}
        and isinstance(record.payload, dict)
        and record.payload.get("track_id") is not None
    ]
    produced = 0
    input_stages: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    for record in temporal_records:
        payload = _payload(record)
        produced += payload.get("output_status") == "PRODUCED"
        input_stage = payload.get("input_stage")
        method = payload.get("method")
        if isinstance(input_stage, str) and input_stage:
            input_stages[input_stage] += 1
        if isinstance(method, str) and method:
            methods[method] += 1
    return {
        "hand_frame_count": len(temporal_records),
        "produced_count": produced,
        "not_produced_count": len(temporal_records) - produced,
        "input_stage_counts": dict(sorted(input_stages.items())),
        "method_counts": dict(sorted(methods.items())),
    }


def _worker_snapshot(reader: RunArtifactReader, records: tuple[TraceRecord, ...]) -> Any:
    references = {
        (blob.sha256, blob.relative_path): blob
        for record in records
        for blob in record.blobs
        if blob.role == "worker_manifest"
    }
    if not references:
        return None
    if len(references) != 1:
        raise BaselineExtractionError("run references multiple distinct worker manifests")
    blob = next(iter(references.values()))
    path = reader.root / blob.relative_path
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BaselineExtractionError(f"cannot read worker configuration snapshot: {exc}") from exc
    if not isinstance(manifest, dict):
        raise BaselineExtractionError("worker manifest must be an object")
    return {
        "manifest_sha256": blob.sha256,
        "request": manifest.get("request"),
        "configuration": manifest.get("configuration"),
        "calibration": manifest.get("calibration"),
        "models": manifest.get("models"),
    }


def _configuration_snapshot(
    reader: RunArtifactReader, records: tuple[TraceRecord, ...]
) -> dict[str, Any]:
    manifest = reader.manifest
    value = {
        "core": {
            "manifest_hash": manifest["manifest_hash"],
            "pipeline_version": manifest["pipeline_version"],
            "config": manifest["config"],
            "inputs": manifest["inputs"],
            "metadata": manifest["metadata"],
        },
        "worker": _worker_snapshot(reader, records),
    }
    return {"sha256": hashlib.sha256(_canonical_json(value)).hexdigest(), "value": value}


def extract_baseline_metrics(
    source: str | Path | RunArtifactReader,
    *,
    verify_blobs: bool = True,
) -> dict[str, Any]:
    """Extract a versioned baseline through the canonical trace reader seam."""

    reader = source if isinstance(source, RunArtifactReader) else RunArtifactReader(source)
    validation = reader.validate(verify_blobs=verify_blobs)
    records = reader.records()
    manifest = reader.manifest

    sync_records = tuple(record for record in records if record.event == "stereo_pair_selected")
    frame_ids: set[str] = set()
    frame_indices: list[int] = []
    for record in sync_records:
        payload = _payload(record)
        frame_id = payload.get("frame_id")
        frame_index = payload.get("frame_index")
        if not isinstance(frame_id, str) or not frame_id:
            raise BaselineExtractionError(f"record {record.record_id} frame_id must be non-empty")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise BaselineExtractionError(
                f"record {record.record_id} frame_index must be an integer"
            )
        frame_ids.add(frame_id)
        frame_indices.append(frame_index)

    detection_records = tuple(
        record for record in records if record.event == "hand_candidates_detected"
    )
    detection_counts: list[int] = []
    per_view_counts: dict[str, list[int]] = defaultdict(list)
    for record in detection_records:
        payload = _payload(record)
        view_id = payload.get("view_id")
        if not isinstance(view_id, str) or not view_id:
            raise BaselineExtractionError(f"record {record.record_id} view_id must be non-empty")
        count = len(_list_field(record, "detections"))
        detection_counts.append(count)
        per_view_counts[view_id].append(count)

    association_records = tuple(
        record for record in records if record.event == "cross_view_hands_associated"
    )
    match_counts: list[int] = []
    unmatched_left_count = 0
    unmatched_right_count = 0
    for record in association_records:
        match_counts.append(len(_list_field(record, "matches")))
        unmatched_left_count += len(_list_field(record, "unmatched_left_indices"))
        unmatched_right_count += len(_list_field(record, "unmatched_right_indices"))

    configuration_snapshot = _configuration_snapshot(reader, records)
    worker = configuration_snapshot["value"]["worker"]
    worker_configuration = worker.get("configuration") if isinstance(worker, dict) else None
    if isinstance(worker_configuration, dict) and "mano" in worker_configuration:
        mano_configured: bool | None = worker_configuration["mano"] is not None
    elif any(record.event == "mano_models_loaded" for record in records):
        mano_configured = True
    elif any(
        record.event in {"mano_not_configured", "mano_frame_not_configured"} for record in records
    ):
        mano_configured = False
    else:
        mano_configured = None
    raw_observations = _raw_observations(records)
    hand_metrics, track_metrics = _hand_and_track_metrics(raw_observations, records, frame_ids)

    return {
        "schema_version": BASELINE_SCHEMA,
        "run": {
            "run_id": validation.run_id,
            "status": validation.status.value,
            "pipeline_version": manifest["pipeline_version"],
            "record_count": validation.record_count,
            "last_record_hash": validation.last_hash,
        },
        "configuration_snapshot": configuration_snapshot,
        "frames": {
            "pair_count": len(sync_records),
            "unique_frame_count": len(frame_ids),
            "frame_index_min": min(frame_indices) if frame_indices else None,
            "frame_index_max": max(frame_indices) if frame_indices else None,
        },
        "detection": {
            "view_frame_count": len(detection_records),
            "candidate_count": sum(detection_counts),
            "candidate_count_histogram": _histogram(detection_counts),
            "per_view": {
                view_id: {
                    "view_frame_count": len(counts),
                    "candidate_count": sum(counts),
                    "candidate_count_histogram": _histogram(counts),
                }
                for view_id, counts in sorted(per_view_counts.items())
            },
        },
        "association": {
            "frame_count": len(association_records),
            "match_count": sum(match_counts),
            "match_count_histogram": _histogram(match_counts),
            "two_hand_frame_count": sum(count == 2 for count in match_counts),
            "unmatched_left_count": unmatched_left_count,
            "unmatched_right_count": unmatched_right_count,
        },
        "hands": hand_metrics,
        "raw_3d": _raw_metrics(raw_observations),
        "bone_lengths_m": _bone_metrics(raw_observations),
        "tracks": track_metrics,
        "mano": _mano_metrics(records, configured=mano_configured),
        "temporal": _temporal_metrics(records),
    }


__all__ = [
    "BASELINE_SCHEMA",
    "BaselineExtractionError",
    "extract_baseline_metrics",
]
