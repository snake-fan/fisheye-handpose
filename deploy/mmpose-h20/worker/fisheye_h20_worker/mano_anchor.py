"""Deterministic, pose-agnostic MANO anchor selection over complete tracks."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Literal

ANCHOR_METHOD_ID = "mano_anchor_quality_v1"


@dataclass(frozen=True)
class ManoFrameEvidence:
    track_id: str
    frame_id: str
    timestamp_ns: int
    raw_validity: tuple[bool, ...]
    keypoint_scores_2d: tuple[float | None, ...]
    reprojection_errors_px: tuple[float | None, ...]
    bone_outlier_fraction: float
    triangulation_quality: float


@dataclass(frozen=True)
class AnchorScoreComponent:
    name: str
    normalized_value: float
    weight: float
    contribution: float


@dataclass(frozen=True)
class AnchorDecision:
    track_id: str
    frame_id: str
    timestamp_ns: int
    accepted: bool
    quality_accepted: bool
    reason: str
    score: float | None
    score_components: tuple[AnchorScoreComponent, ...]
    selection_rank: int | None = None
    pose_prior_applied: bool = False
    nearest_selected_delta_ns: int | None = None


@dataclass(frozen=True)
class TrackAnchorSelection:
    track_id: str
    anchors: tuple[AnchorDecision, ...]
    decisions: tuple[AnchorDecision, ...]
    status: Literal["VALID", "NO_ACCEPTABLE_ANCHOR"]
    reason: str | None

    def decision_for(self, frame_id: str) -> AnchorDecision:
        for decision in self.decisions:
            if decision.frame_id == frame_id:
                return decision
        raise KeyError(frame_id)


@dataclass(frozen=True)
class AnchorSelectionBatch:
    tracks: tuple[TrackAnchorSelection, ...]
    method_id: str = ANCHOR_METHOD_ID

    def for_track(self, track_id: str) -> TrackAnchorSelection:
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        raise KeyError(track_id)


class ManoAnchorSelector:
    """Choose top-quality, time-diverse anchors without a first-frame or flat-pose prior."""

    _COMPONENTS = (
        ("raw_valid_joint_ratio", 0.30),
        ("mean_2d_score", 0.20),
        ("reprojection_quality", 0.20),
        ("bone_quality", 0.10),
        ("triangulation_quality", 0.20),
    )

    def __init__(
        self,
        *,
        top_k: int = 3,
        min_time_separation_ns: int = 200_000_000,
        min_valid_joints: int = 15,
        min_mean_2d_score: float = 0.25,
        max_median_reprojection_px: float = 5.0,
        max_bone_outlier_fraction: float = 0.25,
        min_triangulation_quality: float = 0.30,
    ) -> None:
        self.top_k = top_k
        self.min_time_separation_ns = min_time_separation_ns
        self.min_valid_joints = min_valid_joints
        self.min_mean_2d_score = min_mean_2d_score
        self.max_median_reprojection_px = max_median_reprojection_px
        self.max_bone_outlier_fraction = max_bone_outlier_fraction
        self.min_triangulation_quality = min_triangulation_quality

    def select(self, evidence: list[ManoFrameEvidence]) -> AnchorSelectionBatch:
        grouped: dict[str, list[ManoFrameEvidence]] = defaultdict(list)
        for frame in evidence:
            grouped[frame.track_id].append(frame)
        tracks = tuple(
            self._select_track(track_id, grouped[track_id]) for track_id in sorted(grouped)
        )
        return AnchorSelectionBatch(tracks=tracks)

    def _select_track(
        self, track_id: str, evidence: list[ManoFrameEvidence]
    ) -> TrackAnchorSelection:
        evaluated = [self._evaluate(frame) for frame in evidence]
        candidates = sorted(
            (decision for decision in evaluated if decision.quality_accepted),
            key=lambda decision: (
                -float(decision.score),
                decision.timestamp_ns,
                decision.frame_id,
            ),
        )
        replacements: dict[str, AnchorDecision] = {}
        selected: list[AnchorDecision] = []
        for candidate in candidates:
            if len(selected) >= self.top_k:
                replacements[candidate.frame_id] = replace(
                    candidate, accepted=False, reason="TOP_K_LIMIT"
                )
                continue
            selected_deltas = [
                abs(candidate.timestamp_ns - anchor.timestamp_ns) for anchor in selected
            ]
            nearest_delta = min(selected_deltas) if selected_deltas else None
            if nearest_delta is not None and nearest_delta < self.min_time_separation_ns:
                replacements[candidate.frame_id] = replace(
                    candidate,
                    accepted=False,
                    reason="TIME_REDUNDANT",
                    nearest_selected_delta_ns=nearest_delta,
                )
                continue
            chosen = replace(
                candidate,
                accepted=True,
                reason="SELECTED",
                selection_rank=len(selected) + 1,
                nearest_selected_delta_ns=nearest_delta,
            )
            replacements[candidate.frame_id] = chosen
            selected.append(chosen)
        decisions = tuple(
            sorted(
                (replacements.get(decision.frame_id, decision) for decision in evaluated),
                key=lambda decision: (decision.timestamp_ns, decision.frame_id),
            )
        )
        anchors = tuple(sorted(selected, key=lambda decision: int(decision.selection_rank or 0)))
        status: Literal["VALID", "NO_ACCEPTABLE_ANCHOR"] = (
            "VALID" if anchors else "NO_ACCEPTABLE_ANCHOR"
        )
        return TrackAnchorSelection(
            track_id=track_id,
            anchors=anchors,
            decisions=decisions,
            status=status,
            reason=None if anchors else "NO_ACCEPTABLE_ANCHOR",
        )

    def _evaluate(self, frame: ManoFrameEvidence) -> AnchorDecision:
        numeric_values = [
            frame.bone_outlier_fraction,
            frame.triangulation_quality,
            *(value for value in frame.keypoint_scores_2d if value is not None),
            *(value for value in frame.reprojection_errors_px if value is not None),
        ]
        try:
            finite = all(math.isfinite(float(value)) for value in numeric_values)
        except (TypeError, ValueError):
            finite = False
        if not finite:
            return AnchorDecision(
                track_id=frame.track_id,
                frame_id=frame.frame_id,
                timestamp_ns=frame.timestamp_ns,
                accepted=False,
                quality_accepted=False,
                reason="NON_FINITE_EVIDENCE",
                score=None,
                score_components=(),
            )
        valid_indices = [index for index, valid in enumerate(frame.raw_validity) if valid]
        scores = [float(frame.keypoint_scores_2d[index]) for index in valid_indices]
        reprojections = [float(frame.reprojection_errors_px[index]) for index in valid_indices]
        mean_2d_score = statistics.fmean(scores) if scores else 0.0
        median_reprojection_px = statistics.median(reprojections) if reprojections else 1e9
        values = (
            len(valid_indices) / 21.0,
            mean_2d_score,
            1.0 / (1.0 + median_reprojection_px / 3.0),
            1.0 - frame.bone_outlier_fraction,
            frame.triangulation_quality,
        )
        components = tuple(
            AnchorScoreComponent(
                name=name,
                normalized_value=value,
                weight=weight,
                contribution=value * weight,
            )
            for (name, weight), value in zip(self._COMPONENTS, values, strict=True)
        )
        score = sum(component.contribution for component in components)
        reason = "CANDIDATE"
        if len(valid_indices) < self.min_valid_joints:
            reason = "INSUFFICIENT_RAW_SUPPORT"
        elif mean_2d_score < self.min_mean_2d_score:
            reason = "LOW_2D_SCORE"
        elif median_reprojection_px > self.max_median_reprojection_px:
            reason = "HIGH_REPROJECTION_ERROR"
        elif frame.bone_outlier_fraction > self.max_bone_outlier_fraction:
            reason = "BONE_LENGTH_OUTLIER"
        elif frame.triangulation_quality < self.min_triangulation_quality:
            reason = "LOW_TRIANGULATION_QUALITY"
        quality_accepted = reason == "CANDIDATE"
        return AnchorDecision(
            track_id=frame.track_id,
            frame_id=frame.frame_id,
            timestamp_ns=frame.timestamp_ns,
            accepted=False,
            quality_accepted=quality_accepted,
            reason=reason,
            score=score,
            score_components=components,
        )


__all__ = [
    "ANCHOR_METHOD_ID",
    "AnchorDecision",
    "AnchorScoreComponent",
    "AnchorSelectionBatch",
    "ManoAnchorSelector",
    "ManoFrameEvidence",
    "TrackAnchorSelection",
]
