"""Backend-neutral, geometry-preserving contracts for hand-pose perception.

Array-valued fields deliberately remain backend-agnostic, but their shapes and coordinate
spaces are normative. Identifiers on the surrounding dataclasses prevent an array from
crossing a module boundary without its geometry, time, calibration, and provenance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from .calibration import Mat3, Transform3D
from .joints import FHP21, LandmarkMappingRecord, MappingQuality

LANDMARK_COUNT = 21


def _require_text(value: str, label: str) -> None:
    if not value.strip():
        raise ValueError(f"{label} must be non-empty")


def _require_probability(value: float, label: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be a finite probability in [0, 1]")


def _require_landmark_count(values: Sequence[Any], label: str) -> None:
    if len(values) != LANDMARK_COUNT:
        raise ValueError(f"{label} must contain {LANDMARK_COUNT} entries")


def _array_shape(value: Any) -> tuple[int, ...] | None:
    """Return a NumPy/Torch/regular-sequence shape without adopting an array library."""

    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(int(dimension) for dimension in shape)
        except (TypeError, ValueError):
            return None
    dimensions: list[int] = []
    current = value
    while isinstance(current, (list, tuple)):
        dimensions.append(len(current))
        if not current:
            break
        current = current[0]
    return tuple(dimensions) if dimensions else None


def _require_shape(value: Any, expected: tuple[int, ...], label: str) -> None:
    shape = _array_shape(value)
    if shape != expected:
        raise ValueError(f"{label} must have shape {expected}, got {shape}")


class CommercialUse(StrEnum):
    ALLOWED = "ALLOWED"
    DENIED = "DENIED"
    CONDITIONAL = "CONDITIONAL"
    UNKNOWN = "UNKNOWN"


class PixelSpaceKind(StrEnum):
    RAW_DISTORTED = "RAW_DISTORTED"
    VIRTUAL_PINHOLE = "VIRTUAL_PINHOLE"
    RECTIFIED_PINHOLE = "RECTIFIED_PINHOLE"


class FrameKind(StrEnum):
    CAMERA = "CAMERA"
    VIRTUAL_CAMERA = "VIRTUAL_CAMERA"
    RIG = "RIG"


class Validity(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"


class EvidenceSource(StrEnum):
    MULTIVIEW = "MULTIVIEW"
    MONOCULAR = "MONOCULAR"
    NONE = "NONE"


class EstimateStage(StrEnum):
    RAW_FUSION = "RAW_FUSION"
    KINEMATIC_REFINEMENT = "KINEMATIC_REFINEMENT"
    TEMPORAL_REFINEMENT = "TEMPORAL_REFINEMENT"


class EstimateKind(StrEnum):
    MEASURED = "MEASURED"
    REFINED = "REFINED"
    PREDICTED = "PREDICTED"


class TemporalMode(StrEnum):
    CAUSAL = "CAUSAL"
    FIXED_LAG = "FIXED_LAG"
    OFFLINE = "OFFLINE"


class ResetReason(StrEnum):
    END_OF_TRACK = "END_OF_TRACK"
    TRACK_ID_SWITCH = "TRACK_ID_SWITCH"
    SCENE_CUT = "SCENE_CUT"
    CALIBRATION_CHANGE = "CALIBRATION_CHANGE"
    TIME_GAP = "TIME_GAP"
    CALLER_REQUEST = "CALLER_REQUEST"


@dataclass(frozen=True, slots=True)
class BackendManifest:
    """Deployment and compatibility declaration for a model adapter.

    License strings are SPDX expressions where possible and otherwise stable terms URIs.
    A deployment policy must treat ``UNKNOWN`` as not approved, rather than guessing.
    """

    api_version: str
    backend_name: str
    backend_version: str
    code_license: str
    weights_license: str | None
    commercial_use: CommercialUse
    input_joint_set_ids: tuple[str, ...] = ()
    output_joint_set_id: str | None = None
    input_pixel_space_kinds: tuple[PixelSpaceKind, ...] = ()
    capabilities: frozenset[str] = frozenset()
    source_revision: str | None = None
    weights_hash: str | None = None
    terms_uri: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("api_version", self.api_version),
            ("backend_name", self.backend_name),
            ("backend_version", self.backend_version),
            ("code_license", self.code_license),
        ):
            _require_text(value, label)
        if not isinstance(self.commercial_use, CommercialUse):
            raise TypeError("commercial_use must be an explicit CommercialUse value")


@dataclass(frozen=True, slots=True)
class PixelSpace:
    """A named 2D pixel domain.

    Coordinates are zero-based ``(u, v)`` locations measured at pixel centres. Bounding
    boxes use half-open ``(x_min, y_min, x_max, y_max)`` edges. ``image_size_wh`` is
    always ``(width, height)``. Covariances expressed in this space are in squared pixels
    of this exact domain, after adapters have undone resize, letterbox, and mirror steps.
    """

    pixel_space_id: str
    kind: PixelSpaceKind
    image_size_wh: tuple[int, int]
    calibration_id: str
    source_camera_id: str

    def __post_init__(self) -> None:
        _require_text(self.pixel_space_id, "pixel_space_id")
        _require_text(self.calibration_id, "calibration_id")
        _require_text(self.source_camera_id, "source_camera_id")
        if not isinstance(self.kind, PixelSpaceKind):
            raise TypeError("pixel-space kind must be a PixelSpaceKind value")
        if self.image_size_wh[0] <= 0 or self.image_size_wh[1] <= 0:
            raise ValueError("pixel-space image dimensions must be positive")


@dataclass(frozen=True, slots=True)
class CoordinateFrame3D:
    """Named 3D coordinate frame; all contract lengths are metres."""

    frame_id: str
    kind: FrameKind
    axis_convention: str
    length_unit: str = "m"

    def __post_init__(self) -> None:
        _require_text(self.frame_id, "frame_id")
        _require_text(self.axis_convention, "axis_convention")
        if not isinstance(self.kind, FrameKind):
            raise TypeError("coordinate-frame kind must be a FrameKind value")
        if self.length_unit != "m":
            raise ValueError("contract 3D lengths must be expressed in metres")


@dataclass(frozen=True, slots=True)
class HandednessProbabilities:
    """Probabilities of anatomical hand side, never image-left/image-right position."""

    left: float
    right: float
    unknown: float

    def __post_init__(self) -> None:
        for label, value in (("left", self.left), ("right", self.right), ("unknown", self.unknown)):
            _require_probability(value, f"handedness.{label}")
        if not math.isclose(self.left + self.right + self.unknown, 1.0, abs_tol=1e-6):
            raise ValueError("handedness probabilities must sum to one")


@dataclass(frozen=True, slots=True)
class ImageView:
    """One physical camera sample in its native distorted pixel domain."""

    view_id: str
    camera_id: str
    timestamp_ns: int
    calibration_id: str
    pixel_space: PixelSpace
    image: Any

    def __post_init__(self) -> None:
        _require_text(self.view_id, "view_id")
        _require_text(self.camera_id, "camera_id")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self.pixel_space.kind is not PixelSpaceKind.RAW_DISTORTED:
            raise ValueError("ImageView must identify the native distorted camera image")
        if self.pixel_space.source_camera_id != self.camera_id:
            raise ValueError("ImageView camera and pixel-space camera do not match")
        if self.pixel_space.calibration_id != self.calibration_id:
            raise ValueError("ImageView calibration and pixel-space calibration do not match")


@dataclass(frozen=True, slots=True)
class FrameSet:
    """Synchronized named views and their actual capture times.

    ``timestamp_ns`` is the representative pipeline timestamp. Each ``ImageView`` still
    retains its actual exposure timestamp for synchronization QA and future motion-aware
    extensions.
    """

    frame_set_id: str
    sequence_id: str
    timestamp_ns: int
    calibration_id: str
    rig_frame: CoordinateFrame3D
    sync_tolerance_ns: int
    views: tuple[ImageView, ...]

    def __post_init__(self) -> None:
        _require_text(self.frame_set_id, "frame_set_id")
        _require_text(self.sequence_id, "sequence_id")
        if self.timestamp_ns < 0 or self.sync_tolerance_ns < 0:
            raise ValueError("frame-set timestamp and sync tolerance must be non-negative")
        if self.rig_frame.kind is not FrameKind.RIG:
            raise ValueError("FrameSet.rig_frame must be a RIG frame")
        if not self.views:
            raise ValueError("a FrameSet must contain at least one view")
        if len({view.view_id for view in self.views}) != len(self.views):
            raise ValueError("FrameSet view IDs must be unique")
        if len({view.camera_id for view in self.views}) != len(self.views):
            raise ValueError("FrameSet may contain at most one sample from each camera")
        if any(view.calibration_id != self.calibration_id for view in self.views):
            raise ValueError("all FrameSet views must use its declared calibration")
        timestamps = [view.timestamp_ns for view in self.views]
        if not min(timestamps) <= self.timestamp_ns <= max(timestamps):
            raise ValueError("representative timestamp must lie within the source timestamps")
        if self.observed_skew_ns > self.sync_tolerance_ns:
            raise ValueError("source timestamp spread exceeds sync_tolerance_ns")

    @property
    def observed_skew_ns(self) -> int:
        timestamps = [view.timestamp_ns for view in self.views]
        return max(timestamps) - min(timestamps)


@dataclass(frozen=True, slots=True)
class Detection2D:
    """A hand candidate returned in the source view's native pixel space."""

    detection_id: str
    view_id: str
    bbox_xyxy: tuple[float, float, float, float]
    detector_score: float
    pixel_space_id: str
    handedness: HandednessProbabilities | None = None
    embedding: Any | None = None

    def __post_init__(self) -> None:
        _require_text(self.detection_id, "detection_id")
        _require_text(self.view_id, "view_id")
        x_min, y_min, x_max, y_max = self.bbox_xyxy
        if not all(math.isfinite(value) for value in self.bbox_xyxy):
            raise ValueError("detection bounding box must be finite")
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("detection bounding box must have positive area")
        if not math.isfinite(self.detector_score):
            raise ValueError("detector_score must be finite")


@dataclass(frozen=True, slots=True)
class VirtualCamera:
    """Local pinhole camera used by a hand-centred perspective crop.

    The virtual and source physical cameras have the same optical centre. A crop pixel is
    mapped authoritatively by unprojecting through ``intrinsics`` and
    ``T_rig_from_virtual``, then projecting the resulting rig ray through the calibrated
    source camera. A single image homography is not assumed.
    """

    virtual_camera_id: str
    pixel_space: PixelSpace
    source_camera_id: str
    calibration_id: str
    intrinsics: Mat3
    T_rig_from_virtual: Transform3D

    def __post_init__(self) -> None:
        _require_text(self.virtual_camera_id, "virtual_camera_id")
        if self.pixel_space.kind is not PixelSpaceKind.VIRTUAL_PINHOLE:
            raise ValueError("VirtualCamera pixel space must be VIRTUAL_PINHOLE")
        if self.pixel_space.source_camera_id != self.source_camera_id:
            raise ValueError("virtual and pixel-space source cameras do not match")
        if self.pixel_space.calibration_id != self.calibration_id:
            raise ValueError("virtual and pixel-space calibrations do not match")
        if self.T_rig_from_virtual.from_frame != self.virtual_camera_id:
            raise ValueError("T_rig_from_virtual has the wrong source frame")
        fx, fy = self.intrinsics[0][0], self.intrinsics[1][1]
        if (
            fx <= 0
            or fy <= 0
            or not all(math.isfinite(value) for row in self.intrinsics for value in row)
        ):
            raise ValueError("virtual-camera intrinsics must be finite with positive focal lengths")

    @property
    def rig_frame_id(self) -> str:
        return self.T_rig_from_virtual.to_frame


@dataclass(frozen=True, slots=True)
class PerspectiveCrop:
    """A physical (never implicitly mirrored) virtual-pinhole hand crop."""

    crop_id: str
    source_view_id: str
    source_detection_id: str
    source_pixel_space_id: str
    virtual_camera: VirtualCamera
    crop_policy_id: str
    image: Any
    valid_mask: Any

    def __post_init__(self) -> None:
        for label, value in (
            ("crop_id", self.crop_id),
            ("source_view_id", self.source_view_id),
            ("source_detection_id", self.source_detection_id),
            ("source_pixel_space_id", self.source_pixel_space_id),
            ("crop_policy_id", self.crop_policy_id),
        ):
            _require_text(value, label)

    @property
    def pixel_space(self) -> PixelSpace:
        return self.virtual_camera.pixel_space


@dataclass(frozen=True, slots=True)
class NativeViewEvidence:
    """Backend-native per-view evidence before canonical landmark mapping.

    Arrays have shapes ``mean_uv[J,2]``, ``covariance_uv_px2[J,2,2]``, and
    ``visibility_probability[J]`` in ``pixel_space_id``. The adapter must undo any model
    resize, letterbox, or mirror before constructing this value.
    """

    evidence_id: str
    crop_id: str
    pixel_space_id: str
    joint_set_id: str
    mean_uv: Any
    covariance_uv_px2: Any
    visibility_probability: Any
    presence_probability: float
    feature_ref: Any | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("evidence_id", self.evidence_id),
            ("crop_id", self.crop_id),
            ("pixel_space_id", self.pixel_space_id),
            ("joint_set_id", self.joint_set_id),
        ):
            _require_text(value, label)
        _require_probability(self.presence_probability, "presence_probability")
        mean_shape = _array_shape(self.mean_uv)
        if mean_shape is None or len(mean_shape) != 2 or mean_shape[1] != 2:
            raise ValueError(f"mean_uv must have shape [J,2], got {mean_shape}")
        joint_count = mean_shape[0]
        _require_shape(
            self.covariance_uv_px2,
            (joint_count, 2, 2),
            "covariance_uv_px2",
        )
        _require_shape(
            self.visibility_probability,
            (joint_count,),
            "visibility_probability",
        )


@dataclass(frozen=True, slots=True)
class CanonicalViewEvidence:
    """Exactly 21 mapped landmarks, and the only evidence type accepted by fusion.

    Arrays have shapes ``mean_uv[21,2]``, ``covariance_uv_px2[21,2,2]``, and
    ``visibility_probability[21]``. A ``MISSING`` landmark is represented by invalid/NaN
    coordinates, unbounded covariance, and zero visibility; fusion must not consume it as
    a measurement.
    """

    evidence_id: str
    source_evidence_id: str
    crop_id: str
    pixel_space_id: str
    schema_version: str
    mapping_id: str
    mapping_quality: tuple[MappingQuality, ...]
    mean_uv: Any
    covariance_uv_px2: Any
    visibility_probability: Any
    presence_probability: float
    feature_ref: Any | None = None

    def __post_init__(self) -> None:
        if self.schema_version != FHP21.version:
            raise ValueError(f"fusion evidence must use {FHP21.version}")
        _require_landmark_count(self.mapping_quality, "mapping_quality")
        if any(not isinstance(quality, MappingQuality) for quality in self.mapping_quality):
            raise TypeError("mapping_quality entries must be MappingQuality values")
        _require_probability(self.presence_probability, "presence_probability")
        _require_shape(self.mean_uv, (LANDMARK_COUNT, 2), "mean_uv")
        _require_shape(
            self.covariance_uv_px2,
            (LANDMARK_COUNT, 2, 2),
            "covariance_uv_px2",
        )
        _require_shape(
            self.visibility_probability,
            (LANDMARK_COUNT,),
            "visibility_probability",
        )
        for label, value in (
            ("evidence_id", self.evidence_id),
            ("source_evidence_id", self.source_evidence_id),
            ("crop_id", self.crop_id),
            ("pixel_space_id", self.pixel_space_id),
            ("mapping_id", self.mapping_id),
        ):
            _require_text(value, label)


@dataclass(frozen=True, slots=True)
class MultiViewHandMember:
    """One fully linked per-view candidate within an association group."""

    source_view: ImageView
    detection: Detection2D
    crop: PerspectiveCrop
    evidence: CanonicalViewEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, CanonicalViewEvidence):
            raise TypeError("fusion members require CanonicalViewEvidence after mapping")
        if self.detection.view_id != self.source_view.view_id:
            raise ValueError("detection and source view do not match")
        if self.detection.pixel_space_id != self.source_view.pixel_space.pixel_space_id:
            raise ValueError("detection is not expressed in its source native pixel space")
        if self.crop.source_view_id != self.source_view.view_id:
            raise ValueError("crop and source view do not match")
        if self.crop.source_detection_id != self.detection.detection_id:
            raise ValueError("crop and source detection do not match")
        if self.crop.source_pixel_space_id != self.source_view.pixel_space.pixel_space_id:
            raise ValueError("crop source pixel space does not match the source view")
        if self.crop.virtual_camera.source_camera_id != self.source_view.camera_id:
            raise ValueError("crop virtual camera has the wrong source camera")
        if self.crop.virtual_camera.calibration_id != self.source_view.calibration_id:
            raise ValueError("crop and source view calibrations do not match")
        if self.evidence.crop_id != self.crop.crop_id:
            raise ValueError("evidence and crop do not match")
        if self.evidence.pixel_space_id != self.crop.pixel_space.pixel_space_id:
            raise ValueError("evidence is not expressed in physical crop pixels")


@dataclass(frozen=True, slots=True)
class MultiViewHandGroup:
    """Self-contained, variable-view input to calibrated fusion."""

    group_id: str
    track_id: str
    frame_set: FrameSet
    members: tuple[MultiViewHandMember, ...]
    association_probability: float | None = None
    handedness: HandednessProbabilities | None = None

    def __post_init__(self) -> None:
        _require_text(self.group_id, "group_id")
        _require_text(self.track_id, "track_id")
        if not self.members:
            raise ValueError("a MultiViewHandGroup must contain at least one member")
        if self.association_probability is not None:
            _require_probability(self.association_probability, "association_probability")
        if self.handedness is not None and not isinstance(self.handedness, HandednessProbabilities):
            raise TypeError("handedness must use HandednessProbabilities")
        camera_ids = [member.source_view.camera_id for member in self.members]
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("an association group may contain at most one member per camera")
        frame_views = {view.view_id: view for view in self.frame_set.views}
        for member in self.members:
            expected = frame_views.get(member.source_view.view_id)
            if expected is None:
                raise ValueError("group member source view is absent from its FrameSet")
            if (
                expected.camera_id != member.source_view.camera_id
                or expected.timestamp_ns != member.source_view.timestamp_ns
                or expected.calibration_id != member.source_view.calibration_id
                or expected.pixel_space != member.source_view.pixel_space
            ):
                raise ValueError("group member source metadata conflicts with its FrameSet")
            if member.crop.virtual_camera.rig_frame_id != self.frame_set.rig_frame.frame_id:
                raise ValueError("virtual camera and FrameSet rig frame do not match")


@dataclass(frozen=True, slots=True)
class ViewResidual:
    view_id: str
    error_px: float

    def __post_init__(self) -> None:
        _require_text(self.view_id, "view_id")
        if not math.isfinite(self.error_px) or self.error_px < 0:
            raise ValueError("reprojection error must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class SpatialObservation:
    """Immutable raw output of fusion; it is never overwritten by a refiner.

    Numerical shapes are ``landmarks_xyz_m[21,3]``, ``covariance_m2[21,3,3]``,
    ``visibility_probability[21]``, and optional ``confidence_probability[21]``. When
    present, confidence is the probability that Euclidean position error is within
    ``confidence_radius_m``; both confidence fields are absent if it is not calibrated.
    Fusion may use a prior for initialization/regularization, but a landmark without a
    current measurement must remain invalid rather than being labelled measured.
    """

    observation_id: str
    group_id: str
    sequence_id: str
    track_id: str
    timestamp_ns: int
    schema_version: str
    calibration_id: str
    output_frame: CoordinateFrame3D
    handedness: HandednessProbabilities
    landmarks_xyz_m: Any
    covariance_m2: Any
    validity: tuple[Validity, ...]
    evidence_source: tuple[EvidenceSource, ...]
    visibility_probability: Any
    confidence_probability: Any | None
    confidence_radius_m: float | None
    support_view_ids: tuple[tuple[str, ...], ...]
    reprojection_residuals: tuple[tuple[ViewResidual, ...], ...]
    mapping_ids: tuple[str, ...]
    backend_provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != FHP21.version:
            raise ValueError(f"SpatialObservation must use {FHP21.version}")
        if self.output_frame.kind not in (FrameKind.CAMERA, FrameKind.RIG):
            raise ValueError("current outputs are limited to CAMERA or RIG frames")
        if not isinstance(self.handedness, HandednessProbabilities):
            raise TypeError("handedness must use HandednessProbabilities")
        for label, values in (
            ("validity", self.validity),
            ("evidence_source", self.evidence_source),
            ("support_view_ids", self.support_view_ids),
            ("reprojection_residuals", self.reprojection_residuals),
        ):
            _require_landmark_count(values, label)
        if any(not isinstance(value, Validity) for value in self.validity):
            raise TypeError("validity entries must be Validity values")
        if any(not isinstance(value, EvidenceSource) for value in self.evidence_source):
            raise TypeError("evidence_source entries must be EvidenceSource values")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        _require_shape(self.landmarks_xyz_m, (LANDMARK_COUNT, 3), "landmarks_xyz_m")
        _require_shape(self.covariance_m2, (LANDMARK_COUNT, 3, 3), "covariance_m2")
        _require_shape(
            self.visibility_probability,
            (LANDMARK_COUNT,),
            "visibility_probability",
        )
        if (self.confidence_probability is None) != (self.confidence_radius_m is None):
            raise ValueError("confidence probability and radius must be present together")
        if self.confidence_probability is not None and self.confidence_radius_m is not None:
            _require_shape(
                self.confidence_probability,
                (LANDMARK_COUNT,),
                "confidence_probability",
            )
            if not math.isfinite(self.confidence_radius_m) or self.confidence_radius_m <= 0:
                raise ValueError("confidence_radius_m must be finite and positive")
        if any(
            source is EvidenceSource.NONE and validity is Validity.VALID
            for source, validity in zip(self.evidence_source, self.validity, strict=True)
        ):
            raise ValueError("a raw valid observation must have current image evidence")
        if not self.mapping_ids or not self.backend_provenance:
            raise ValueError("SpatialObservation must preserve mapping and backend provenance")
        if any(not value for value in (*self.mapping_ids, *self.backend_provenance)):
            raise ValueError("mapping and backend provenance identifiers must be non-empty")
        for label, value in (
            ("observation_id", self.observation_id),
            ("group_id", self.group_id),
            ("sequence_id", self.sequence_id),
            ("track_id", self.track_id),
            ("calibration_id", self.calibration_id),
        ):
            _require_text(value, label)

    @property
    def stage(self) -> EstimateStage:
        return EstimateStage.RAW_FUSION


@dataclass(frozen=True, slots=True)
class PoseEstimate:
    """A new refined/predicted value linked back to immutable raw observations."""

    estimate_id: str
    source_observation_ids: tuple[str, ...]
    sequence_id: str
    track_id: str
    timestamp_ns: int
    schema_version: str
    calibration_id: str
    output_frame: CoordinateFrame3D
    handedness: HandednessProbabilities
    stage: EstimateStage
    kind: tuple[EstimateKind, ...]
    landmarks_xyz_m: Any
    covariance_m2: Any
    validity: tuple[Validity, ...]
    evidence_source: tuple[EvidenceSource, ...]
    visibility_probability: Any
    confidence_probability: Any | None
    confidence_radius_m: float | None
    support_view_ids: tuple[tuple[str, ...], ...]
    reprojection_residuals: tuple[tuple[ViewResidual, ...], ...]
    mapping_ids: tuple[str, ...]
    backend_provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != FHP21.version:
            raise ValueError(f"PoseEstimate must use {FHP21.version}")
        if self.output_frame.kind not in (FrameKind.CAMERA, FrameKind.RIG):
            raise ValueError("current outputs are limited to CAMERA or RIG frames")
        if not isinstance(self.handedness, HandednessProbabilities):
            raise TypeError("handedness must use HandednessProbabilities")
        if not isinstance(self.stage, EstimateStage):
            raise TypeError("stage must be an EstimateStage value")
        if self.stage is EstimateStage.RAW_FUSION:
            raise ValueError("raw fusion values are SpatialObservation, not PoseEstimate")
        for label, values in (
            ("kind", self.kind),
            ("validity", self.validity),
            ("evidence_source", self.evidence_source),
            ("support_view_ids", self.support_view_ids),
            ("reprojection_residuals", self.reprojection_residuals),
        ):
            _require_landmark_count(values, label)
        if any(not isinstance(value, EstimateKind) for value in self.kind):
            raise TypeError("kind entries must be EstimateKind values")
        if any(not isinstance(value, Validity) for value in self.validity):
            raise TypeError("validity entries must be Validity values")
        if any(not isinstance(value, EvidenceSource) for value in self.evidence_source):
            raise TypeError("evidence_source entries must be EvidenceSource values")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        _require_shape(self.landmarks_xyz_m, (LANDMARK_COUNT, 3), "landmarks_xyz_m")
        _require_shape(self.covariance_m2, (LANDMARK_COUNT, 3, 3), "covariance_m2")
        _require_shape(
            self.visibility_probability,
            (LANDMARK_COUNT,),
            "visibility_probability",
        )
        if (self.confidence_probability is None) != (self.confidence_radius_m is None):
            raise ValueError("confidence probability and radius must be present together")
        if self.confidence_probability is not None and self.confidence_radius_m is not None:
            _require_shape(
                self.confidence_probability,
                (LANDMARK_COUNT,),
                "confidence_probability",
            )
            if not math.isfinite(self.confidence_radius_m) or self.confidence_radius_m <= 0:
                raise ValueError("confidence_radius_m must be finite and positive")
        for label, value in (
            ("estimate_id", self.estimate_id),
            ("sequence_id", self.sequence_id),
            ("track_id", self.track_id),
            ("calibration_id", self.calibration_id),
        ):
            _require_text(value, label)
        if not self.source_observation_ids:
            raise ValueError("PoseEstimate must reference at least one raw observation")
        if not self.mapping_ids or not self.backend_provenance:
            raise ValueError("PoseEstimate must preserve mapping and backend provenance")
        if any(not value for value in (*self.mapping_ids, *self.backend_provenance)):
            raise ValueError("mapping and backend provenance identifiers must be non-empty")


@dataclass(frozen=True, slots=True)
class TrackMetadata:
    sequence_id: str
    track_id: str
    schema_version: str
    calibration_id: str
    output_frame: CoordinateFrame3D

    def __post_init__(self) -> None:
        if self.schema_version != FHP21.version:
            raise ValueError(f"temporal tracks must use {FHP21.version}")
        if self.output_frame.kind not in (FrameKind.CAMERA, FrameKind.RIG):
            raise ValueError("current outputs are limited to CAMERA or RIG frames")
        for label, value in (
            ("sequence_id", self.sequence_id),
            ("track_id", self.track_id),
            ("calibration_id", self.calibration_id),
        ):
            _require_text(value, label)


@dataclass(frozen=True, slots=True)
class TemporalCapabilities:
    """Latency and evidence requirements advertised by a temporal backend."""

    mode: TemporalMode
    latency_frames: int | None
    latency_ns: int | None
    uses_per_view_reprojection: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.mode, TemporalMode):
            raise TypeError("mode must be an explicit TemporalMode value")
        if self.mode is TemporalMode.CAUSAL:
            if self.latency_frames != 0 or self.latency_ns != 0:
                raise ValueError("a causal refiner must declare zero latency")
        elif self.mode is TemporalMode.FIXED_LAG:
            if self.latency_frames is None and self.latency_ns is None:
                raise ValueError("a fixed-lag refiner must declare a finite lag")
        elif self.latency_frames is not None or self.latency_ns is not None:
            raise ValueError("an offline refiner has unbounded latency until flush")
        if self.latency_frames is not None and self.latency_frames < 0:
            raise ValueError("latency_frames must be non-negative")
        if self.latency_ns is not None and self.latency_ns < 0:
            raise ValueError("latency_ns must be non-negative")


class Detector(Protocol):
    manifest: BackendManifest

    def infer(self, views: Sequence[ImageView]) -> list[Detection2D]: ...


class VirtualCropper(Protocol):
    def crop(self, view: ImageView, detection: Detection2D) -> PerspectiveCrop: ...


class PoseEvidenceBackend(Protocol):
    manifest: BackendManifest

    def infer(self, crops: Sequence[PerspectiveCrop]) -> list[NativeViewEvidence]: ...


class LandmarkMapper(Protocol):
    mapping_record: LandmarkMappingRecord

    def map(self, evidence: NativeViewEvidence) -> CanonicalViewEvidence: ...


class CrossViewAssociator(Protocol):
    def associate(
        self,
        frame_set: FrameSet,
        candidates: Sequence[MultiViewHandMember],
    ) -> list[MultiViewHandGroup]: ...


class FusionBackend(Protocol):
    manifest: BackendManifest

    def fuse(
        self,
        group: MultiViewHandGroup,
        prior: PoseEstimate | None = None,
    ) -> SpatialObservation: ...


class KinematicRefiner(Protocol):
    manifest: BackendManifest

    def refine(self, observation: SpatialObservation) -> PoseEstimate: ...


class TemporalRefiner(Protocol):
    """Stateful, single-track temporal refinement contract.

    One instance owns at most one open track. ``open_track`` is required before ``push``.
    Inputs are either immutable raw observations or kinematic estimates; they must match
    the open metadata and have strictly increasing timestamps. Every accepted input must
    yield exactly one estimate, returned once by ``push``, ``flush``, or ``reset``. Causal
    implementations emit immediately, fixed-lag
    implementations may emit older timestamps, and offline implementations emit on
    ``flush``/``reset``. Both terminal methods emit pending values in timestamp order and
    close the track; callers must call ``open_track`` before re-use. A reset boundary must
    not share state across track-ID switches, scene cuts, calibration changes, or long gaps.
    """

    manifest: BackendManifest
    capabilities: TemporalCapabilities

    def open_track(self, metadata: TrackMetadata) -> None: ...

    def push(self, value: SpatialObservation | PoseEstimate) -> list[PoseEstimate]: ...

    def flush(self) -> list[PoseEstimate]: ...

    def reset(self, reason: ResetReason) -> list[PoseEstimate]: ...
