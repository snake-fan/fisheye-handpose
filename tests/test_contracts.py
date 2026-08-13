from __future__ import annotations

from dataclasses import replace
from typing import get_type_hints

import pytest

from fisheye_handpose.calibration import Transform3D
from fisheye_handpose.contracts import (
    CanonicalViewEvidence,
    CoordinateFrame3D,
    Detection2D,
    EstimateKind,
    EstimateStage,
    EvidenceSource,
    FrameKind,
    FrameSet,
    FusionBackend,
    HandednessProbabilities,
    ImageView,
    MultiViewHandGroup,
    MultiViewHandMember,
    NativeViewEvidence,
    PerspectiveCrop,
    PixelSpace,
    PixelSpaceKind,
    PoseEstimate,
    SpatialObservation,
    TemporalCapabilities,
    TemporalMode,
    TemporalRefiner,
    Validity,
    ViewResidual,
    VirtualCamera,
)
from fisheye_handpose.joints import (
    FHP21,
    FHP21_IDENTITY_MAPPING,
    LandmarkMappingEntry,
    LandmarkMappingRecord,
    MappingQuality,
)

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
ZERO_2D = tuple((0.0, 0.0) for _ in range(21))
COV_2D = tuple(((1.0, 0.0), (0.0, 1.0)) for _ in range(21))
ZERO_3D = tuple((0.0, 0.0, 0.5) for _ in range(21))
COV_3D = tuple(((1e-4, 0.0, 0.0), (0.0, 1e-4, 0.0), (0.0, 0.0, 1e-4)) for _ in range(21))
PROBABILITIES = tuple(0.9 for _ in range(21))


def make_member(*, view_id: str = "view-0", camera_id: str = "cam-0") -> MultiViewHandMember:
    raw_space = PixelSpace(
        pixel_space_id=f"raw:{camera_id}",
        kind=PixelSpaceKind.RAW_DISTORTED,
        image_size_wh=(1280, 800),
        calibration_id="cal-1",
        source_camera_id=camera_id,
    )
    view = ImageView(
        view_id=view_id,
        camera_id=camera_id,
        timestamp_ns=1_000,
        calibration_id="cal-1",
        pixel_space=raw_space,
        image=object(),
    )
    detection = Detection2D(
        detection_id=f"det:{view_id}",
        view_id=view_id,
        bbox_xyxy=(100.0, 100.0, 300.0, 350.0),
        detector_score=0.8,
        pixel_space_id=raw_space.pixel_space_id,
    )
    crop_space = PixelSpace(
        pixel_space_id=f"crop:{view_id}",
        kind=PixelSpaceKind.VIRTUAL_PINHOLE,
        image_size_wh=(256, 256),
        calibration_id="cal-1",
        source_camera_id=camera_id,
    )
    virtual_camera = VirtualCamera(
        virtual_camera_id=f"virtual:{view_id}",
        pixel_space=crop_space,
        source_camera_id=camera_id,
        calibration_id="cal-1",
        intrinsics=((200.0, 0.0, 127.5), (0.0, 200.0, 127.5), (0.0, 0.0, 1.0)),
        T_rig_from_virtual=Transform3D(
            f"virtual:{view_id}",
            "rig-0",
            IDENTITY,
            (0.0, 0.0, 0.0),
        ),
    )
    crop = PerspectiveCrop(
        crop_id=f"crop-object:{view_id}",
        source_view_id=view_id,
        source_detection_id=detection.detection_id,
        source_pixel_space_id=raw_space.pixel_space_id,
        virtual_camera=virtual_camera,
        crop_policy_id="hand-centred/v1",
        image=object(),
        valid_mask=object(),
    )
    evidence = CanonicalViewEvidence(
        evidence_id=f"canonical:{view_id}",
        source_evidence_id=f"native:{view_id}",
        crop_id=crop.crop_id,
        pixel_space_id=crop_space.pixel_space_id,
        schema_version=FHP21.version,
        mapping_id=FHP21_IDENTITY_MAPPING.mapping_id,
        mapping_quality=FHP21_IDENTITY_MAPPING.qualities,
        mean_uv=ZERO_2D,
        covariance_uv_px2=COV_2D,
        visibility_probability=PROBABILITIES,
        presence_probability=0.95,
    )
    return MultiViewHandMember(view, detection, crop, evidence)


def make_group() -> MultiViewHandGroup:
    member = make_member()
    frame_set = FrameSet(
        frame_set_id="frames-0",
        sequence_id="sequence-0",
        timestamp_ns=member.source_view.timestamp_ns,
        calibration_id="cal-1",
        rig_frame=CoordinateFrame3D(
            "rig-0",
            FrameKind.RIG,
            "opencv_x_right_y_down_z_forward",
        ),
        sync_tolerance_ns=1_000,
        views=(member.source_view,),
    )
    return MultiViewHandGroup(
        group_id="group-0",
        track_id="track-0",
        frame_set=frame_set,
        members=(member,),
        handedness=HandednessProbabilities(0.0, 0.0, 1.0),
    )


def make_observation() -> SpatialObservation:
    group = make_group()
    return SpatialObservation(
        observation_id="observation-0",
        group_id=group.group_id,
        sequence_id=group.frame_set.sequence_id,
        track_id=group.track_id,
        timestamp_ns=group.frame_set.timestamp_ns,
        schema_version=FHP21.version,
        calibration_id=group.frame_set.calibration_id,
        output_frame=group.frame_set.rig_frame,
        handedness=group.handedness or HandednessProbabilities(0.0, 0.0, 1.0),
        landmarks_xyz_m=ZERO_3D,
        covariance_m2=COV_3D,
        validity=(Validity.VALID,) * 21,
        evidence_source=(EvidenceSource.MONOCULAR,) * 21,
        visibility_probability=PROBABILITIES,
        confidence_probability=tuple(0.8 for _ in range(21)),
        confidence_radius_m=0.02,
        support_view_ids=(("view-0",),) * 21,
        reprojection_residuals=((ViewResidual("view-0", 0.5),),) * 21,
        mapping_ids=(FHP21_IDENTITY_MAPPING.mapping_id,),
        backend_provenance=("geometric-fuser@1",),
    )


def test_fusion_protocol_requires_self_contained_group():
    hints = get_type_hints(FusionBackend.fuse)
    assert hints["group"] is MultiViewHandGroup
    group = make_group()
    assert group.frame_set.calibration_id == "cal-1"
    assert group.members[0].crop.virtual_camera.rig_frame_id == "rig-0"
    assert group.members[0].source_view.timestamp_ns == 1_000


def test_group_rejects_more_than_one_candidate_from_one_camera():
    group = make_group()
    with pytest.raises(ValueError, match="one member per camera"):
        replace(group, members=(group.members[0], group.members[0]))


def test_fusion_member_rejects_backend_native_evidence():
    member = make_member()
    native = NativeViewEvidence(
        evidence_id="native",
        crop_id=member.crop.crop_id,
        pixel_space_id=member.crop.pixel_space.pixel_space_id,
        joint_set_id="backend/native-v1",
        mean_uv=((0.0, 0.0),),
        covariance_uv_px2=(((1.0, 0.0), (0.0, 1.0)),),
        visibility_probability=(1.0,),
        presence_probability=1.0,
    )
    with pytest.raises(TypeError, match="CanonicalViewEvidence"):
        MultiViewHandMember(member.source_view, member.detection, member.crop, native)  # type: ignore[arg-type]


def test_mapping_record_must_cover_all_canonical_targets():
    entry = LandmarkMappingEntry(
        target_index=1,
        quality=MappingQuality.EXACT,
        source_indices=(0,),
        source_construction_id=FHP21.definitions[1].construction_id,
        method="identity",
        source_definition="same normative definition",
    )
    with pytest.raises(ValueError, match="cover all 21"):
        LandmarkMappingRecord(
            mapping_id="bad",
            source_joint_set_id="source/v1",
            target_schema_version=FHP21.version,
            source_landmark_names=("wrist",),
            entries=(entry,),
            provenance="test",
        )


def test_identity_mapping_preserves_normative_definitions():
    assert len(FHP21_IDENTITY_MAPPING.entries) == 21
    assert all(entry.quality is MappingQuality.EXACT for entry in FHP21_IDENTITY_MAPPING.entries)
    assert tuple(entry.source_definition for entry in FHP21_IDENTITY_MAPPING.entries) == tuple(
        definition.definition for definition in FHP21.definitions
    )
    assert tuple(entry.source_construction_id for entry in FHP21_IDENTITY_MAPPING.entries) == tuple(
        definition.construction_id for definition in FHP21.definitions
    )
    assert "operational realization" in FHP21.definitions[0].definition
    assert "operational realization" in FHP21.definitions[4].definition


def test_exact_mapping_requires_same_operational_construction_version():
    entries = list(FHP21_IDENTITY_MAPPING.entries)
    entries[0] = replace(entries[0], source_construction_id="another-convention/v1:wrist")
    with pytest.raises(ValueError, match="operational construction/version"):
        LandmarkMappingRecord(
            mapping_id="invalid-exact",
            source_joint_set_id=FHP21.version,
            target_schema_version=FHP21.version,
            source_landmark_names=FHP21.names,
            entries=tuple(entries),
            provenance="test",
        )


def test_raw_observation_cannot_label_prior_only_value_as_measured():
    observation = make_observation()
    assert observation.stage is EstimateStage.RAW_FUSION
    with pytest.raises(ValueError, match="current image evidence"):
        replace(observation, evidence_source=(EvidenceSource.NONE,) * 21)


def test_current_outputs_are_limited_to_camera_or_rig_frames():
    observation = make_observation()
    virtual_frame = CoordinateFrame3D(
        "virtual-camera-0",
        FrameKind.VIRTUAL_CAMERA,
        "opencv_x_right_y_down_z_forward",
    )
    with pytest.raises(ValueError, match="CAMERA or RIG"):
        replace(observation, output_frame=virtual_frame)
    assert not hasattr(FrameKind, "WORLD")


def test_pose_estimate_is_distinct_and_references_raw_observation():
    observation = make_observation()
    estimate = PoseEstimate(
        estimate_id="estimate-0",
        source_observation_ids=(observation.observation_id,),
        sequence_id=observation.sequence_id,
        track_id=observation.track_id,
        timestamp_ns=observation.timestamp_ns,
        schema_version=observation.schema_version,
        calibration_id=observation.calibration_id,
        output_frame=observation.output_frame,
        handedness=observation.handedness,
        stage=EstimateStage.TEMPORAL_REFINEMENT,
        kind=(EstimateKind.REFINED,) * 21,
        landmarks_xyz_m=observation.landmarks_xyz_m,
        covariance_m2=observation.covariance_m2,
        validity=observation.validity,
        evidence_source=observation.evidence_source,
        visibility_probability=observation.visibility_probability,
        confidence_probability=observation.confidence_probability,
        confidence_radius_m=observation.confidence_radius_m,
        support_view_ids=observation.support_view_ids,
        reprojection_residuals=observation.reprojection_residuals,
        mapping_ids=observation.mapping_ids,
        backend_provenance=(*observation.backend_provenance, "temporal-refiner@1"),
    )
    assert estimate.source_observation_ids == (observation.observation_id,)
    with pytest.raises(ValueError, match="SpatialObservation"):
        replace(estimate, stage=EstimateStage.RAW_FUSION)


def test_thresholded_confidence_is_optional_but_atomic():
    observation = make_observation()
    assert replace(observation, confidence_probability=None, confidence_radius_m=None)
    with pytest.raises(ValueError, match="present together"):
        replace(observation, confidence_probability=None)


def test_temporal_capabilities_make_latency_mode_explicit():
    assert TemporalCapabilities(TemporalMode.CAUSAL, 0, 0).latency_frames == 0
    assert TemporalCapabilities(TemporalMode.FIXED_LAG, 3, None).latency_frames == 3
    assert TemporalCapabilities(TemporalMode.OFFLINE, None, None).mode is TemporalMode.OFFLINE
    with pytest.raises(ValueError, match="zero latency"):
        TemporalCapabilities(TemporalMode.CAUSAL, 1, 0)
    with pytest.raises(ValueError, match="finite lag"):
        TemporalCapabilities(TemporalMode.FIXED_LAG, None, None)
    push_type = get_type_hints(TemporalRefiner.push)["value"]
    assert push_type == SpatialObservation | PoseEstimate
