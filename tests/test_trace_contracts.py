from __future__ import annotations

from dataclasses import replace
from math import nan

import pytest

from fisheye_handpose.contracts import (
    EstimateKind,
    EstimateStage,
    PoseEstimate,
    Validity,
)
from fisheye_handpose.trace_contracts import TraceSerializationError, contract_to_trace_payload
from tests.test_contracts import make_observation


def test_raw_observation_serializes_as_fhp21_json_payload() -> None:
    observation = make_observation()

    payload = contract_to_trace_payload(observation)

    assert payload["record_type"] == "SpatialObservation"
    assert payload["stage"] == "RAW_FUSION"
    assert payload["schema_version"] == "fhp21/v1"
    assert payload["output_frame"] == {
        "frame_id": "rig-0",
        "kind": "RIG",
        "axis_convention": "opencv_x_right_y_down_z_forward",
        "length_unit": "m",
    }
    assert payload["validity"] == ["VALID"] * 21
    assert payload["landmarks_xyz_m"][0] == [0.0, 0.0, 0.5]
    assert payload["reprojection_residuals"][0] == [{"view_id": "view-0", "error_px": 0.5}]


def test_invalid_landmark_nonfinite_values_become_json_null() -> None:
    observation = make_observation()
    xyz = list(observation.landmarks_xyz_m)
    covariance = list(observation.covariance_m2)
    visibility = list(observation.visibility_probability)
    confidence = list(observation.confidence_probability)
    xyz[3] = (nan, nan, nan)
    covariance[3] = ((nan, nan, nan),) * 3
    visibility[3] = nan
    confidence[3] = nan
    validity = list(observation.validity)
    validity[3] = Validity.INVALID
    observation = replace(
        observation,
        landmarks_xyz_m=tuple(xyz),
        covariance_m2=tuple(covariance),
        visibility_probability=tuple(visibility),
        confidence_probability=tuple(confidence),
        validity=tuple(validity),
    )

    payload = contract_to_trace_payload(observation)

    assert payload["landmarks_xyz_m"][3] == [None, None, None]
    assert payload["covariance_m2"][3] == [[None, None, None]] * 3
    assert payload["visibility_probability"][3] is None
    assert payload["confidence_probability"][3] is None


def test_valid_landmark_rejects_nonfinite_values() -> None:
    observation = make_observation()
    xyz = list(observation.landmarks_xyz_m)
    xyz[0] = (nan, 0.0, 0.5)

    with pytest.raises(TraceSerializationError, match="valid landmark 0"):
        contract_to_trace_payload(replace(observation, landmarks_xyz_m=tuple(xyz)))


def test_pose_estimate_preserves_raw_link_and_refinement_kind() -> None:
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
        confidence_probability=None,
        confidence_radius_m=None,
        support_view_ids=observation.support_view_ids,
        reprojection_residuals=observation.reprojection_residuals,
        mapping_ids=observation.mapping_ids,
        backend_provenance=(*observation.backend_provenance, "temporal@1"),
    )

    payload = contract_to_trace_payload(estimate)

    assert payload["record_type"] == "PoseEstimate"
    assert payload["source_observation_ids"] == ["observation-0"]
    assert payload["stage"] == "TEMPORAL_REFINEMENT"
    assert payload["kind"] == ["REFINED"] * 21
    assert payload["confidence_probability"] is None
    assert payload["confidence_radius_m"] is None
