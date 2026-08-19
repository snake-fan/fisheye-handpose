"""Generated from contracts/project-contract-v1.json; do not edit."""

from __future__ import annotations

PROJECT_CONTRACT_SCHEMA = "fisheye-handpose/project-contract/v1"
PROJECT_CONTRACT_VERSION = 1

TRACE_STAGE_VALUES = (
    "SYSTEM",
    "DISCOVERY",
    "CALIBRATION",
    "DECODE",
    "SYNCHRONIZATION",
    "RECTIFICATION",
    "DETECTION",
    "POSE_2D",
    "CROSS_VIEW_ASSOCIATION",
    "RAW_FUSION",
    "KINEMATIC_REFINEMENT",
    "TEMPORAL_REFINEMENT",
    "QA",
    "EXPORT",
)

TRACE_STATUS_VALUES = (
    "STARTED",
    "SUCCEEDED",
    "FAILED",
    "WARNING",
    "SKIPPED",
)

RUN_STATUS_VALUES = (
    "ACTIVE",
    "COMPLETED",
    "FAILED",
)

FHP21_SCHEMA_ID = "fhp21/v1"
FHP21_NAMES = (
    "wrist_center",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
)
FHP21_PARENTS = (
    -1,
    0,
    1,
    2,
    3,
    0,
    5,
    6,
    7,
    0,
    9,
    10,
    11,
    0,
    13,
    14,
    15,
    0,
    17,
    18,
    19,
)
FHP21_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)
FHP21_TIP_INDICES = (
    4,
    8,
    12,
    16,
    20,
)

AUDIT_SCHEMA = "fisheye-handpose/audit/v1"
BASELINE_METRICS_SCHEMA = "fisheye-handpose/baseline-metrics/v1"
DOCTOR_SCHEMA = "fisheye-handpose/doctor/v1"
FHP21_OUTPUT_SCHEMA = "fisheye-handpose/fhp21-output/v1"
H20_ENVIRONMENT_SCHEMA = "fisheye-handpose/h20-environment/v1"
H20_EXECUTOR_SCHEMA = "fisheye-handpose/h20-executor/v1"
H20_WORKER_EVENT_SCHEMA = "fisheye-handpose/h20-worker-event/v1"
H20_WORKER_MANIFEST_SCHEMA = "fisheye-handpose/h20-worker-manifest/v1"
H20_WORKER_REQUEST_SCHEMA = "fisheye-handpose/h20-worker-request/v1"
H20_WORKER_SUMMARY_SCHEMA = "fisheye-handpose/h20-worker-summary/v1"
MANO_ASSETS_SCHEMA = "fisheye-handpose/mano-assets/v1"
MANO_SMOKE_SCHEMA = "fisheye-handpose/mano-smoke/v1"
MODEL_ASSETS_SCHEMA = "fisheye-handpose/model-assets/v1"
MP4_EXPORT_SCHEMA = "fisheye-handpose/mp4-export/v1"
OVERLAY_VIDEO_SCHEMA = "fisheye-handpose/overlay-video/v1"
OVERLAY_VIDEO_TIMELINE_SCHEMA = "fisheye-handpose/overlay-video-timeline/v1"
RTMPOSE_SMOKE_SCHEMA = "fisheye-handpose/rtmpose-smoke/v1"
RUN_MANIFEST_SCHEMA = "fisheye-handpose/run-manifest/v1"
RUN_SUMMARY_SCHEMA = "fisheye-handpose/run-summary/v1"
TRACE_RECORD_SCHEMA = "fisheye-handpose/trace-record/v1"

SCHEMA_IDS = {
    "AUDIT": AUDIT_SCHEMA,
    "BASELINE_METRICS": BASELINE_METRICS_SCHEMA,
    "DOCTOR": DOCTOR_SCHEMA,
    "FHP21_OUTPUT": FHP21_OUTPUT_SCHEMA,
    "H20_ENVIRONMENT": H20_ENVIRONMENT_SCHEMA,
    "H20_EXECUTOR": H20_EXECUTOR_SCHEMA,
    "H20_WORKER_EVENT": H20_WORKER_EVENT_SCHEMA,
    "H20_WORKER_MANIFEST": H20_WORKER_MANIFEST_SCHEMA,
    "H20_WORKER_REQUEST": H20_WORKER_REQUEST_SCHEMA,
    "H20_WORKER_SUMMARY": H20_WORKER_SUMMARY_SCHEMA,
    "MANO_ASSETS": MANO_ASSETS_SCHEMA,
    "MANO_SMOKE": MANO_SMOKE_SCHEMA,
    "MODEL_ASSETS": MODEL_ASSETS_SCHEMA,
    "MP4_EXPORT": MP4_EXPORT_SCHEMA,
    "OVERLAY_VIDEO": OVERLAY_VIDEO_SCHEMA,
    "OVERLAY_VIDEO_TIMELINE": OVERLAY_VIDEO_TIMELINE_SCHEMA,
    "RTMPOSE_SMOKE": RTMPOSE_SMOKE_SCHEMA,
    "RUN_MANIFEST": RUN_MANIFEST_SCHEMA,
    "RUN_SUMMARY": RUN_SUMMARY_SCHEMA,
    "TRACE_RECORD": TRACE_RECORD_SCHEMA,
}

FHP21_IDENTITY_MAPPING_ID = "fhp21/v1:identity"
MANO_FHP21_MAPPING_ID = "mano-v1.2-j16-tips-to-fhp21/v1"
RTMPOSE_FHP21_MAPPING_ID = "rtmpose-hand5-native21-to-fhp21/v1"

MAPPING_IDS = {
    "FHP21_IDENTITY": FHP21_IDENTITY_MAPPING_ID,
    "MANO_FHP21": MANO_FHP21_MAPPING_ID,
    "RTMPOSE_FHP21": RTMPOSE_FHP21_MAPPING_ID,
}

__all__ = [
    "AUDIT_SCHEMA",
    "BASELINE_METRICS_SCHEMA",
    "DOCTOR_SCHEMA",
    "FHP21_EDGES",
    "FHP21_IDENTITY_MAPPING_ID",
    "FHP21_NAMES",
    "FHP21_OUTPUT_SCHEMA",
    "FHP21_PARENTS",
    "FHP21_SCHEMA_ID",
    "FHP21_TIP_INDICES",
    "H20_ENVIRONMENT_SCHEMA",
    "H20_EXECUTOR_SCHEMA",
    "H20_WORKER_EVENT_SCHEMA",
    "H20_WORKER_MANIFEST_SCHEMA",
    "H20_WORKER_REQUEST_SCHEMA",
    "H20_WORKER_SUMMARY_SCHEMA",
    "MANO_ASSETS_SCHEMA",
    "MANO_FHP21_MAPPING_ID",
    "MANO_SMOKE_SCHEMA",
    "MAPPING_IDS",
    "MODEL_ASSETS_SCHEMA",
    "MP4_EXPORT_SCHEMA",
    "OVERLAY_VIDEO_SCHEMA",
    "OVERLAY_VIDEO_TIMELINE_SCHEMA",
    "PROJECT_CONTRACT_SCHEMA",
    "PROJECT_CONTRACT_VERSION",
    "RTMPOSE_FHP21_MAPPING_ID",
    "RTMPOSE_SMOKE_SCHEMA",
    "RUN_MANIFEST_SCHEMA",
    "RUN_STATUS_VALUES",
    "RUN_SUMMARY_SCHEMA",
    "SCHEMA_IDS",
    "TRACE_RECORD_SCHEMA",
    "TRACE_STAGE_VALUES",
    "TRACE_STATUS_VALUES",
]
