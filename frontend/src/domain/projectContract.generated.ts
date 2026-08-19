// Generated from contracts/project-contract-v1.json; do not edit.

export const PROJECT_CONTRACT_SCHEMA = "fisheye-handpose/project-contract/v1" as const;
export const PROJECT_CONTRACT_VERSION = 1 as const;

export const TRACE_STAGES = [
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
] as const;
export type TraceStage = (typeof TRACE_STAGES)[number];

export const TRACE_STATUSES = [
  "STARTED",
  "SUCCEEDED",
  "FAILED",
  "WARNING",
  "SKIPPED",
] as const;
export type TraceStatus = (typeof TRACE_STATUSES)[number];

export const RUN_STATUSES = [
  "ACTIVE",
  "COMPLETED",
  "FAILED",
] as const;
export type RunStatus = (typeof RUN_STATUSES)[number];

export const FHP21_SCHEMA_ID = "fhp21/v1" as const;
export const FHP21_NAMES = [
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
] as const;
export const FHP21_PARENTS = [
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
] as const;
export const FHP21_EDGES = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
] as const;
export const FHP21_TIP_INDICES = [
  4,
  8,
  12,
  16,
  20,
] as const;

export const SCHEMA_IDS = {
  AUDIT: "fisheye-handpose/audit/v1",
  BASELINE_METRICS: "fisheye-handpose/baseline-metrics/v1",
  DOCTOR: "fisheye-handpose/doctor/v1",
  FHP21_OUTPUT: "fisheye-handpose/fhp21-output/v1",
  H20_ENVIRONMENT: "fisheye-handpose/h20-environment/v1",
  H20_EXECUTOR: "fisheye-handpose/h20-executor/v1",
  H20_WORKER_EVENT: "fisheye-handpose/h20-worker-event/v1",
  H20_WORKER_MANIFEST: "fisheye-handpose/h20-worker-manifest/v1",
  H20_WORKER_REQUEST: "fisheye-handpose/h20-worker-request/v1",
  H20_WORKER_SUMMARY: "fisheye-handpose/h20-worker-summary/v1",
  MANO_ASSETS: "fisheye-handpose/mano-assets/v1",
  MANO_SMOKE: "fisheye-handpose/mano-smoke/v1",
  MODEL_ASSETS: "fisheye-handpose/model-assets/v1",
  MP4_EXPORT: "fisheye-handpose/mp4-export/v1",
  OVERLAY_VIDEO: "fisheye-handpose/overlay-video/v1",
  OVERLAY_VIDEO_TIMELINE: "fisheye-handpose/overlay-video-timeline/v1",
  RTMPOSE_SMOKE: "fisheye-handpose/rtmpose-smoke/v1",
  RUN_MANIFEST: "fisheye-handpose/run-manifest/v1",
  RUN_SUMMARY: "fisheye-handpose/run-summary/v1",
  TRACE_RECORD: "fisheye-handpose/trace-record/v1",
} as const;

export const MAPPING_IDS = {
  FHP21_IDENTITY: "fhp21/v1:identity",
  MANO_FHP21: "mano-v1.2-j16-tips-to-fhp21/v1",
  RTMPOSE_FHP21: "rtmpose-hand5-native21-to-fhp21/v1",
} as const;
