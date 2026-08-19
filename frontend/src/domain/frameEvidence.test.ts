import { expect, test } from "vitest";

import type { TraceRecord } from "../api/types";
import { createFrameEvidence, outputAvailableFor } from "./frameEvidence";


test("raw frame records are normalized and indexed once across compatibility vocabularies", () => {
  const records: TraceRecord[] = [
    {
      record_id: "pose-left",
      stage: "POSE_2D",
      status: "SUCCEEDED",
      event: "view_keypoints_inferred",
      payload: {
        view: "LEFT",
        track: "track-0001",
        output_status: "PRODUCED",
      },
      blobs: [{ role: "source_left", relative_path: "blobs/source-left.jpg" }],
      artifacts: [{ role: "overlay_left", path: "artifacts/overlay-left.jpg" }],
    },
    {
      record_id: "raw-rejected",
      stage: "RAW_FUSION",
      status: "WARNING",
      event: "raw_hand_gate_not_produced",
      payload: {
        output_status: "NOT_PRODUCED",
        reason: "",
        hand_reason: "INSUFFICIENT_PALM_SUPPORT",
        selection: { gate: { reason: "LOWER_PRIORITY_GATE_REASON" } },
      },
    },
  ];

  const evidence = createFrameEvidence(records);
  const pose = evidence.records[0];
  const rejected = evidence.records[1];

  expect(evidence.recordsForStage("POSE_2D")[0]).toBe(pose);
  expect(evidence.recordsForView("left")[0]).toBe(pose);
  expect(evidence.recordsForTrack("track-0001")[0]).toBe(pose);
  expect(pose.payload).toMatchObject({ view_id: "left", track_id: "track-0001" });
  expect(evidence.artifactsForRole("source_left")[0]?.record).toBe(pose);
  expect(pose.artifactRefs.map((artifact) => artifact.role)).toEqual([
    "source_left",
    "overlay_left",
  ]);
  expect(pose.outputStatus).toBe("PRODUCED");
  expect(rejected.outputStatus).toBe("NOT_PRODUCED");
  expect(rejected.failureReason).toBe("INSUFFICIENT_PALM_SUPPORT");
});


test("legacy status and selection fields produce stable output and failure semantics", () => {
  const evidence = createFrameEvidence([
    {
      stage: "EXPORT",
      status: "SUCCEEDED",
      payload: null as unknown as Record<string, unknown>,
    },
    {
      stage: "KINEMATIC_REFINEMENT",
      status: "FAILED",
      payload: { selection: { gate: { reason: "ROBUST_GATE_REJECTED" } } },
    },
    {
      stage: "TEMPORAL_REFINEMENT",
      status: "WARNING",
      event: "temporal_pending",
      payload: {},
    },
    {
      stage: "RAW_FUSION",
      payload: {
        projected_keypoints_uv: { left: [[12, 34]], right: [[10, 34]] },
      },
    },
  ]);

  expect(evidence.records.map((record) => record.outputStatus)).toEqual([
    "PRODUCED",
    "NOT_PRODUCED",
    "UNKNOWN",
    "UNKNOWN",
  ]);
  expect(outputAvailableFor(evidence.records[2])).toBe(true);
  expect(outputAvailableFor(evidence.records[3])).toBe(true);
  expect(evidence.records[0].payload).toEqual({});
  expect(evidence.records[1].failureReason).toBe("ROBUST_GATE_REJECTED");
  expect(evidence.records[2].failureReason).toBe("");
});
