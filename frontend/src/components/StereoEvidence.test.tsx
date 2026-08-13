import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";

import type { TraceRecord } from "../api/types";
import { StereoEvidence } from "./StereoEvidence";

test("worker-native instances render both FHP21 overlays and detection boxes", () => {
  const keypoints = Array.from({ length: 21 }, (_, index) => [120 + index * 4, 90 + index * 3]);
  const records: TraceRecord[] = (["left", "right"] as const).flatMap((view) => [
    {
      record_id: `sync-${view}`,
      stage: "SYNCHRONIZATION",
      status: "SUCCEEDED",
      blobs: [{
        role: `source_${view}`,
        relative_path: `blobs/sha256/${view}/frame.png`,
        media_type: "image/png",
      }],
      payload: { view_id: view, image_width: 640, image_height: 480 },
    },
    {
      record_id: `detection-${view}`,
      stage: "DETECTION",
      status: "SUCCEEDED",
      payload: {
        view_id: view,
        instances: [{ candidate_id: `${view}-0`, bbox_xyxy: [90, 65, 250, 245], bbox_score: 0.98 }],
      },
    },
    {
      record_id: `pose-${view}`,
      stage: "POSE_2D",
      status: "SUCCEEDED",
      payload: {
        view_id: view,
        landmark_schema: "fhp21/v1",
        instances: [{
          candidate_id: `${view}-0`,
          bbox_xyxy: [90, 65, 250, 245],
          bbox_score: 0.98,
          keypoints_uv: keypoints,
          keypoint_scores: Array(21).fill(0.96),
        }],
      },
    },
  ]);

  const { container } = render(<StereoEvidence runKey="opaque-run-key" records={records} trackId="" />);

  expect(screen.getAllByText("21 / 21 visible")).toHaveLength(2);
  expect(screen.getByRole("img", { name: "左目 2D 叠加层" })).toBeVisible();
  expect(screen.getByRole("img", { name: "右目 2D 叠加层" })).toBeVisible();
  expect(container.querySelectorAll(".detection-box")).toHaveLength(2);
});

test("the default worker threshold hides scores below 0.2 and includes the exact boundary", () => {
  const keypoints = Array.from({ length: 21 }, (_, index) => [120 + index, 90 + index]);
  const records: TraceRecord[] = [{
    record_id: "pose-left-default-threshold",
    stage: "POSE_2D",
    status: "SUCCEEDED",
    payload: {
      view_id: "left",
      keypoints_uv: keypoints,
      keypoint_scores: [0.19, 0.2, ...Array(19).fill(0.9)],
    },
  }];

  const { container } = render(<StereoEvidence runKey="threshold-run" records={records} trackId="" />);

  expect(screen.getByText("20 / 21 visible")).toBeVisible();
  expect(container.querySelectorAll("circle.hidden-keypoint")).toHaveLength(1);
});

test("a nested worker threshold in the record payload controls 2D visibility", () => {
  const keypoints = Array.from({ length: 21 }, (_, index) => [120 + index, 90 + index]);
  const records: TraceRecord[] = [{
    record_id: "pose-left-nested-threshold",
    stage: "POSE_2D",
    status: "SUCCEEDED",
    payload: {
      view_id: "left",
      thresholds: { keypoint_score: 0.6 },
      keypoints_uv: keypoints,
      keypoint_scores: [0.59, 0.6, ...Array(19).fill(0.9)],
    },
  }];

  const { container } = render(<StereoEvidence runKey="threshold-run" records={records} trackId="" />);

  expect(screen.getByText("20 / 21 visible")).toBeVisible();
  expect(container.querySelectorAll("circle.hidden-keypoint")).toHaveLength(1);
});

test("a direct record threshold takes priority over the nested worker threshold", () => {
  const keypoints = Array.from({ length: 21 }, (_, index) => [120 + index, 90 + index]);
  const records: TraceRecord[] = [{
    record_id: "pose-left-direct-threshold",
    stage: "POSE_2D",
    status: "SUCCEEDED",
    payload: {
      view_id: "left",
      keypoint_score_threshold: 0.8,
      thresholds: { keypoint_score: 0.1 },
      keypoints_uv: keypoints,
      keypoint_scores: [0.79, 0.8, ...Array(19).fill(0.9)],
    },
  }];

  const { container } = render(<StereoEvidence runKey="threshold-run" records={records} trackId="" />);

  expect(screen.getByText("20 / 21 visible")).toBeVisible();
  expect(container.querySelectorAll("circle.hidden-keypoint")).toHaveLength(1);
});
