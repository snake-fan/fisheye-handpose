import { render, screen, within } from "@testing-library/react";
import { expect, test, vi } from "vitest";

import type { TraceRecord } from "../api/types";
import { PipelineNodeRail } from "./PipelineNodeRail";

test("operator sees every pipeline node in a fixed order with explicit missing output", () => {
  const records: TraceRecord[] = [
    {
      record_id: "sync",
      stage: "SYNCHRONIZATION",
      status: "SUCCEEDED",
      blobs: [
        { role: "source_left", relative_path: "blobs/source-left.jpg" },
        { role: "source_right", relative_path: "blobs/source-right.jpg" },
      ],
      payload: { frame_id: "frame/1" },
    },
    {
      record_id: "detection-left",
      stage: "DETECTION",
      status: "SUCCEEDED",
      payload: { frame_id: "frame/1", view_id: "left", detections: [{ bbox_xyxy: [1, 2, 3, 4] }] },
    },
    {
      record_id: "mano-failed",
      stage: "KINEMATIC_REFINEMENT",
      status: "WARNING",
      event: "mano_frame_not_produced",
      payload: {
        frame_id: "frame/1",
        output_status: "NOT_PRODUCED",
        selection: { decision: "NO_HIGH_QUALITY_FIT" },
      },
    },
  ];

  render(
    <PipelineNodeRail
      records={records}
      selectedNodeId="SOURCE_RGB"
      onSelect={vi.fn()}
    />,
  );

  const rail = screen.getByRole("navigation", { name: "Pipeline 节点" });
  const nodes = within(rail).getAllByRole("button");
  expect(nodes).toHaveLength(10);
  expect(nodes.map((node) => node.getAttribute("data-node-id"))).toEqual([
    "SOURCE_RGB",
    "FISHEYE_UNDISTORTION",
    "STEREO_RECTIFICATION",
    "HAND_DETECTION",
    "HAND_POSE_2D",
    "CROSS_VIEW_ASSOCIATION",
    "STEREO_TRIANGULATION_RAW_3D",
    "MANO_FRAMEWISE",
    "TEMPORAL_REFINEMENT",
    "STABLE_FHP21_EXPORT",
  ]);
  expect(screen.getByRole("button", { name: /OpenCV Fisheye Undistortion · DEBUG_ONLY/ })).toBeVisible();
  expect(screen.getByRole("button", { name: /Stereo Rectification · DEBUG_ONLY/ })).toBeVisible();
  expect(screen.getByRole("button", { name: /Hand Detection · NATIVE INPUT/ })).toBeVisible();
  expect(within(nodes[0]).getByText("PRODUCED")).toBeVisible();
  expect(within(nodes[1]).getByText("NOT_PRODUCED")).toBeVisible();
  expect(within(nodes[1]).getByText("此帧未产生去畸变图像")).toBeVisible();
  expect(within(nodes[7]).getByText("NO_HIGH_QUALITY_FIT")).toBeVisible();
});

test("mixed produced and rejected hands mark each affected stage as partial with its failure reason", () => {
  const records: TraceRecord[] = [
    {
      record_id: "raw-produced",
      stage: "RAW_FUSION",
      status: "SUCCEEDED",
      payload: { track_id: "track-0000", output_status: "PRODUCED" },
    },
    {
      record_id: "raw-rejected",
      stage: "RAW_FUSION",
      status: "WARNING",
      payload: {
        track_id: null,
        output_status: "NOT_PRODUCED",
        hand_reason: "INSUFFICIENT_PALM_SUPPORT",
      },
    },
    {
      record_id: "temporal-produced",
      stage: "TEMPORAL_REFINEMENT",
      status: "SUCCEEDED",
      payload: { track_id: "track-0000", output_status: "PRODUCED" },
    },
    {
      record_id: "temporal-rejected",
      stage: "TEMPORAL_REFINEMENT",
      status: "WARNING",
      payload: { track_id: null, output_status: "NOT_PRODUCED", reason: "RAW_HAND_GATE_REJECTED" },
    },
  ];

  render(
    <PipelineNodeRail
      records={records}
      selectedNodeId="STEREO_TRIANGULATION_RAW_3D"
      onSelect={vi.fn()}
    />,
  );

  const raw = screen.getByRole("button", { name: /Stereo Triangulation · Raw 3D/ });
  const temporal = screen.getByRole("button", { name: /Temporal Refinement/ });
  expect(raw).toHaveClass("partial");
  expect(within(raw).getByText("PARTIAL")).toBeVisible();
  expect(within(raw).getByText("INSUFFICIENT_PALM_SUPPORT")).toBeVisible();
  expect(temporal).toHaveClass("partial");
  expect(within(temporal).getByText("PARTIAL")).toBeVisible();
  expect(within(temporal).getByText("RAW_HAND_GATE_REJECTED")).toBeVisible();
});
