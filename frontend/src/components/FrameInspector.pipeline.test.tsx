import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test } from "vitest";

import type { FrameDetail, RunDetail } from "../api/types";
import { FrameInspector } from "./FrameInspector";

const runDetail: RunDetail = {
  run: {
    run_key: "run-key",
    item_id: "item",
    run_id: "run",
    status: "COMPLETED",
    created_at_utc: null,
    finalized_at_utc: null,
    pipeline_version: "test",
    record_count: 1,
    frame_count: 1,
    stage_counts: {},
    warning_count: 0,
    failure_count: 0,
  },
  manifest: {},
  summary: {},
  validation: { ok: true, errors: [], warnings: [] },
  stages: [],
  track_ids: [],
  view_ids: ["left", "right"],
};

const frameDetail: FrameDetail = {
  run_id: "run",
  frame: {
    frame_key: "frame-key",
    frame_id: "frame/1",
    frame_index: 1,
    timestamp_ns: 1_000_000,
    record_ids: ["sync"],
    stages: ["SYNCHRONIZATION"],
    statuses: ["SUCCEEDED"],
    track_ids: [],
    view_ids: ["left", "right"],
  },
  records: [{
    record_id: "sync",
    stage: "SYNCHRONIZATION",
    status: "SUCCEEDED",
    blobs: [
      { role: "source_left", relative_path: "source-left.jpg" },
      { role: "source_right", relative_path: "source-right.jpg" },
    ],
    payload: { frame_id: "frame/1" },
  }],
};

test("frame inspector lets the operator move through pipeline node comparisons", async () => {
  render(
    <FrameInspector
      runKey="run-key"
      runDetail={runDetail}
      frameDetail={frameDetail}
      selectedTrack=""
      loading={false}
      error=""
    />,
  );

  expect(screen.getByRole("navigation", { name: "Pipeline 节点" })).toBeVisible();
  const comparison = screen.getByRole("region", { name: "阶段前后对比" });
  expect(within(comparison).getByText("PIPELINE INPUT · NATIVE FISHEYE PIXELS")).toBeVisible();

  await userEvent.click(screen.getByRole("button", { name: /OpenCV Fisheye Undistortion/ }));

  expect(within(comparison).getAllByText("NOT_PRODUCED")).toHaveLength(2);
});
