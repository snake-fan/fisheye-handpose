import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";

import type { RunDetail } from "../api/types";
import { OverlayVideoPlayer } from "./OverlayVideoPlayer";

function detailWith(records: RunDetail["global_records"]): RunDetail {
  return {
    run: {
      run_key: "video-run",
      item_id: "item",
      run_id: "run",
      status: "COMPLETED",
      created_at_utc: null,
      finalized_at_utc: null,
      pipeline_version: "test",
      record_count: 1,
      frame_count: 120,
      stage_counts: { EXPORT: 1 },
      warning_count: 0,
      failure_count: 0,
    },
    manifest: {},
    summary: {},
    validation: { ok: true, errors: [], warnings: [] },
    stages: ["EXPORT"],
    track_ids: ["track-0000", "track-0001"],
    view_ids: ["left", "right"],
    global_records: records,
  };
}

test("operator plays the stereo overlay with timeline and truthful RAW to EMA provenance", () => {
  const detail = detailWith([{
    record_id: "overlay-video",
    stage: "EXPORT",
    status: "SUCCEEDED",
    event: "overlay_videos_exported",
    payload: {
      output_status: "PRODUCED",
      input_stage: "RAW_FUSION",
      temporal_method: "causal_time_ema_v1",
      frame_count: 120,
      projected_keypoints_space: "rectified",
    },
    blobs: [
      {
        role: "overlay_video_raw_vs_stable_stereo_rectified",
        relative_path: "blobs/sha256/aa/overlay.mp4",
        media_type: "video/mp4",
      },
      {
        role: "overlay_video_timeline",
        relative_path: "blobs/sha256/bb/timeline.json",
        media_type: "application/json",
      },
    ],
  }]);

  render(<OverlayVideoPlayer runKey="video-run" detail={detail} />);

  const video = screen.getByLabelText("Raw 与 Stable 双目骨架抖动对比视频");
  expect(video).toHaveAttribute("controls");
  expect(video).toHaveAttribute("preload", "metadata");
  expect(video).toHaveAttribute("src", expect.stringContaining("/overlay.mp4"));
  expect(screen.getByText("MANO 未产出 · RAW → EMA")).toBeVisible();
  expect(screen.getByText("RAW_FUSION → causal_time_ema_v1")).toBeVisible();
  expect(screen.getByRole("link", { name: "下载帧时间映射" })).toHaveAttribute(
    "href",
    expect.stringContaining("/timeline.json"),
  );
  expect(screen.getByText("120 frames · RECTIFIED")).toBeVisible();
});

test("an old run without video remains inspectable and explains the fallback", () => {
  render(<OverlayVideoPlayer runKey="legacy-run" detail={detailWith([])} />);

  expect(screen.queryByLabelText("Raw 与 Stable 双目骨架抖动对比视频")).not.toBeInTheDocument();
  expect(screen.getByText("此运行没有叠加视频")).toBeVisible();
  expect(screen.getByText("逐帧节点对比仍然可用")).toBeVisible();
});

test("video provenance reports the complete mixed stable input set from worker metadata", () => {
  const detail = detailWith([{
    record_id: "overlay-video",
    stage: "EXPORT",
    status: "SUCCEEDED",
    payload: {
      output_status: "PRODUCED",
      stable_input_stages: ["KINEMATIC_REFINEMENT", "RAW_FUSION"],
      comparison_stages: ["RAW_FUSION", "TEMPORAL_REFINEMENT"],
      image_space: "rectified",
      frame_count: 120,
    },
    blobs: [{
      role: "overlay_video_raw_vs_stable_stereo_rectified",
      relative_path: "mixed.mp4",
      media_type: "video/mp4",
    }],
  }]);

  render(<OverlayVideoPlayer runKey="mixed-run" detail={detail} />);

  expect(screen.getByText("RAW_FUSION + KINEMATIC_REFINEMENT → TEMPORAL_REFINEMENT")).toBeVisible();
  expect(screen.getByText("混合输入 · RAW / MANO → EMA")).toBeVisible();
  expect(screen.queryByText("MANO → Temporal")).not.toBeInTheDocument();
});

test("video provenance does not claim MANO when a legacy artifact has no input metadata", () => {
  const detail = detailWith([{
    record_id: "overlay-video",
    stage: "EXPORT",
    status: "SUCCEEDED",
    payload: { output_status: "PRODUCED" },
    blobs: [{
      role: "overlay_video_raw_vs_stable_stereo_rectified",
      relative_path: "unknown.mp4",
      media_type: "video/mp4",
    }],
  }]);

  render(<OverlayVideoPlayer runKey="unknown-run" detail={detail} />);

  expect(screen.getByText("输入来源未记录")).toBeVisible();
  expect(screen.queryByText("MANO → Temporal")).not.toBeInTheDocument();
});
