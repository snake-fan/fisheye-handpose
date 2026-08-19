import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { expect, test, vi } from "vitest";

import { traceApi } from "../api/client";
import type { OverlayVideoTimeline, RunDetail } from "../api/types";
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

function overlayDetail(withTimeline = true): RunDetail {
  return detailWith([{
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
      ...(withTimeline ? [{
        role: "overlay_video_timeline",
        relative_path: "blobs/sha256/bb/timeline.json",
        media_type: "application/json",
      }] : []),
    ],
  }]);
}

function timelineDocument(frameCount = 120): OverlayVideoTimeline {
  const durationPoints = 3003;
  return {
    schema_version: "fisheye-handpose/overlay-video-timeline/v1",
    frame_rate: { numerator: 30_000, denominator: 1001 },
    time_base: { numerator: 1, denominator: 90_000 },
    frames: Array.from({ length: frameCount }, (_, index) => ({
      video_frame_index: index,
      video_pts: index * durationPoints,
      duration_pts: durationPoints,
      frame_id: `frame/${String(index).padStart(6, "0")}`,
      frame_index: index + 40,
      timestamp_ns: 1_000_000_000 + index * 33_366_667,
      track_ids: ["track-0000"],
    })),
  };
}

test("operator plays the stereo overlay with timeline and truthful RAW to EMA provenance", async () => {
  vi.spyOn(traceApi, "getArtifactJson").mockResolvedValue(timelineDocument());

  render(<OverlayVideoPlayer runKey="video-run" detail={overlayDetail()} />);

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
  const slider = await screen.findByRole("slider", { name: "帧时间轴" });
  expect(slider).toHaveAttribute("min", "0");
  expect(slider).toHaveAttribute("max", "119");
  expect(slider).toHaveAttribute("step", "1");
  expect(slider).toBeDisabled();
});

test("operator drags and steps through exact video presentation frames", async () => {
  const timeline = timelineDocument();
  vi.spyOn(traceApi, "getArtifactJson").mockResolvedValue(timeline);
  const pause = vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
  render(<OverlayVideoPlayer runKey="video-run" detail={overlayDetail()} />);

  const video = screen.getByLabelText("Raw 与 Stable 双目骨架抖动对比视频") as HTMLVideoElement;
  const slider = await screen.findByRole("slider", { name: "帧时间轴" });
  fireEvent.loadedMetadata(video);
  expect(slider).toBeEnabled();

  fireEvent.change(slider, { target: { value: "7" } });
  const expectedFrameSevenTime = (timeline.frames[7].video_pts
    + timeline.frames[7].duration_pts / 2)
    * timeline.time_base.numerator / timeline.time_base.denominator;
  expect(pause).toHaveBeenCalledTimes(1);
  expect(video.currentTime).toBeCloseTo(expectedFrameSevenTime, 8);
  expect(slider).toHaveValue("7");
  expect(screen.getByText(/8 \/ 120/)).toBeVisible();
  expect(screen.getByText(/源帧 000047/)).toBeVisible();

  fireEvent.click(screen.getByRole("button", { name: "下一帧" }));
  expect(slider).toHaveValue("8");
  fireEvent.click(screen.getByRole("button", { name: "上一帧" }));
  expect(slider).toHaveValue("7");
});

test("the frame timeline follows normal video playback", async () => {
  const timeline = timelineDocument();
  vi.spyOn(traceApi, "getArtifactJson").mockResolvedValue(timeline);
  render(<OverlayVideoPlayer runKey="video-run" detail={overlayDetail()} />);

  const video = screen.getByLabelText("Raw 与 Stable 双目骨架抖动对比视频") as HTMLVideoElement;
  const slider = await screen.findByRole("slider", { name: "帧时间轴" });
  fireEvent.loadedMetadata(video);
  video.currentTime = timeline.frames[60].video_pts
    * timeline.time_base.numerator / timeline.time_base.denominator;
  fireEvent.timeUpdate(video);

  await waitFor(() => expect(slider).toHaveValue("60"));
  expect(screen.getByText(/61 \/ 120/)).toBeVisible();
});

test("a late timeline load adopts the paused native video position", async () => {
  const timeline = timelineDocument();
  let resolveTimeline: (value: OverlayVideoTimeline) => void = () => {};
  const delayedTimeline = new Promise<OverlayVideoTimeline>((resolve) => {
    resolveTimeline = resolve;
  });
  vi.spyOn(traceApi, "getArtifactJson").mockReturnValue(delayedTimeline);
  render(<OverlayVideoPlayer runKey="video-run" detail={overlayDetail()} />);

  const video = screen.getByLabelText("Raw 与 Stable 双目骨架抖动对比视频") as HTMLVideoElement;
  video.currentTime = timeline.frames[60].video_pts
    * timeline.time_base.numerator / timeline.time_base.denominator;
  fireEvent.loadedMetadata(video);
  resolveTimeline(timeline);

  const slider = await screen.findByRole("slider", { name: "帧时间轴" });
  await waitFor(() => expect(slider).toHaveValue("60"));
});

test("a legacy CFR video gets frame stepping from duration when no timeline artifact exists", async () => {
  const pause = vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
  render(<OverlayVideoPlayer runKey="video-run" detail={overlayDetail(false)} />);

  const video = screen.getByLabelText("Raw 与 Stable 双目骨架抖动对比视频") as HTMLVideoElement;
  Object.defineProperty(video, "duration", { configurable: true, value: 4 });
  fireEvent.loadedMetadata(video);
  const slider = await screen.findByRole("slider", { name: "帧时间轴" });
  expect(screen.getByText(/CFR 估算/)).toBeVisible();

  fireEvent.change(slider, { target: { value: "30" } });
  expect(pause).toHaveBeenCalledTimes(1);
  expect(video.currentTime).toBeCloseTo((30.5 * 4) / 120, 8);
  expect(slider).toHaveValue("30");
});

test("a malformed timeline never breaks native video playback", async () => {
  vi.spyOn(traceApi, "getArtifactJson").mockResolvedValue({ schema_version: "wrong" });
  render(<OverlayVideoPlayer runKey="video-run" detail={overlayDetail()} />);

  expect(await screen.findByText("帧时间轴不可用，仍可使用视频进度条")).toBeVisible();
  expect(screen.getByLabelText("Raw 与 Stable 双目骨架抖动对比视频")).toHaveAttribute("controls");
  expect(screen.queryByRole("slider", { name: "帧时间轴" })).not.toBeInTheDocument();
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


test("latest successful produced EXPORT owns the video and optional timeline as one record", () => {
  const getArtifact = vi.spyOn(traceApi, "getArtifactJson");
  const qualified = (recordId: string, video: string, timeline?: string) => ({
    record_id: recordId,
    stage: "EXPORT",
    status: "SUCCEEDED",
    payload: { output_status: "PRODUCED", frame_count: 12 },
    blobs: [
      {
        role: "overlay_video_raw_vs_stable_stereo_rectified",
        relative_path: video,
        media_type: "video/mp4",
      },
      ...(timeline ? [{
        role: "overlay_video_timeline",
        relative_path: timeline,
        media_type: "application/json",
      }] : []),
    ],
  });
  const detail = detailWith([
    qualified("old-pair", "old.mp4", "old-timeline.json"),
    qualified("latest-video", "latest.mp4"),
    {
      ...qualified("newer-failed", "failed.mp4", "failed-timeline.json"),
      status: "FAILED",
    },
  ]);

  render(<OverlayVideoPlayer runKey="paired-run" detail={detail} />);

  expect(screen.getByLabelText("Raw 与 Stable 双目骨架抖动对比视频"))
    .toHaveAttribute("src", expect.stringContaining("/latest.mp4"));
  expect(screen.queryByRole("link", { name: "下载帧时间映射" })).not.toBeInTheDocument();
  expect(getArtifact).not.toHaveBeenCalled();
});
