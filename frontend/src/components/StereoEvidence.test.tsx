import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test } from "vitest";

import type { TraceRecord } from "../api/types";
import { StereoEvidence } from "./StereoEvidence";

function sourceFrame(side: "left" | "right" = "left"): TraceRecord {
  return {
    record_id: `sync-${side}`,
    stage: "SYNCHRONIZATION",
    status: "SUCCEEDED",
    blobs: [{
      role: `source_${side}`,
      relative_path: `${side}-source.png`,
      media_type: "image/png",
    }],
    payload: { frame_index: 0 },
  };
}

test("changing evidence never carries native skeletons onto an image without matching coordinates", async () => {
  const user = userEvent.setup();
  const keypoints = Array.from({ length: 21 }, (_, index) => [120 + index * 4, 90 + index * 3]);
  const records: TraceRecord[] = [
    {
      record_id: "sync",
      stage: "SYNCHRONIZATION",
      status: "SUCCEEDED",
      blobs: [{ role: "source_left", relative_path: "source-left.png", media_type: "image/png" }],
      payload: { frame_index: 0 },
    },
    {
      record_id: "rectification",
      stage: "RECTIFICATION",
      status: "SUCCEEDED",
      blobs: [
        { role: "undistorted_left", relative_path: "undistorted-left.png", media_type: "image/png" },
        { role: "rectified_left", relative_path: "rectified-left.png", media_type: "image/png" },
      ],
      payload: { frame_index: 0, image_width: 640, image_height: 480 },
    },
    {
      record_id: "tracked-pose-left",
      stage: "POSE_2D",
      status: "SUCCEEDED",
      event: "tracked_view_keypoints_recorded",
      payload: {
        view_id: "left",
        track_id: "track-0000",
        candidate_id: "left-det-0000",
        keypoints_uv: keypoints,
        keypoint_scores: Array(21).fill(0.96),
      },
    },
  ];

  render(<StereoEvidence runKey="atomic-layer-run" records={records} trackId="track-0000" />);

  expect(screen.getByRole("img", { name: "左目 2D 叠加层" })).toBeVisible();
  await user.click(screen.getByRole("button", { name: "放大预览：左目 source_left" }));
  const preview = screen.getByRole("dialog", { name: "左目 source_left" });
  expect(within(preview).getByRole("img", { name: "左目 2D 叠加层" })).toBeVisible();
  await user.keyboard("{Escape}");

  await user.selectOptions(screen.getByRole("combobox", { name: "左目证据层" }), "undistorted:left");

  expect(screen.getByRole("img", { name: "左目 undistorted_left" })).toBeVisible();
  expect(screen.queryByRole("img", { name: "左目 2D 叠加层" })).not.toBeInTheDocument();
  expect(screen.getByText("无对应骨骼 · UNDISTORTED 无严格 UV 映射")).toBeVisible();
});

test("rectified evidence uses rectified keypoints and the rectification image size", async () => {
  const user = userEvent.setup();
  const native = Array.from({ length: 21 }, (_, index) => [100 + index, 80 + index]);
  const rectified = Array.from({ length: 21 }, (_, index) => [300 + index, 180 + index]);
  const records: TraceRecord[] = [
    {
      record_id: "sync",
      stage: "SYNCHRONIZATION",
      status: "SUCCEEDED",
      blobs: [{ role: "source_left", relative_path: "source-left.png", media_type: "image/png" }],
      payload: { frame_index: 0 },
    },
    {
      record_id: "rectification",
      stage: "RECTIFICATION",
      status: "SUCCEEDED",
      blobs: [{ role: "rectified_left", relative_path: "rectified-left.png", media_type: "image/png" }],
      payload: { frame_index: 0, image_width: 800, image_height: 600 },
    },
    {
      record_id: "tracked-pose-left",
      stage: "POSE_2D",
      status: "SUCCEEDED",
      event: "tracked_view_keypoints_recorded",
      payload: {
        view_id: "left",
        track_id: "track-0000",
        candidate_id: "left-det-0000",
        image_width: 640,
        image_height: 480,
        keypoints_uv: native,
        keypoints_uv_rectified: rectified,
        keypoint_scores: Array(21).fill(0.97),
      },
    },
  ];

  render(<StereoEvidence runKey="rectified-layer-run" records={records} trackId="track-0000" />);

  const select = screen.getByRole("combobox", { name: "左目证据层" });
  await user.selectOptions(select, screen.getByRole("option", { name: "RECTIFIED · POSE_2D" }));

  const overlay = screen.getByRole("img", { name: "左目 2D 叠加层" });
  expect(overlay).toHaveAttribute("viewBox", "0 0 800 600");
  expect(overlay.querySelector("circle")).toHaveAttribute("cx", "300");
});

test("an unfiltered evidence layer keeps every hand instead of taking instances zero", () => {
  const hand = (offset: number) => Array.from(
    { length: 21 },
    (_, index) => [offset + index, 70 + index],
  );
  const records: TraceRecord[] = [
    {
      record_id: "sync-left",
      stage: "SYNCHRONIZATION",
      status: "SUCCEEDED",
      blobs: [{ role: "source_left", relative_path: "source-left.png", media_type: "image/png" }],
      payload: { frame_index: 0 },
    },
    {
      record_id: "pose-left",
      stage: "POSE_2D",
      status: "SUCCEEDED",
      event: "view_keypoints_inferred",
      payload: {
        view_id: "left",
        image_width: 640,
        image_height: 480,
        instances: [
          {
            candidate_id: "left-det-0000",
            keypoints_uv: hand(100),
            keypoint_scores: Array(21).fill(0.91),
          },
          {
            candidate_id: "left-det-0001",
            keypoints_uv: hand(300),
            keypoint_scores: Array(21).fill(0.92),
          },
        ],
      },
    },
  ];

  render(<StereoEvidence runKey="all-hands-run" records={records} trackId="" />);

  expect(screen.getByLabelText("left-det-0000 POSE_2D · NATIVE")).toBeInTheDocument();
  expect(screen.getByLabelText("left-det-0001 POSE_2D · NATIVE")).toBeInTheDocument();
  expect(screen.getByText("42 / 42 visible")).toBeVisible();
});

test("a selected track exposes only its exact virtual crop and crop-space keypoints", async () => {
  const user = userEvent.setup();
  const points = (offset: number) => Array.from(
    { length: 21 },
    (_, index) => [offset + index, 50 + index],
  );
  const cropRecord = (candidateId: string, offset: number): TraceRecord => ({
    record_id: `crop:${candidateId}`,
    stage: "POSE_2D",
    status: "SUCCEEDED",
    event: "virtual_crop_pose_inferred",
    blobs: [
      { role: "virtual_crop", relative_path: "same-content-crop.png", media_type: "image/png" },
      { role: "virtual_crop_valid_mask", relative_path: `${candidateId}-mask.png`, media_type: "image/png" },
    ],
    payload: {
      view_id: "left",
      candidate_id: candidateId,
      output_status: "PRODUCED",
      keypoints_uv_crop: points(offset),
      keypoint_scores: Array(21).fill(0.94),
      virtual_camera: { output_size: [256, 192] },
    },
  });
  const records: TraceRecord[] = [
    {
      record_id: "sync-left",
      stage: "SYNCHRONIZATION",
      status: "SUCCEEDED",
      blobs: [{ role: "source_left", relative_path: "source-left.png", media_type: "image/png" }],
      payload: { frame_index: 0 },
    },
    cropRecord("left-det-0000", 100),
    cropRecord("left-det-0001", 300),
    {
      record_id: "tracked-pose-left",
      stage: "POSE_2D",
      status: "SUCCEEDED",
      event: "tracked_view_keypoints_recorded",
      payload: {
        view_id: "left",
        track_id: "track-0001",
        candidate_id: "left-det-0001",
        image_width: 640,
        image_height: 480,
        keypoints_uv: points(500),
        keypoint_scores: Array(21).fill(0.95),
      },
    },
  ];

  render(<StereoEvidence runKey="crop-layer-run" records={records} trackId="track-0001" />);

  const select = screen.getByRole("combobox", { name: "左目证据层" });
  expect(screen.queryByRole("option", { name: "VIRTUAL CROP · left-det-0000" })).not.toBeInTheDocument();
  expect(screen.queryByRole("option", { name: /MASK/ })).not.toBeInTheDocument();

  await user.selectOptions(select, screen.getByRole("option", { name: "VIRTUAL CROP · left-det-0001" }));

  const overlay = screen.getByRole("img", { name: "左目 2D 叠加层" });
  expect(screen.getByRole("img", { name: "左目 virtual_crop · left-det-0001" })).toBeVisible();
  expect(overlay).toHaveAttribute("viewBox", "0 0 256 192");
  expect(overlay.querySelector("circle")).toHaveAttribute("cx", "300");
});

test("RAW_FUSION evidence uses only its rectified projection and preserves nullable joint indexes", async () => {
  const user = userEvent.setup();
  const projected: Array<[number, number] | null> = Array.from(
    { length: 21 },
    (_, index) => [400 + index, 200 + index],
  );
  projected[1] = null;
  const records: TraceRecord[] = [
    {
      record_id: "sync-left",
      stage: "SYNCHRONIZATION",
      status: "SUCCEEDED",
      blobs: [{ role: "source_left", relative_path: "source-left.png", media_type: "image/png" }],
      payload: { frame_index: 0 },
    },
    {
      record_id: "rectification",
      stage: "RECTIFICATION",
      status: "SUCCEEDED",
      blobs: [{ role: "rectified_left", relative_path: "rectified-left.png", media_type: "image/png" }],
      payload: { frame_index: 0, image_width: 800, image_height: 600 },
    },
    {
      record_id: "raw-track-0",
      stage: "RAW_FUSION",
      status: "SUCCEEDED",
      event: "raw_landmarks_triangulated",
      payload: {
        track_id: "track-0000",
        output_status: "PRODUCED",
        projected_keypoints_space: "rectified",
        projected_keypoints_uv: { left: projected, right: projected },
      },
    },
  ];

  render(<StereoEvidence runKey="raw-layer-run" records={records} trackId="track-0000" />);

  const select = screen.getByRole("combobox", { name: "左目证据层" });
  await user.selectOptions(select, within(select).getByRole("option", { name: "RECTIFIED · RAW_FUSION" }));

  const overlay = screen.getByRole("img", { name: "左目 2D 叠加层" });
  expect(screen.getByRole("img", { name: "左目 rectified_left" })).toBeVisible();
  expect(overlay.querySelector("circle[data-joint-index='1']")).not.toBeInTheDocument();
  expect(overlay.querySelector("circle[data-joint-index='2']")).toHaveAttribute("cx", "402");
  expect(overlay.querySelectorAll("circle")).toHaveLength(20);
});

test("MANO, temporal, and export selections each render their own stage projection", async () => {
  const user = userEvent.setup();
  const projected = (offset: number) => Array.from(
    { length: 21 },
    (_, index) => [offset + index, 220 + index],
  );
  const projectedRecord = (stage: string, offset: number): TraceRecord => ({
    record_id: `${stage.toLowerCase()}-track-0`,
    stage,
    status: "SUCCEEDED",
    payload: {
      track_id: "track-0000",
      output_status: "PRODUCED",
      projected_keypoints_space: "rectified",
      projected_keypoints_uv: { left: projected(offset), right: projected(offset + 10) },
    },
  });
  const records: TraceRecord[] = [
    {
      record_id: "rectification",
      stage: "RECTIFICATION",
      status: "SUCCEEDED",
      blobs: [{ role: "rectified_left", relative_path: "rectified-left.png", media_type: "image/png" }],
      payload: { frame_index: 0, image_width: 800, image_height: 600 },
    },
    projectedRecord("KINEMATIC_REFINEMENT", 450),
    projectedRecord("TEMPORAL_REFINEMENT", 550),
    projectedRecord("EXPORT", 650),
  ];

  render(<StereoEvidence runKey="refinement-layer-run" records={records} trackId="track-0000" />);

  const select = screen.getByRole("combobox", { name: "左目证据层" });
  for (const [label, expectedX] of [
    ["RECTIFIED · MANO", "450"],
    ["RECTIFIED · TEMPORAL", "550"],
    ["RECTIFIED · EXPORT", "650"],
  ] as const) {
    await user.selectOptions(select, within(select).getByRole("option", { name: label }));
    expect(screen.getByRole("img", { name: "左目 2D 叠加层" }).querySelector("circle"))
      .toHaveAttribute("cx", expectedX);
  }
});

test("a rejected MANO layer stays empty instead of falling back to RAW projection", async () => {
  const user = userEvent.setup();
  const projected = (offset: number) => Array.from(
    { length: 21 },
    (_, index) => [offset + index, 200 + index],
  );
  const records: TraceRecord[] = [
    {
      record_id: "rectification",
      stage: "RECTIFICATION",
      status: "SUCCEEDED",
      blobs: [{ role: "rectified_left", relative_path: "rectified-left.png", media_type: "image/png" }],
      payload: { image_width: 800, image_height: 600 },
    },
    {
      record_id: "raw-track-0",
      stage: "RAW_FUSION",
      status: "SUCCEEDED",
      payload: {
        track_id: "track-0000",
        output_status: "PRODUCED",
        projected_keypoints_space: "rectified",
        projected_keypoints_uv: { left: projected(400), right: projected(410) },
      },
    },
    {
      record_id: "mano-track-0-rejected",
      stage: "KINEMATIC_REFINEMENT",
      status: "SKIPPED",
      event: "mano_frame_not_produced",
      payload: {
        track_id: "track-0000",
        output_status: "NOT_PRODUCED",
        reason: "ROBUST_GATE_REJECTED",
      },
    },
  ];

  render(<StereoEvidence runKey="mano-reject-run" records={records} trackId="track-0000" />);

  const select = screen.getByRole("combobox", { name: "左目证据层" });
  await user.selectOptions(select, within(select).getByRole("option", { name: "RECTIFIED · MANO" }));

  expect(screen.getByRole("img", { name: "左目 rectified_left" })).toBeVisible();
  expect(screen.queryByRole("img", { name: "左目 2D 叠加层" })).not.toBeInTheDocument();
  expect(screen.getByText("无对应骨骼 · KINEMATIC_REFINEMENT 未提供 track-0000 的 RECTIFIED 投影"))
    .toBeVisible();
});

test("a projected layer without its own rectified frame never borrows the source background", async () => {
  const user = userEvent.setup();
  const projected = Array.from({ length: 21 }, (_, index) => [400 + index, 200 + index]);
  const records: TraceRecord[] = [
    {
      record_id: "sync-left",
      stage: "SYNCHRONIZATION",
      status: "SUCCEEDED",
      blobs: [{ role: "source_left", relative_path: "source-left.png", media_type: "image/png" }],
      payload: { frame_index: 0 },
    },
    {
      record_id: "raw-track-0",
      stage: "RAW_FUSION",
      status: "SUCCEEDED",
      payload: {
        track_id: "track-0000",
        output_status: "PRODUCED",
        projected_keypoints_space: "rectified",
        projected_keypoints_uv: { left: projected, right: projected },
      },
    },
  ];

  render(<StereoEvidence runKey="unsampled-frame-run" records={records} trackId="track-0000" />);

  const select = screen.getByRole("combobox", { name: "左目证据层" });
  await user.selectOptions(select, within(select).getByRole("option", { name: "RECTIFIED · RAW_FUSION" }));

  expect(screen.queryByRole("img", { name: "左目 source_left" })).not.toBeInTheDocument();
  expect(screen.queryByRole("img", { name: "左目 2D 叠加层" })).not.toBeInTheDocument();
  expect(screen.getByText("无可叠加证据 · 本帧未保存 rectified_left")).toBeVisible();
});

test("a non-empty track filter never falls back to an unrelated instances entry", () => {
  const keypoints = (offset: number) => Array.from(
    { length: 21 },
    (_, index) => [offset + index, 80 + index],
  );
  const records: TraceRecord[] = [
    sourceFrame(),
    {
      record_id: "aggregate-pose-left",
      stage: "POSE_2D",
      status: "SUCCEEDED",
      event: "view_keypoints_inferred",
      payload: {
        view_id: "left",
        image_width: 640,
        image_height: 480,
        instances: [
          { candidate_id: "left-det-0000", keypoints_uv: keypoints(100) },
          { candidate_id: "left-det-0001", keypoints_uv: keypoints(300) },
        ],
      },
    },
    {
      record_id: "tracked-other-hand",
      stage: "POSE_2D",
      status: "SUCCEEDED",
      event: "tracked_view_keypoints_recorded",
      payload: {
        view_id: "left",
        track_id: "track-0000",
        candidate_id: "left-det-0000",
        keypoints_uv: keypoints(100),
      },
    },
  ];

  render(<StereoEvidence runKey="strict-track-run" records={records} trackId="track-9999" />);

  expect(screen.getByRole("img", { name: "左目 source_left" })).toBeVisible();
  expect(screen.queryByRole("img", { name: "左目 2D 叠加层" })).not.toBeInTheDocument();
  expect(screen.getByText("无对应骨骼 · POSE_2D 未提供 NATIVE UV")).toBeVisible();
});

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
  const records: TraceRecord[] = [
    sourceFrame(),
    {
      record_id: "pose-left-default-threshold",
      stage: "POSE_2D",
      status: "SUCCEEDED",
      payload: {
        view_id: "left",
        keypoints_uv: keypoints,
        keypoint_scores: [0.19, 0.2, ...Array(19).fill(0.9)],
      },
    },
  ];

  const { container } = render(<StereoEvidence runKey="threshold-run" records={records} trackId="" />);

  expect(screen.getByText("20 / 21 visible")).toBeVisible();
  expect(container.querySelectorAll("circle.hidden-keypoint")).toHaveLength(1);
});

test("a nested worker threshold in the record payload controls 2D visibility", () => {
  const keypoints = Array.from({ length: 21 }, (_, index) => [120 + index, 90 + index]);
  const records: TraceRecord[] = [
    sourceFrame(),
    {
      record_id: "pose-left-nested-threshold",
      stage: "POSE_2D",
      status: "SUCCEEDED",
      payload: {
        view_id: "left",
        thresholds: { keypoint_score: 0.6 },
        keypoints_uv: keypoints,
        keypoint_scores: [0.59, 0.6, ...Array(19).fill(0.9)],
      },
    },
  ];

  const { container } = render(<StereoEvidence runKey="threshold-run" records={records} trackId="" />);

  expect(screen.getByText("20 / 21 visible")).toBeVisible();
  expect(container.querySelectorAll("circle.hidden-keypoint")).toHaveLength(1);
});

test("a direct record threshold takes priority over the nested worker threshold", () => {
  const keypoints = Array.from({ length: 21 }, (_, index) => [120 + index, 90 + index]);
  const records: TraceRecord[] = [
    sourceFrame(),
    {
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
    },
  ];

  const { container } = render(<StereoEvidence runKey="threshold-run" records={records} trackId="" />);

  expect(screen.getByText("20 / 21 visible")).toBeVisible();
  expect(container.querySelectorAll("circle.hidden-keypoint")).toHaveLength(1);
});
