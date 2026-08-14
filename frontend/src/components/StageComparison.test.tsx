import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test } from "vitest";

import type { TraceRecord } from "../api/types";
import { StageComparison } from "./StageComparison";

test("source node presents the synchronized fisheye pair as the pipeline input", () => {
  const records: TraceRecord[] = [{
    record_id: "sync",
    stage: "SYNCHRONIZATION",
    blobs: [
      { role: "source_left", relative_path: "source-left.jpg" },
      { role: "source_right", relative_path: "source-right.jpg" },
    ],
    payload: { frame_id: "frame/1" },
  }];

  render(
    <StageComparison
      runKey="source-run"
      records={records}
      selectedNodeId="SOURCE_RGB"
      selectedTrack=""
    />,
  );

  expect(screen.getByRole("img", { name: "左目 Pipeline source_left" })).toBeVisible();
  expect(screen.getByRole("img", { name: "右目 Pipeline source_right" })).toBeVisible();
  expect(screen.getByText("PIPELINE INPUT · NATIVE FISHEYE PIXELS")).toBeVisible();
});

test("operator compares source and undistorted images for both stereo views", () => {
  const records: TraceRecord[] = [
    {
      record_id: "sync",
      stage: "SYNCHRONIZATION",
      blobs: [
        { role: "source_left", relative_path: "blobs/sha256/aa/source-left.jpg", media_type: "image/jpeg" },
        { role: "source_right", relative_path: "blobs/sha256/bb/source-right.jpg", media_type: "image/jpeg" },
      ],
      payload: { frame_id: "frame/1" },
    },
    {
      record_id: "undistorted",
      stage: "RECTIFICATION",
      blobs: [
        { role: "undistorted_left", relative_path: "blobs/sha256/cc/undistorted-left.jpg", media_type: "image/jpeg" },
        { role: "undistorted_right", relative_path: "blobs/sha256/dd/undistorted-right.jpg", media_type: "image/jpeg" },
      ],
      payload: { frame_id: "frame/1", output_status: "PRODUCED" },
    },
  ];

  render(
    <StageComparison
      runKey="opaque-run"
      records={records}
      selectedNodeId="FISHEYE_UNDISTORTION"
      selectedTrack=""
    />,
  );

  expect(screen.getByRole("img", { name: "左目 before source_left" })).toHaveAttribute(
    "src",
    expect.stringContaining("/api/v1/runs/opaque-run/artifacts/blobs/sha256/aa/source-left.jpg"),
  );
  expect(screen.getByRole("img", { name: "左目 after undistorted_left" })).toBeVisible();
  expect(screen.getByRole("img", { name: "右目 before source_right" })).toBeVisible();
  expect(screen.getByRole("img", { name: "右目 after undistorted_right" })).toBeVisible();
  expect(screen.getByText("DEBUG_ONLY QA BRANCH · DETECTION INPUT REMAINS NATIVE FISHEYE RGB")).toBeVisible();
});

test("stereo rectification comparison starts from undistorted rather than native images", () => {
  const records: TraceRecord[] = [{
    record_id: "rectification",
    stage: "RECTIFICATION",
    blobs: [
      { role: "source_left", relative_path: "source-left.jpg" },
      { role: "undistorted_left", relative_path: "undistorted-left.jpg" },
      { role: "undistorted_right", relative_path: "undistorted-right.jpg" },
      { role: "rectified_left", relative_path: "rectified-left.jpg" },
      { role: "rectified_right", relative_path: "rectified-right.jpg" },
    ],
    payload: { frame_id: "frame/1" },
  }];

  render(
    <StageComparison
      runKey="opaque-run"
      records={records}
      selectedNodeId="STEREO_RECTIFICATION"
      selectedTrack=""
    />,
  );

  expect(screen.getByRole("img", { name: "左目 before undistorted_left" })).toBeVisible();
  expect(screen.getByRole("img", { name: "右目 after rectified_right" })).toBeVisible();
  expect(screen.queryByRole("img", { name: /before source_left/ })).not.toBeInTheDocument();
});

test("RTMPose comparison keeps every hand visible and only highlights the selected track", () => {
  const points = (offset: number) => Array.from({ length: 21 }, (_, index) => [offset + index, 50 + index]);
  const records: TraceRecord[] = [
    {
      record_id: "sync",
      stage: "SYNCHRONIZATION",
      blobs: [
        { role: "source_left", relative_path: "source-left.jpg" },
        { role: "source_right", relative_path: "source-right.jpg" },
      ],
      payload: { frame_id: "frame/1" },
    },
    ...(["left", "right"] as const).flatMap((view) => [
      {
        record_id: `pose-${view}-0`,
        stage: "POSE_2D",
        status: "SUCCEEDED",
        payload: {
          frame_id: "frame/1",
          view_id: view,
          track_id: "track-0000",
          keypoints_uv: points(20),
          keypoint_scores: Array(21).fill(0.9),
          image_width: 320,
          image_height: 240,
        },
      },
      {
        record_id: `pose-${view}-1`,
        stage: "POSE_2D",
        status: "SUCCEEDED",
        payload: {
          frame_id: "frame/1",
          view_id: view,
          track_id: "track-0001",
          keypoints_uv: points(150),
          keypoint_scores: Array(21).fill(0.95),
          image_width: 320,
          image_height: 240,
        },
      },
    ]),
  ];

  const { rerender } = render(
    <StageComparison
      runKey="two-hand-run"
      records={records}
      selectedNodeId="HAND_POSE_2D"
      selectedTrack=""
    />,
  );

  const left = screen.getByRole("region", { name: "左目 HAND_POSE_2D" });
  expect(screen.queryByRole("region", { name: "Virtual crop RTMPose diagnostics" })).not.toBeInTheDocument();
  expect(within(left).getByRole("img", { name: "左目 RTMPose 全手叠加" })).toBeVisible();
  expect(within(left).getByText("track-0000")).toHaveAttribute("aria-current", "false");
  expect(within(left).getByText("track-0001")).toHaveAttribute("aria-current", "false");

  rerender(
    <StageComparison
      runKey="two-hand-run"
      records={records}
      selectedNodeId="HAND_POSE_2D"
      selectedTrack="track-0001"
    />,
  );

  expect(within(left).getByText("track-0001")).toHaveAttribute("aria-current", "true");
  expect(within(left).getByText("track-0000")).toHaveAttribute("aria-current", "false");
  expect(within(left).getByLabelText("track-0000 2D 骨架")).toHaveAttribute("stroke", "#75f6c4");
  expect(within(left).getByLabelText("track-0001 2D 骨架")).toHaveAttribute("stroke", "#ffb454");
});

test("virtual-crop RTMPose diagnostics lazily load only the selected candidate evidence", async () => {
  const user = userEvent.setup();
  const cropPoints = (offset: number) => Array.from(
    { length: 21 },
    (_, index) => [offset + index, 20 + index],
  );
  const nativePoints = (offset: number) => Array.from(
    { length: 21 },
    (_, index) => [offset + index, 120 + index],
  );
  const virtualCamera = {
    virtual_camera_id: "sha256:camera-left-0",
    crop_policy_id: "virtual-perspective-kb4/v1",
    side: "left",
    source_bbox_xyxy: [90, 100, 210, 240],
    output_size: [256, 256],
    K_virtual: [[256, 0, 127.5], [0, 256, 127.5], [0, 0, 1]],
    valid_fraction: 0.9375,
  };
  const records: TraceRecord[] = [
    {
      record_id: "sync",
      stage: "SYNCHRONIZATION",
      blobs: [
        { role: "source_left", relative_path: "source-left.jpg" },
        { role: "source_right", relative_path: "source-right.jpg" },
      ],
      payload: { frame_id: "frame/1" },
    },
    {
      record_id: "crop-left-0",
      stage: "POSE_2D",
      event: "virtual_crop_pose_inferred",
      status: "SUCCEEDED",
      blobs: [
        { role: "virtual_crop", relative_path: "left-0-crop.jpg" },
        { role: "virtual_crop_valid_mask", relative_path: "left-0-mask.png" },
      ],
      payload: {
        view_id: "left",
        image_width: 640,
        image_height: 480,
        candidate_id: "left-0",
        output_status: "PRODUCED",
        reason: null,
        model_input_space: "virtual_pinhole",
        virtual_camera: virtualCamera,
        keypoints_uv_crop: cropPoints(10),
        keypoints_uv_native: nativePoints(110),
      },
    },
    {
      record_id: "crop-left-1",
      stage: "POSE_2D",
      event: "virtual_crop_pose_not_produced",
      status: "WARNING",
      blobs: [
        { role: "virtual_crop", relative_path: "left-1-crop.jpg" },
        { role: "virtual_crop_valid_mask", relative_path: "left-1-mask.png" },
      ],
      payload: {
        view_id: "left",
        image_width: 640,
        image_height: 480,
        candidate_id: "left-1",
        output_status: "NOT_PRODUCED",
        reason: "CROP_VALID_FRACTION_BELOW_THRESHOLD",
        model_input_space: "virtual_pinhole",
        virtual_camera: { ...virtualCamera, virtual_camera_id: "sha256:camera-left-1", valid_fraction: 0.42 },
      },
    },
    {
      record_id: "crop-right-0",
      stage: "POSE_2D",
      event: "virtual_crop_pose_inferred",
      status: "SUCCEEDED",
      blobs: [{ role: "virtual_crop", relative_path: "right-0-crop.jpg" }],
      payload: {
        view_id: "right",
        image_width: 640,
        image_height: 480,
        candidate_id: "right-0",
        output_status: "PRODUCED",
        reason: null,
        model_input_space: "virtual_pinhole",
        virtual_camera: { ...virtualCamera, virtual_camera_id: "sha256:camera-right-0", side: "right" },
        keypoints_uv_crop: cropPoints(30),
        keypoints_uv_native: nativePoints(330),
      },
    },
  ];

  const { container } = render(
    <StageComparison
      runKey="virtual-run"
      records={records}
      selectedNodeId="HAND_POSE_2D"
      selectedTrack=""
    />,
  );

  expect(screen.getByText("VIRTUAL PERSPECTIVE CROP DIAGNOSTICS")).toBeVisible();
  const left0 = screen.getByRole("article", { name: "left-0 virtual crop diagnostic" });
  const left1 = screen.getByRole("article", { name: "left-1 virtual crop diagnostic" });
  expect(left0).toBeVisible();
  expect(left1).toBeVisible();
  expect(screen.getByRole("article", { name: "right-0 virtual crop diagnostic" })).toBeVisible();

  expect(screen.queryByRole("img", { name: "left-0 virtual crop" })).not.toBeInTheDocument();
  expect(screen.queryByRole("img", { name: "left-1 virtual crop" })).not.toBeInTheDocument();
  expect(screen.queryByRole("img", { name: "right-0 virtual crop" })).not.toBeInTheDocument();
  expect(screen.queryByRole("img", { name: /valid mask/ })).not.toBeInTheDocument();

  await user.click(within(left0).getByRole("button", { name: "展开 left-0 图像证据" }));
  expect(screen.getByRole("img", { name: "left-0 virtual crop" })).toHaveAttribute(
    "src",
    expect.stringContaining("/api/v1/runs/virtual-run/artifacts/left-0-crop.jpg"),
  );
  expect(screen.getByRole("img", { name: "left-0 native fisheye source" })).toHaveAttribute(
    "src",
    expect.stringContaining("/api/v1/runs/virtual-run/artifacts/source-left.jpg"),
  );
  expect(screen.getByRole("img", { name: "left-0 valid mask" })).toBeVisible();
  expect(screen.queryByRole("img", { name: "left-1 virtual crop" })).not.toBeInTheDocument();
  expect(screen.queryByRole("img", { name: "right-0 virtual crop" })).not.toBeInTheDocument();
  expect(screen.queryByRole("img", { name: "right-0 valid mask" })).not.toBeInTheDocument();
  expect(screen.getByRole("img", { name: "left-0 crop-space keypoints" })).toHaveAttribute("viewBox", "0 0 256 256");
  expect(screen.getByRole("img", { name: "left-0 native-space keypoints" })).toHaveAttribute("viewBox", "0 0 640 480");
  expect(container.querySelector('[aria-label="left-0 crop-space keypoints"] circle[cx="10"]')).toBeInTheDocument();
  expect(container.querySelector('[aria-label="left-0 native-space keypoints"] circle[cx="110"]')).toBeInTheDocument();
  expect(screen.getAllByText("virtual-perspective-kb4/v1")).toHaveLength(3);
  expect(screen.getAllByText("FOV 53.1° × 53.1°")).toHaveLength(3);
  expect(within(left0).getByText("VALID 93.8%")).toBeVisible();
  expect(within(left1).getByText("CROP_VALID_FRACTION_BELOW_THRESHOLD")).toBeVisible();

  await user.click(screen.getByRole("button", { name: "展开 right-0 图像证据" }));
  expect(screen.queryByRole("img", { name: "left-0 virtual crop" })).not.toBeInTheDocument();
  expect(screen.queryByRole("img", { name: "left-0 native fisheye source" })).not.toBeInTheDocument();
  expect(screen.getByRole("img", { name: "right-0 virtual crop" })).toBeVisible();
  expect(screen.getByRole("img", { name: "right-0 native fisheye source" })).toHaveAttribute(
    "src",
    expect.stringContaining("/api/v1/runs/virtual-run/artifacts/source-right.jpg"),
  );
  expect(screen.queryByRole("img", { name: /valid mask/ })).not.toBeInTheDocument();
});

test("raw 3D projection on a rectified background never falls back to native keypoints", () => {
  const projected = Array.from({ length: 21 }, (_, index) => [30 + index, 40 + index]);
  const native = Array.from({ length: 21 }, (_, index) => [900 + index, 800 + index]);
  const records: TraceRecord[] = [
    {
      record_id: "rectified",
      stage: "RECTIFICATION",
      blobs: [
        { role: "rectified_left", relative_path: "rectified-left.jpg" },
        { role: "rectified_right", relative_path: "rectified-right.jpg" },
      ],
      payload: { frame_id: "frame/1", output_width: 640, output_height: 480 },
    },
    {
      record_id: "pose-native",
      stage: "POSE_2D",
      payload: { frame_id: "frame/1", view_id: "left", track_id: "track-0000", keypoints_uv: native },
    },
    {
      record_id: "raw",
      stage: "RAW_FUSION",
      status: "SUCCEEDED",
      payload: {
        frame_id: "frame/1",
        track_id: "track-0000",
        output_status: "PRODUCED",
        projected_keypoints_space: "rectified",
        projected_keypoints_uv: { left: projected, right: projected.map(([x, y]) => [x - 8, y]) },
        landmarks_xyz_m: Array.from({ length: 21 }, () => [0.01, 0.02, 0.4]),
        validity: Array(21).fill("VALID"),
      },
    },
  ];

  const { container } = render(
    <StageComparison
      runKey="projection-run"
      records={records}
      selectedNodeId="STEREO_TRIANGULATION_RAW_3D"
      selectedTrack=""
    />,
  );

  expect(screen.getByRole("img", { name: "左目 Raw 3D rectified projection" })).toBeVisible();
  expect(container.querySelector('circle[cx="30"]')).toBeInTheDocument();
  expect(container.querySelector('circle[cx="900"]')).not.toBeInTheDocument();
  expect(screen.getByText("RECTIFIED PIXEL SPACE")).toBeVisible();
});

test("hand detection compares the source image with every detected bounding box", () => {
  const records: TraceRecord[] = [
    {
      record_id: "sync",
      stage: "SYNCHRONIZATION",
      blobs: [
        { role: "source_left", relative_path: "source-left.jpg" },
        { role: "source_right", relative_path: "source-right.jpg" },
      ],
      payload: { frame_id: "frame/1" },
    },
    ...(["left", "right"] as const).map((view) => ({
      record_id: `detection-${view}`,
      stage: "DETECTION",
      status: "SUCCEEDED",
      payload: {
        frame_id: "frame/1",
        view_id: view,
        image_width: 320,
        image_height: 240,
        output_status: "PRODUCED",
        detections: [
          { candidate_id: `${view}-0`, bbox_xyxy: [10, 20, 80, 120], bbox_score: 0.92 },
          { candidate_id: `${view}-1`, bbox_xyxy: [140, 30, 220, 150], bbox_score: 0.88 },
        ],
      },
    })),
  ];

  render(
    <StageComparison
      runKey="detection-run"
      records={records}
      selectedNodeId="HAND_DETECTION"
      selectedTrack=""
    />,
  );

  const left = screen.getByRole("region", { name: "左目 HAND_DETECTION" });
  expect(within(left).getByRole("img", { name: "左目 detection input" })).toBeVisible();
  expect(within(left).getByRole("img", { name: "左目 detection output background" })).toBeVisible();
  expect(within(left).getByLabelText("left-0 detection")).toBeInTheDocument();
  expect(within(left).getByLabelText("left-1 detection")).toBeInTheDocument();
  expect(within(left).getByText("2 HAND CANDIDATES")).toBeVisible();
  expect(screen.queryByText("RAW DETECTOR PROPOSALS → BOUNDED ASSOCIATION POOL")).not.toBeInTheDocument();
});

test("candidate-aware detection audits every raw proposal before the bounded association pool", () => {
  const decision = (
    candidateId: string,
    sourceIndex: number,
    score: number,
    classification: "SEED" | "RECOVERY" | "REJECTED",
    reason: string,
    eligible: boolean,
  ) => ({
    candidate_id: candidateId,
    source_index: sourceIndex,
    bbox_xyxy: [10 + sourceIndex * 40, 20, 42 + sourceIndex * 40, 100],
    score,
    bbox_score: score,
    label: sourceIndex === 3 ? 1 : 0,
    classification,
    reason,
    eligible_for_association: eligible,
    final_selection: null,
  });
  const seed = decision(
    "left-det-0000",
    0,
    0.91,
    "SEED",
    "SCORE_MEETS_SEED_THRESHOLD",
    true,
  );
  const recovery = decision(
    "left-det-0001",
    1,
    0.24,
    "RECOVERY",
    "SCORE_MEETS_RECOVERY_THRESHOLD",
    true,
  );
  const belowThreshold = decision(
    "left-det-0002",
    2,
    0.12,
    "REJECTED",
    "SCORE_BELOW_RECOVERY_THRESHOLD",
    false,
  );
  const wrongCategory = decision(
    "left-det-0003",
    3,
    0.83,
    "REJECTED",
    "CATEGORY_MISMATCH",
    false,
  );
  const records: TraceRecord[] = [
    {
      record_id: "sync",
      stage: "SYNCHRONIZATION",
      blobs: [
        { role: "source_left", relative_path: "source-left.jpg" },
        { role: "source_right", relative_path: "source-right.jpg" },
      ],
      payload: { frame_id: "frame/1" },
    },
    {
      record_id: "detection-left",
      stage: "DETECTION",
      event: "hand_candidates_detected",
      status: "SUCCEEDED",
      payload: {
        frame_id: "frame/1",
        view_id: "left",
        image_width: 320,
        image_height: 240,
        output_status: "PRODUCED",
        candidate_decisions: [seed, recovery, belowThreshold, wrongCategory],
        candidate_pool: [seed, recovery],
        detections: [seed, recovery],
      },
    },
  ];

  render(
    <StageComparison
      runKey="candidate-run"
      records={records}
      selectedNodeId="HAND_DETECTION"
      selectedTrack=""
    />,
  );

  expect(screen.getByText("RAW DETECTOR PROPOSALS → BOUNDED ASSOCIATION POOL")).toBeVisible();
  const left = screen.getByRole("region", { name: "左目 HAND_DETECTION" });
  expect(within(left).getByText("4 RAW → 2 POOL")).toBeVisible();
  expect(within(left).getByRole("img", { name: "左目 raw detector proposals" })).toBeVisible();
  expect(within(left).getByRole("img", { name: "左目 bounded association pool" })).toBeVisible();
  expect(within(left).getByLabelText("left-det-0002 REJECTED raw proposal")).toBeInTheDocument();
  expect(within(left).queryByLabelText("left-det-0002 REJECTED pool candidate")).not.toBeInTheDocument();

  const seedRow = within(left).getByRole("listitem", { name: "left-det-0000 candidate decision" });
  const recoveryRow = within(left).getByRole("listitem", { name: "left-det-0001 candidate decision" });
  const belowRow = within(left).getByRole("listitem", { name: "left-det-0002 candidate decision" });
  const categoryRow = within(left).getByRole("listitem", { name: "left-det-0003 candidate decision" });
  expect(seedRow).toHaveAttribute("data-classification", "SEED");
  expect(seedRow).toHaveAttribute("data-in-pool", "true");
  expect(within(seedRow).getByText("SCORE 91.0%")).toBeVisible();
  expect(within(seedRow).getByText("SOURCE #0")).toBeVisible();
  expect(within(seedRow).getByText("ELIGIBLE YES")).toBeVisible();
  expect(within(seedRow).getByText("SCORE_MEETS_SEED_THRESHOLD")).toBeVisible();
  expect(recoveryRow).toHaveAttribute("data-classification", "RECOVERY");
  expect(within(recoveryRow).getByText("SCORE 24.0%")).toBeVisible();
  expect(belowRow).toHaveAttribute("data-classification", "REJECTED");
  expect(belowRow).toHaveAttribute("data-in-pool", "false");
  expect(within(belowRow).getByText("ELIGIBLE NO")).toBeVisible();
  expect(within(belowRow).getByText("SCORE_BELOW_RECOVERY_THRESHOLD")).toBeVisible();
  expect(within(categoryRow).getByText("CATEGORY_MISMATCH")).toBeVisible();
});

test("association uses rectified candidate coordinates and exposes matches and tracks", () => {
  const native = Array.from({ length: 21 }, (_, index) => [900 + index, 800 + index]);
  const rectified = (offset: number) => Array.from({ length: 21 }, (_, index) => [offset + index, 50 + index]);
  const records: TraceRecord[] = [
    {
      record_id: "rectification",
      stage: "RECTIFICATION",
      blobs: [
        { role: "rectified_left", relative_path: "rectified-left.jpg" },
        { role: "rectified_right", relative_path: "rectified-right.jpg" },
      ],
      payload: { frame_id: "frame/1", output_width: 320, output_height: 240 },
    },
    ...(["left", "right"] as const).map((view) => ({
      record_id: `pose-${view}`,
      stage: "POSE_2D",
      payload: {
        frame_id: "frame/1",
        view_id: view,
        instances: [
          { candidate_id: `${view}-0`, keypoints_uv: native, keypoints_uv_rectified: rectified(20) },
          { candidate_id: `${view}-1`, keypoints_uv: native, keypoints_uv_rectified: rectified(140) },
        ],
      },
    })),
    {
      record_id: "association",
      stage: "CROSS_VIEW_ASSOCIATION",
      event: "cross_view_hands_associated",
      status: "SUCCEEDED",
      payload: {
        frame_id: "frame/1",
        output_status: "PRODUCED",
        matches: [
          { match_id: "match-0", left_candidate_id: "left-0", right_candidate_id: "right-0" },
          { match_id: "match-1", left_candidate_id: "left-1", right_candidate_id: "right-1" },
        ],
        unmatched_left_indices: [],
        unmatched_right_indices: [],
      },
    },
    {
      record_id: "tracking",
      stage: "CROSS_VIEW_ASSOCIATION",
      event: "sequence_tracks_assigned",
      payload: {
        frame_id: "frame/1",
        assignments: [
          { observation_id: "frame:match-0", track_id: "track-0000", decision: "NEW" },
          { observation_id: "frame:match-1", track_id: "track-0001", decision: "NEW" },
        ],
      },
    },
  ];

  const { container } = render(
    <StageComparison
      runKey="association-run"
      records={records}
      selectedNodeId="CROSS_VIEW_ASSOCIATION"
      selectedTrack="track-0001"
    />,
  );

  expect(screen.getByText("CANDIDATES → MATCHED / TRACKED")).toBeVisible();
  expect(screen.getByText("match-0 → track-0000")).toBeVisible();
  expect(screen.getByText("match-1 → track-0001")).toBeVisible();
  expect(screen.getAllByLabelText(/track-0000 association/)).toHaveLength(2);
  expect(container.querySelector('circle[cx="20"]')).toBeInTheDocument();
  expect(container.querySelector('circle[cx="900"]')).not.toBeInTheDocument();
});

test("association keeps candidate identity when tracked pose records coexist with aggregate instances", () => {
  const points = Array.from({ length: 21 }, (_, index) => [40 + index, 50 + index]);
  const records: TraceRecord[] = [
    {
      record_id: "rectification",
      stage: "RECTIFICATION",
      blobs: [
        { role: "rectified_left", relative_path: "rectified-left.jpg" },
        { role: "rectified_right", relative_path: "rectified-right.jpg" },
      ],
      payload: { image_width: 320, image_height: 240 },
    },
    ...(["left", "right"] as const).flatMap((view) => [
      {
        record_id: `pose-aggregate-${view}`,
        stage: "POSE_2D",
        payload: { view_id: view, instances: [{ candidate_id: `${view}-0`, keypoints_uv_rectified: points }] },
      },
      {
        record_id: `pose-tracked-${view}`,
        stage: "POSE_2D",
        event: "tracked_view_keypoints_recorded",
        payload: {
          view_id: view,
          candidate_id: `${view}-0`,
          track_id: "track-0000",
          keypoints_uv_rectified: points,
        },
      },
    ]),
    {
      record_id: "association",
      stage: "CROSS_VIEW_ASSOCIATION",
      payload: {
        matches: [{
          match_id: "match-0",
          left_candidate_id: "left-0",
          right_candidate_id: "right-0",
        }],
      },
    },
    {
      record_id: "tracking",
      stage: "CROSS_VIEW_ASSOCIATION",
      payload: { assignments: [{ observation_id: "frame:match-0", track_id: "track-0000" }] },
    },
  ];

  render(
    <StageComparison
      runKey="association-real-shape"
      records={records}
      selectedNodeId="CROSS_VIEW_ASSOCIATION"
      selectedTrack=""
    />,
  );

  expect(screen.getByLabelText("left-0 candidate keypoints")).toBeInTheDocument();
  expect(screen.getByLabelText("right-0 candidate keypoints")).toBeInTheDocument();
  expect(screen.getAllByLabelText(/track-0000 association/)).toHaveLength(2);
  expect(screen.queryByText("track-0000 · UNMATCHED")).not.toBeInTheDocument();
});

test("mixed-hand raw-gate rejection stays untracked and exposes its downstream failure chain", () => {
  const rectified = (offset: number) => Array.from(
    { length: 21 },
    (_, index) => [offset + index, 50 + index],
  );
  const projected = (offset: number) => ({
    left: rectified(offset),
    right: rectified(offset - 8),
  });
  const records: TraceRecord[] = [
    {
      record_id: "rectification",
      stage: "RECTIFICATION",
      blobs: [
        { role: "rectified_left", relative_path: "rectified-left.jpg" },
        { role: "rectified_right", relative_path: "rectified-right.jpg" },
      ],
      payload: { output_width: 320, output_height: 240 },
    },
    ...(["left", "right"] as const).map((view) => ({
      record_id: `pose-${view}`,
      stage: "POSE_2D",
      payload: {
        view_id: view,
        instances: [
          { candidate_id: `${view}-0`, keypoints_uv_rectified: rectified(20) },
          { candidate_id: `${view}-1`, keypoints_uv_rectified: rectified(140) },
        ],
      },
    })),
    {
      record_id: "association",
      stage: "CROSS_VIEW_ASSOCIATION",
      payload: {
        matches: [
          { match_id: "match-0", left_candidate_id: "left-0", right_candidate_id: "right-0" },
          { match_id: "match-1", left_candidate_id: "left-1", right_candidate_id: "right-1" },
        ],
      },
    },
    {
      record_id: "tracking",
      stage: "CROSS_VIEW_ASSOCIATION",
      payload: { assignments: [{ observation_id: "frame:match-0", track_id: "track-0000" }] },
    },
    {
      record_id: "raw-produced",
      stage: "RAW_FUSION",
      status: "SUCCEEDED",
      payload: {
        track_id: "track-0000",
        output_status: "PRODUCED",
        projected_keypoints_space: "rectified",
        projected_keypoints_uv: projected(24),
      },
    },
    {
      record_id: "raw-rejected",
      stage: "RAW_FUSION",
      event: "raw_hand_gate_not_produced",
      status: "WARNING",
      payload: {
        track_id: null,
        observation_id: "frame:match-1",
        output_status: "NOT_PRODUCED",
        hand_validity: "INVALID",
        hand_reason: "INSUFFICIENT_PALM_SUPPORT",
        match: { match_id: "match-1", left_candidate_id: "left-1", right_candidate_id: "right-1" },
      },
    },
    {
      record_id: "mano-rejected",
      stage: "KINEMATIC_REFINEMENT",
      status: "WARNING",
      payload: { track_id: null, output_status: "NOT_PRODUCED", reason: "RAW_HAND_GATE_REJECTED" },
    },
    {
      record_id: "temporal-produced",
      stage: "TEMPORAL_REFINEMENT",
      status: "SUCCEEDED",
      payload: {
        track_id: "track-0000",
        input_stage: "RAW_FUSION",
        output_status: "PRODUCED",
        projected_keypoints_space: "rectified",
        projected_keypoints_uv: projected(30),
      },
    },
    {
      record_id: "temporal-rejected",
      stage: "TEMPORAL_REFINEMENT",
      status: "WARNING",
      payload: { track_id: null, output_status: "NOT_PRODUCED", reason: "RAW_HAND_GATE_REJECTED" },
    },
  ];

  const { rerender } = render(
    <StageComparison
      runKey="mixed-gate-run"
      records={records}
      selectedNodeId="CROSS_VIEW_ASSOCIATION"
      selectedTrack=""
    />,
  );

  expect(screen.getByText("match-0 → track-0000")).toBeVisible();
  expect(screen.getByText("match-1 · MATCHED · UNTRACKED")).toBeVisible();
  expect(screen.queryByText("match-1 → match-1")).not.toBeInTheDocument();
  expect(screen.getAllByLabelText(/match-1 · UNTRACKED association/)).toHaveLength(2);

  rerender(
    <StageComparison
      runKey="mixed-gate-run"
      records={records}
      selectedNodeId="STEREO_TRIANGULATION_RAW_3D"
      selectedTrack=""
    />,
  );

  expect(screen.getAllByLabelText(/track-0000 ASSOCIATION/)).toHaveLength(2);
  expect(screen.queryByLabelText(/match-1.*ASSOCIATION/)).not.toBeInTheDocument();
  expect(screen.getByText("NO_TRACK · NOT_PRODUCED · INSUFFICIENT_PALM_SUPPORT")).toBeVisible();
  expect(screen.getByText("DOWNSTREAM · NOT_PRODUCED · RAW_HAND_GATE_REJECTED")).toBeVisible();

  rerender(
    <StageComparison
      runKey="mixed-gate-run"
      records={records}
      selectedNodeId="TEMPORAL_REFINEMENT"
      selectedTrack=""
    />,
  );

  expect(screen.getByText("track-0000 · RAW_FUSION → TEMPORAL_REFINEMENT · RAW → EMA")).toBeVisible();
  expect(screen.queryByText(/NO_TRACK · UNKNOWN_INPUT/)).not.toBeInTheDocument();
  expect(screen.getByText("NO_TRACK · NOT_PRODUCED · RAW_HAND_GATE_REJECTED")).toBeVisible();
});

test("MANO compares raw projections and reports a per-track fitting failure without inventing output", () => {
  const projected = (offset: number) => ({
    left: Array.from({ length: 21 }, (_, index) => [offset + index, 40 + index]),
    right: Array.from({ length: 21 }, (_, index) => [offset - 8 + index, 40 + index]),
  });
  const records: TraceRecord[] = [
    {
      record_id: "rectification",
      stage: "RECTIFICATION",
      blobs: [
        { role: "rectified_left", relative_path: "rectified-left.jpg" },
        { role: "rectified_right", relative_path: "rectified-right.jpg" },
      ],
      payload: { output_width: 320, output_height: 240 },
    },
    ...["track-0000", "track-0001"].map((trackId, index) => ({
      record_id: `raw-${trackId}`,
      stage: "RAW_FUSION",
      payload: {
        track_id: trackId,
        output_status: "PRODUCED",
        projected_keypoints_space: "rectified",
        projected_keypoints_uv: projected(20 + index * 120),
      },
    })),
    {
      record_id: "mano-success",
      stage: "KINEMATIC_REFINEMENT",
      status: "SUCCEEDED",
      payload: {
        track_id: "track-0000",
        output_status: "PRODUCED",
        projected_keypoints_space: "rectified",
        projected_keypoints_uv: projected(28),
      },
    },
    {
      record_id: "mano-failure",
      stage: "KINEMATIC_REFINEMENT",
      status: "WARNING",
      payload: {
        track_id: "track-0001",
        output_status: "NOT_PRODUCED",
        selection: { decision: "NO_HIGH_QUALITY_FIT" },
        projected_keypoints_space: "rectified",
        projected_keypoints_uv: { left: Array(21).fill(null), right: Array(21).fill(null) },
      },
    },
  ];

  render(
    <StageComparison
      runKey="mano-run"
      records={records}
      selectedNodeId="MANO_FRAMEWISE"
      selectedTrack=""
    />,
  );

  expect(screen.getByText("RAW_FUSION → MANO v1.2")).toBeVisible();
  expect(screen.getByText("track-0001 · NOT_PRODUCED · NO_HIGH_QUALITY_FIT")).toBeVisible();
  expect(screen.getAllByLabelText(/track-0001 RAW_FUSION/)).toHaveLength(2);
  expect(screen.queryByLabelText(/track-0001 KINEMATIC_REFINEMENT/)).not.toBeInTheDocument();
});

test("temporal comparison follows each track's actual input stage and keeps every output visible", () => {
  const projected = (offset: number) => ({
    left: Array.from({ length: 21 }, (_, index) => [offset + index, 40 + index]),
    right: Array.from({ length: 21 }, (_, index) => [offset - 8 + index, 40 + index]),
  });
  const records: TraceRecord[] = [
    {
      record_id: "rectification",
      stage: "RECTIFICATION",
      blobs: [
        { role: "rectified_left", relative_path: "rectified-left.jpg" },
        { role: "rectified_right", relative_path: "rectified-right.jpg" },
      ],
      payload: { output_width: 320, output_height: 240 },
    },
    {
      record_id: "raw-0",
      stage: "RAW_FUSION",
      payload: { track_id: "track-0000", projected_keypoints_space: "rectified", projected_keypoints_uv: projected(20) },
    },
    {
      record_id: "mano-1",
      stage: "KINEMATIC_REFINEMENT",
      payload: { track_id: "track-0001", output_status: "PRODUCED", projected_keypoints_space: "rectified", projected_keypoints_uv: projected(140) },
    },
    ...[
      ["track-0000", "RAW_FUSION", 30],
      ["track-0001", "KINEMATIC_REFINEMENT", 150],
    ].map(([trackId, inputStage, offset]) => ({
      record_id: `temporal-${trackId}`,
      stage: "TEMPORAL_REFINEMENT",
      payload: {
        track_id: trackId,
        input_stage: inputStage,
        method: "causal_time_ema_v1",
        output_status: "PRODUCED",
        projected_keypoints_space: "rectified",
        projected_keypoints_uv: projected(Number(offset)),
      },
    })),
  ];

  render(
    <StageComparison
      runKey="temporal-run"
      records={records}
      selectedNodeId="TEMPORAL_REFINEMENT"
      selectedTrack="track-0001"
    />,
  );

  expect(screen.getByText("track-0000 · RAW_FUSION → TEMPORAL_REFINEMENT · RAW → EMA")).toBeVisible();
  expect(screen.getByText("track-0001 · KINEMATIC_REFINEMENT → TEMPORAL_REFINEMENT · MANO → EMA")).toBeVisible();
  expect(screen.getAllByLabelText(/track-0000 TEMPORAL_REFINEMENT/)).toHaveLength(2);
  expect(screen.getAllByLabelText(/track-0001 TEMPORAL_REFINEMENT/)).toHaveLength(2);
});

test("legacy temporal records never pretend RAW was the actual input when provenance is absent", () => {
  const projected = {
    left: Array.from({ length: 21 }, (_, index) => [30 + index, 40 + index]),
    right: Array.from({ length: 21 }, (_, index) => [22 + index, 40 + index]),
  };
  const records: TraceRecord[] = [
    {
      record_id: "rectification",
      stage: "RECTIFICATION",
      blobs: [
        { role: "rectified_left", relative_path: "rectified-left.jpg" },
        { role: "rectified_right", relative_path: "rectified-right.jpg" },
      ],
      payload: { image_width: 320, image_height: 240 },
    },
    {
      record_id: "legacy-temporal",
      stage: "TEMPORAL_REFINEMENT",
      payload: {
        track_id: "track-0000",
        output_status: "PRODUCED",
        projected_keypoints_space: "rectified",
        projected_keypoints_uv: projected,
      },
    },
  ];

  render(
    <StageComparison
      runKey="legacy-temporal-run"
      records={records}
      selectedNodeId="TEMPORAL_REFINEMENT"
      selectedTrack=""
    />,
  );

  expect(screen.getByText("track-0000 · UNKNOWN_INPUT → TEMPORAL_REFINEMENT · 来源未记录")).toBeVisible();
  expect(screen.queryByText(/RAW → EMA/)).not.toBeInTheDocument();
});

test("final export compares temporal and exported projections for all tracks", () => {
  const projected = (offset: number) => ({
    left: Array.from({ length: 21 }, (_, index) => [offset + index, 40 + index]),
    right: Array.from({ length: 21 }, (_, index) => [offset - 8 + index, 40 + index]),
  });
  const records: TraceRecord[] = [
    {
      record_id: "rectification",
      stage: "RECTIFICATION",
      blobs: [
        { role: "rectified_left", relative_path: "rectified-left.jpg" },
        { role: "rectified_right", relative_path: "rectified-right.jpg" },
      ],
      payload: { output_width: 320, output_height: 240 },
    },
    ...["track-0000", "track-0001"].flatMap((trackId, index) => [
      {
        record_id: `temporal-${trackId}`,
        stage: "TEMPORAL_REFINEMENT",
        payload: { track_id: trackId, projected_keypoints_space: "rectified", projected_keypoints_uv: projected(20 + index * 120) },
      },
      {
        record_id: `export-${trackId}`,
        stage: "EXPORT",
        status: "SUCCEEDED",
        payload: { track_id: trackId, output_status: "PRODUCED", projected_keypoints_space: "rectified", projected_keypoints_uv: projected(24 + index * 120) },
      },
    ]),
  ];

  render(
    <StageComparison
      runKey="export-run"
      records={records}
      selectedNodeId="STABLE_FHP21_EXPORT"
      selectedTrack="track-0000"
    />,
  );

  expect(screen.getByText("TEMPORAL_REFINEMENT → STABLE FHP21 EXPORT")).toBeVisible();
  expect(screen.getAllByLabelText(/track-0001 TEMPORAL_REFINEMENT/)).toHaveLength(2);
  expect(screen.getAllByLabelText(/track-0001 EXPORT/)).toHaveLength(2);
});

test("projected layers preserve nullable joint indexes instead of shifting topology", () => {
  const partial = Array.from({ length: 21 }, (_, index) => (
    index === 1 ? null : [30 + index * 10, 40 + index]
  ));
  const records: TraceRecord[] = [
    {
      record_id: "rectification",
      stage: "RECTIFICATION",
      blobs: [
        { role: "rectified_left", relative_path: "rectified-left.jpg" },
        { role: "rectified_right", relative_path: "rectified-right.jpg" },
      ],
      payload: { output_width: 640, output_height: 480 },
    },
    {
      record_id: "raw",
      stage: "RAW_FUSION",
      payload: {
        track_id: "track-0000",
        projected_keypoints_space: "rectified",
        projected_keypoints_uv: { left: partial, right: partial },
      },
    },
  ];

  const { container } = render(
    <StageComparison
      runKey="partial-run"
      records={records}
      selectedNodeId="STEREO_TRIANGULATION_RAW_3D"
      selectedTrack=""
    />,
  );

  expect(container.querySelector('circle[data-joint-index="0"][cx="30"]')).toBeInTheDocument();
  expect(container.querySelector('circle[data-joint-index="1"]')).not.toBeInTheDocument();
  expect(container.querySelector('circle[data-joint-index="2"][cx="50"]')).toBeInTheDocument();
});

test("rectified overlays use the worker's image_width and image_height contract", () => {
  const projected = Array.from({ length: 21 }, (_, index) => [300 + index, 400 + index]);
  const records: TraceRecord[] = [
    {
      record_id: "rectification",
      stage: "RECTIFICATION",
      blobs: [
        { role: "rectified_left", relative_path: "rectified-left.jpg" },
        { role: "rectified_right", relative_path: "rectified-right.jpg" },
      ],
      payload: { image_width: 1600, image_height: 1300 },
    },
    {
      record_id: "raw",
      stage: "RAW_FUSION",
      payload: {
        track_id: "track-0000",
        projected_keypoints_space: "rectified",
        projected_keypoints_uv: { left: projected, right: projected },
      },
    },
  ];

  render(
    <StageComparison
      runKey="real-size-run"
      records={records}
      selectedNodeId="STEREO_TRIANGULATION_RAW_3D"
      selectedTrack=""
    />,
  );

  expect(screen.getByRole("img", { name: "左目 Raw 3D rectified projection" })).toHaveAttribute(
    "viewBox",
    "0 0 1600 1300",
  );
});
