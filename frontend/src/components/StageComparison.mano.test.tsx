import { render, screen, within } from "@testing-library/react";
import { expect, test } from "vitest";

import type { TraceRecord } from "../api/types";
import { StageComparison } from "./StageComparison";

function projected(offset: number) {
  return {
    left: Array.from({ length: 21 }, (_, index) => [offset + index, 40 + index]),
    right: Array.from({ length: 21 }, (_, index) => [offset - 8 + index, 40 + index]),
  };
}

function comparisonRecords(mano: TraceRecord): TraceRecord[] {
  return [
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
      record_id: "raw-track-0001",
      stage: "RAW_FUSION",
      status: "SUCCEEDED",
      payload: {
        track_id: "track-0001",
        output_status: "PRODUCED",
        projected_keypoints_space: "rectified",
        projected_keypoints_uv: projected(20),
      },
    },
    mano,
  ];
}

test("operator can audit a produced MANO robust refit without treating its heuristic as probability", () => {
  const inlierMask = Array(21).fill(true);
  inlierMask[0] = false;
  inlierMask[17] = false;
  const jointWeights = inlierMask.map((inlier) => inlier ? 1 : 0);
  const gate = {
    method: "RESIDUAL_TRIM_10PCT_V1",
    status: "HEURISTIC_UNCALIBRATED",
    reason: "ROBUST_INLIER_GATE_PASSED",
    accepted: true,
    triggered: true,
    first_pass_rmse_m: 0.03125,
    full_rmse_m: 0.0275,
    inlier_rmse_m: 0.0125,
    weighted_rmse_m: 0.0125,
    rmse_gate_m: 0.02,
    full_rmse_ceiling_m: 0.04,
    effective_joint_count: 19,
    minimum_effective_joint_count: 15,
    joint_weights: jointWeights,
    inlier_mask: inlierMask,
    trimmed_joint_indices: [17],
    stage_iterations: [
      { stage: "FULL_HUBER", iterations_run: 200 },
      { stage: "WEIGHTED_REFIT", iterations_run: 84 },
    ],
  };
  const records = comparisonRecords({
    record_id: "mano-track-0001",
    stage: "KINEMATIC_REFINEMENT",
    status: "SUCCEEDED",
    event: "mano_frame_fitted",
    payload: {
      track_id: "track-0001",
      output_status: "PRODUCED",
      fit_quality: gate,
      selection: { decision: "SELECTED", gate },
      projected_keypoints_space: "rectified",
      projected_keypoints_uv: projected(28),
    },
  });

  render(
    <StageComparison
      runKey="mano-v3-run"
      records={records}
      selectedNodeId="MANO_FRAMEWISE"
      selectedTrack="track-0001"
    />,
  );

  const diagnostic = screen.getByRole("article", { name: "track-0001 MANO robust gate diagnostic" });
  expect(within(diagnostic).getByText("RESIDUAL_TRIM_10PCT_V1")).toBeVisible();
  expect(within(diagnostic).getByText("HEURISTIC_UNCALIBRATED")).toBeVisible();
  expect(within(diagnostic).getByText("ROBUST_INLIER_GATE_PASSED")).toBeVisible();
  expect(within(diagnostic).getByText("31.25 mm")).toBeVisible();
  expect(within(diagnostic).getByText("27.50 mm")).toBeVisible();
  expect(within(diagnostic).getAllByText("12.50 mm")).toHaveLength(2);
  expect(within(diagnostic).getByText("20.00 mm")).toBeVisible();
  expect(within(diagnostic).getByText("40.00 mm")).toBeVisible();
  expect(within(diagnostic).getByText("19 / 15")).toBeVisible();
  expect(within(diagnostic).getByText("YES · 2 STAGES")).toBeVisible();
  expect(within(diagnostic).getByText("17 · little_mcp")).toBeVisible();
  expect(within(diagnostic).getByText("FULL_HUBER · 200")).toBeVisible();
  expect(within(diagnostic).getByText("WEIGHTED_REFIT · 84")).toBeVisible();
  expect(within(diagnostic).getByRole("list", { name: "track-0001 FHP21 joint weights" }).children).toHaveLength(21);
  expect(within(diagnostic).getByTitle(/0 · wrist_center · weight 0\.00 · NO RAW SUPPORT/)).toBeVisible();
  expect(within(diagnostic).getByTitle(/17 · little_mcp · weight 0\.00 · TRIMMED/)).toBeVisible();
  expect(within(diagnostic).queryByText(/probability|概率/i)).not.toBeInTheDocument();
  expect(screen.getByText("RAW_FUSION → MANO v1.2")).toBeVisible();
  expect(screen.getAllByLabelText(/track-0001 RAW_FUSION/)).toHaveLength(2);
  expect(screen.getAllByLabelText(/track-0001 KINEMATIC_REFINEMENT/)).toHaveLength(2);
});

test("operator sees the robust gate reason when MANO does not produce a frame", () => {
  const gate = {
    method: "RESIDUAL_TRIM_10PCT_V1",
    status: "HEURISTIC_UNCALIBRATED",
    reason: "FULL_RMSE_CEILING_EXCEEDED",
    accepted: false,
    triggered: true,
    first_pass_rmse_m: 0.052,
    full_rmse_m: 0.044,
    inlier_rmse_m: 0.016,
    weighted_rmse_m: 0.016,
    rmse_gate_m: 0.02,
    full_rmse_ceiling_m: 0.04,
    effective_joint_count: 19,
    minimum_effective_joint_count: 15,
    joint_weights: [0, ...Array(16).fill(1), 0, 1, 1, 1],
    inlier_mask: [false, ...Array(16).fill(true), false, true, true, true],
    trimmed_joint_indices: [0, 17],
    stage_iterations: [
      { stage: "FULL_HUBER", iterations_run: 200 },
      { stage: "WEIGHTED_REFIT", iterations_run: 200 },
    ],
  };
  const records = comparisonRecords({
    record_id: "mano-rejected-track-0001",
    stage: "KINEMATIC_REFINEMENT",
    status: "WARNING",
    event: "mano_frame_not_produced",
    payload: {
      track_id: "track-0001",
      output_status: "NOT_PRODUCED",
      selection: {
        decision: "NO_HIGH_QUALITY_FIT",
        status: "REJECTED",
        gate,
      },
    },
  });

  render(
    <StageComparison
      runKey="mano-v3-rejected-run"
      records={records}
      selectedNodeId="MANO_FRAMEWISE"
      selectedTrack=""
    />,
  );

  const diagnostic = screen.getByRole("article", { name: "track-0001 MANO robust gate diagnostic" });
  expect(within(diagnostic).getByText("NOT_PRODUCED")).toBeVisible();
  expect(within(diagnostic).getByText("GATE REJECTED")).toBeVisible();
  expect(within(diagnostic).getByText("FULL_RMSE_CEILING_EXCEEDED")).toBeVisible();
  expect(within(diagnostic).getByText("44.00 mm")).toBeVisible();
  expect(screen.getByText("track-0001 · NOT_PRODUCED · FULL_RMSE_CEILING_EXCEEDED")).toBeVisible();
  expect(screen.getAllByLabelText(/track-0001 RAW_FUSION/)).toHaveLength(2);
  expect(screen.queryByLabelText(/track-0001 KINEMATIC_REFINEMENT/)).not.toBeInTheDocument();
});

test("legacy MANO records keep the original comparison and omit unavailable robust diagnostics", () => {
  const records = comparisonRecords({
    record_id: "legacy-mano-track-0001",
    stage: "KINEMATIC_REFINEMENT",
    status: "WARNING",
    event: "mano_frame_not_produced",
    payload: {
      track_id: "track-0001",
      output_status: "NOT_PRODUCED",
      selection: { decision: "NO_HIGH_QUALITY_FIT" },
    },
  });

  render(
    <StageComparison
      runKey="legacy-mano-run"
      records={records}
      selectedNodeId="MANO_FRAMEWISE"
      selectedTrack=""
    />,
  );

  expect(screen.getByText("RAW_FUSION → MANO v1.2")).toBeVisible();
  expect(screen.getByText("track-0001 · NOT_PRODUCED · NO_HIGH_QUALITY_FIT")).toBeVisible();
  expect(screen.getAllByLabelText(/track-0001 RAW_FUSION/)).toHaveLength(2);
  expect(screen.queryByRole("region", { name: "MANO robust gate diagnostics" })).not.toBeInTheDocument();
  expect(screen.queryByText("METHOD_UNRECORDED")).not.toBeInTheDocument();
});
