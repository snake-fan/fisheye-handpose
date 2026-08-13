import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test } from "vitest";

import type { RunDetail, TraceRecord } from "../api/types";
import { InspectorTabs } from "./InspectorTabs";

const MODEL_HASH = "2".repeat(64);
const REQUEST_HASH = "1".repeat(64);
const CALIBRATION_HASH = "3".repeat(64);
const MANO_HASH = "4".repeat(64);
const OUTPUT_HASH = "5".repeat(64);

function runDetail(overrides: Partial<RunDetail> = {}): RunDetail {
  return {
    run: {
      run_key: "capture 17/run-h20",
      item_id: "capture-17",
      run_id: "run-h20",
      status: "COMPLETED",
      created_at_utc: "2026-08-13T04:02:53Z",
      finalized_at_utc: "2026-08-13T04:04:31Z",
      pipeline_version: "0.2.0",
      record_count: 9,
      frame_count: 1,
      stage_counts: {},
      warning_count: 0,
      failure_count: 0,
    },
    manifest: {},
    summary: { status: "COMPLETED" },
    validation: { ok: true, errors: [], warnings: [] },
    provenance: {
      worker_inputs: {
        record_id: "h20:system:verified",
        payload: {
          request_sha256: REQUEST_HASH,
          model_manifest_sha256: MODEL_HASH,
          mmpose_commit: "0123456789abcdef",
          mano_manifest_sha256: MANO_HASH,
        },
      },
      calibration: {
        record_id: "h20:calibration:rectification",
        payload: { calibration_id: `sha256:${CALIBRATION_HASH}` },
      },
    },
    stages: ["SYSTEM", "RECTIFICATION", "KINEMATIC_REFINEMENT", "EXPORT"],
    track_ids: ["hand-0"],
    view_ids: ["left", "right"],
    global_records: [],
    ...overrides,
  };
}

test("shows evidence-backed H20 provenance and downloads the final FHP21 artifact", async () => {
  const globalRecords: TraceRecord[] = [
    {
      record_id: "h20:system:verified",
      stage: "SYSTEM",
      status: "SUCCEEDED",
      event: "worker_inputs_verified",
      parent_ids: ["audit:complete"],
      blobs: [
        {
          role: "worker_fhp21_output",
          media_type: "application/x-ndjson",
          bytes: 12_480,
          sha256: OUTPUT_HASH,
          relative_path: `blobs/sha256/${OUTPUT_HASH.slice(0, 2)}/${OUTPUT_HASH}.jsonl`,
        },
        {
          role: "worker_manifest",
          media_type: "application/json",
          bytes: 901,
          sha256: "6".repeat(64),
          relative_path: `blobs/sha256/66/${"6".repeat(64)}.json`,
        },
      ],
      payload: {
        request_sha256: REQUEST_HASH,
        model_manifest_sha256: MODEL_HASH,
        mmpose_commit: "0123456789abcdef",
        mano_manifest_sha256: MANO_HASH,
      },
    },
    {
      record_id: "h20:calibration:rectification",
      stage: "RECTIFICATION",
      status: "SUCCEEDED",
      event: "worker_rectification_loaded",
      parent_ids: ["h20:system:verified"],
      payload: { calibration_id: `sha256:${CALIBRATION_HASH}` },
    },
    {
      record_id: "h20:mano:configuration",
      stage: "KINEMATIC_REFINEMENT",
      status: "SUCCEEDED",
      event: "mano_models_loaded",
      parent_ids: ["h20:calibration:rectification"],
      payload: {
        manifest_sha256: MANO_HASH,
        mapping_id: "mano-v1.2-j16-tips-to-fhp21/v1",
      },
    },
  ];
  const detail = runDetail({ global_records: globalRecords });

  render(<InspectorTabs detail={detail} records={globalRecords} />);
  await userEvent.click(screen.getByRole("tab", { name: "来源" }));

  expect(screen.getByText(MODEL_HASH)).toBeVisible();
  expect(screen.getByText("0123456789abcdef")).toBeVisible();
  expect(screen.getByText(`sha256:${CALIBRATION_HASH}`)).toBeVisible();
  expect(screen.getByText(CALIBRATION_HASH)).toBeVisible();
  expect(screen.getByText(MANO_HASH)).toBeVisible();
  expect(screen.getByText("mano-v1.2-j16-tips-to-fhp21/v1")).toBeVisible();

  const finalLink = screen.getByRole("link", { name: "下载 worker_fhp21_output" });
  expect(finalLink).toHaveAttribute(
    "href",
    `/api/v1/runs/capture%2017%2Frun-h20/artifacts/blobs/sha256/55/${OUTPUT_HASH}.jsonl`,
  );
  expect(finalLink).toHaveAttribute("download", `${OUTPUT_HASH}.jsonl`);
  expect(screen.getByText("12.2 KB · sha256:555555555555")).toBeVisible();
  expect(screen.getByText("FINAL FHP21")).toBeVisible();
});

test("falls back to worker-style global and frame records when structured provenance is absent", async () => {
  const globalRecords: TraceRecord[] = [{
    record_id: "h20:system:verified",
    stage: "SYSTEM",
    event: "worker_inputs_verified",
    payload: {
      request_sha256: REQUEST_HASH,
      model_manifest_sha256: MODEL_HASH,
      mmpose_commit: "fallback-commit",
    },
  }];
  const frameRecords: TraceRecord[] = [
    {
      record_id: "h20:calibration:rectification",
      stage: "RECTIFICATION",
      event: "worker_rectification_loaded",
      payload: { calibration_id: `sha256:${CALIBRATION_HASH}` },
    },
    {
      record_id: "h20:part0001:pair000001:mano:match0001",
      stage: "KINEMATIC_REFINEMENT",
      event: "mano_frame_fitted",
      payload: { mapping_id: "mano-v1.2-j16-tips-to-fhp21/v1" },
    },
  ];
  const detail = runDetail({ provenance: undefined, global_records: globalRecords });

  render(<InspectorTabs detail={detail} records={frameRecords} />);
  await userEvent.click(screen.getByRole("tab", { name: "来源" }));

  expect(screen.getByText(MODEL_HASH)).toBeVisible();
  expect(screen.getByText("fallback-commit")).toBeVisible();
  expect(screen.getByText(`sha256:${CALIBRATION_HASH}`)).toBeVisible();
  expect(screen.getByText("mano-v1.2-j16-tips-to-fhp21/v1")).toBeVisible();
});
