import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";

import { App } from "./App";
import type { RunSummary } from "./api/types";

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function runSummary(overrides: Partial<RunSummary> = {}): RunSummary {
  return {
    run_key: "audit-key",
    item_id: "Orbbec_Ego_204253",
    run_id: "run-audit-only",
    status: "COMPLETED",
    created_at_utc: "2026-08-13T04:02:53Z",
    finalized_at_utc: "2026-08-13T04:04:31Z",
    pipeline_version: "0.1.0",
    record_count: 12,
    frame_count: 0,
    stage_counts: { DISCOVERY: 1, DETECTION: 1, POSE_2D: 1 },
    warning_count: 0,
    failure_count: 0,
    ...overrides,
  };
}

test("audit-only runs explain every model stage that was not produced", async () => {
  window.history.replaceState({}, "", "/?run=audit-key&stage=DETECTION");
  const run = runSummary();
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname === "/api/v1/runs") {
      return jsonResponse({ items: [], offset: 0, limit: 50, total: 0 });
    }
    if (url.pathname === "/api/v1/runs/audit-key") {
      return jsonResponse({
        run,
        manifest: { run_id: run.run_id, metadata: { item_id: run.item_id } },
        summary: { status: "COMPLETED" },
        validation: { ok: true, errors: [], warnings: [] },
        stages: [],
        track_ids: [],
        view_ids: [],
        global_records: [
          {
            record_id: "pipeline:skipped:detection",
            stage: "DETECTION",
            status: "SKIPPED",
            event: "stage_output_not_produced",
            parent_ids: ["audit:complete"],
            payload: {
              output_status: "NOT_PRODUCED",
              reason: "no perception, MANO, or temporal backend bundle was configured",
            },
          },
          {
            record_id: "pipeline:skipped:pose_2d",
            stage: "POSE_2D",
            status: "SKIPPED",
            event: "stage_output_not_produced",
            parent_ids: ["pipeline:skipped:detection"],
            payload: {
              output_status: "NOT_PRODUCED",
              reason: "no perception, MANO, or temporal backend bundle was configured",
            },
          },
        ],
      });
    }
    if (url.pathname === "/api/v1/runs/audit-key/frames") {
      return jsonResponse({ items: [], offset: 0, limit: 120, total: 0 });
    }
    return jsonResponse({ error: { message: "not found" } }, 404);
  });

  render(<App />);

  expect(await screen.findByRole("heading", { name: "运行级阶段记录" })).toBeVisible();
  expect(
    within(screen.getByRole("combobox", { name: "阶段" }))
      .getAllByRole("option")
      .map((option) => option.textContent),
  ).toEqual(["全部阶段"]);
  await waitFor(() => {
    expect(new URL(window.location.href).searchParams.has("stage")).toBe(false);
  });
  const panel = screen.getByRole("tabpanel", { name: "阶段" });
  expect(within(panel).getAllByText("SKIPPED")).toHaveLength(2);
  expect(within(panel).getAllByText("NOT_PRODUCED")).toHaveLength(2);
  expect(
    within(panel).getAllByText("no perception, MANO, or temporal backend bundle was configured"),
  ).toHaveLength(2);
});

test("partial FHP21 evidence reports missing 2D and 3D landmarks against the 21-point contract", async () => {
  const frameKey = "d".repeat(64);
  window.history.replaceState({}, "", `/?run=partial-key&frame=${frameKey}`);
  const run = runSummary({
    run_key: "partial-key",
    run_id: "run-partial-pose",
    record_count: 3,
    frame_count: 1,
    stage_counts: { POSE_2D: 2, RAW_FUSION: 1 },
  });
  const leftKeypoints = Array.from({ length: 18 }, (_, index) => [100 + index, 80 + index]);
  const rightKeypoints = Array.from({ length: 21 }, (_, index) => [90 + index, 80 + index]);
  const landmarks = Array.from({ length: 21 }, (_, index) => (
    index < 3 ? [null, null, null] : [index * 0.002, index * 0.003, 0.4]
  ));
  const frame = {
    frame_key: frameKey,
    frame_id: "frame/000003",
    frame_index: 3,
    timestamp_ns: 100_000_000,
    record_ids: ["pose-left-3", "pose-right-3", "fusion-3"],
    stages: ["POSE_2D", "RAW_FUSION"],
    statuses: ["WARNING"],
    track_ids: ["hand-0"],
    view_ids: ["left", "right"],
  };
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname === "/api/v1/runs") {
      return jsonResponse({ items: [], offset: 0, limit: 50, total: 0 });
    }
    if (url.pathname === "/api/v1/runs/partial-key") {
      return jsonResponse({
        run,
        manifest: { run_id: run.run_id, metadata: { item_id: run.item_id } },
        summary: { status: "COMPLETED" },
        validation: { ok: true, errors: [], warnings: ["partial landmarks"] },
        stages: ["POSE_2D", "RAW_FUSION"],
        track_ids: ["hand-0"],
        view_ids: ["left", "right"],
        global_records: [],
      });
    }
    if (url.pathname === "/api/v1/runs/partial-key/frames") {
      return jsonResponse({ items: [frame], offset: 0, limit: 120, total: 1 });
    }
    if (url.pathname === `/api/v1/runs/partial-key/frames/${frameKey}`) {
      return jsonResponse({
        run_key: run.run_key,
        run_id: run.run_id,
        frame,
        records: [
          {
            record_id: "pose-left-3",
            stage: "POSE_2D",
            status: "WARNING",
            event: "partial_keypoints_inferred",
            payload: {
              frame_id: frame.frame_id,
              view_id: "left",
              track_id: "hand-0",
              keypoints_uv: leftKeypoints,
              keypoint_scores: Array(18).fill(0.9),
            },
          },
          {
            record_id: "pose-right-3",
            stage: "POSE_2D",
            status: "SUCCEEDED",
            event: "view_keypoints_inferred",
            payload: {
              frame_id: frame.frame_id,
              view_id: "right",
              track_id: "hand-0",
              keypoints_uv: rightKeypoints,
              keypoint_scores: Array(21).fill(0.95),
            },
          },
          {
            record_id: "fusion-3",
            stage: "RAW_FUSION",
            status: "WARNING",
            event: "partial_landmarks_fused",
            parent_ids: ["pose-left-3", "pose-right-3"],
            payload: {
              frame_id: frame.frame_id,
              track_id: "hand-0",
              landmarks_xyz_m: landmarks,
              validity: [...Array(3).fill(false), ...Array(18).fill(true)],
              coordinate_frame: "rig",
            },
          },
        ],
      });
    }
    return jsonResponse({ error: { message: "not found" } }, 404);
  });

  render(<App />);

  expect(await screen.findByText("18 / 21 visible")).toBeVisible();
  expect(screen.getByText("18 / 21")).toBeVisible();
  expect(screen.getByText("3 missing")).toBeVisible();
  expect(screen.getByRole("img", { name: "FHP21 三维骨架" })).toBeVisible();
});

test("switching between colliding run IDs uses run_key and clears run-scoped filters", async () => {
  const alpha = runSummary({
    run_key: "item-alpha--shared-run",
    item_id: "item-alpha",
    run_id: "shared-run",
    record_count: 10,
    frame_count: 2,
    stage_counts: { RAW_FUSION: 2 },
  });
  const beta = runSummary({
    run_key: "item-beta--shared-run",
    item_id: "item-beta",
    run_id: "shared-run",
    record_count: 7,
    frame_count: 1,
    stage_counts: { POSE_2D: 1 },
  });
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname === "/api/v1/runs") {
      return jsonResponse({ items: [alpha, beta], offset: 0, limit: 50, total: 2 });
    }
    const selected = url.pathname.includes("item-alpha--shared-run") ? alpha : beta;
    if (url.pathname === `/api/v1/runs/${selected.run_key}`) {
      return jsonResponse({
        run: selected,
        manifest: { run_id: selected.run_id, metadata: { item_id: selected.item_id } },
        summary: { status: "COMPLETED" },
        validation: { ok: true, errors: [], warnings: [] },
        stages: selected === alpha ? ["RAW_FUSION"] : ["POSE_2D"],
        track_ids: selected === alpha ? ["hand-0"] : ["hand-1"],
        view_ids: ["left", "right"],
        global_records: [],
      });
    }
    if (url.pathname === `/api/v1/runs/${selected.run_key}/frames`) {
      return jsonResponse({ items: [], offset: 0, limit: 120, total: 0 });
    }
    return jsonResponse({ error: { message: "not found" } }, 404);
  });

  render(<App />);
  await userEvent.click(await screen.findByRole("button", { name: /item-alpha.*shared-run/i }));
  expect(await within(screen.getByRole("main")).findByText("item-alpha")).toBeVisible();

  await userEvent.selectOptions(screen.getByRole("combobox", { name: "阶段" }), "RAW_FUSION");
  await userEvent.selectOptions(screen.getByRole("combobox", { name: "Track" }), "hand-0");
  await userEvent.selectOptions(screen.getByRole("combobox", { name: "状态" }), "WARNING");
  await userEvent.click(screen.getByRole("button", { name: /item-beta.*shared-run/i }));

  expect(await within(screen.getByRole("main")).findByText("item-beta")).toBeVisible();
  const url = new URL(window.location.href);
  expect(url.searchParams.get("run")).toBe(beta.run_key);
  expect(url.searchParams.has("stage")).toBe(false);
  expect(url.searchParams.has("track")).toBe(false);
  expect(url.searchParams.has("status")).toBe(false);
  expect(url.searchParams.has("frame")).toBe(false);
  expect(url.searchParams.has("offset")).toBe(false);
  expect(fetchMock.mock.calls.some(([input]) => {
    const request = new URL(String(input), window.location.origin);
    return request.pathname === `/api/v1/runs/${beta.run_key}/frames`
      && !request.searchParams.has("stage")
      && !request.searchParams.has("track_id")
      && !request.searchParams.has("status");
  })).toBe(true);
});

test("API failures surface the backend message in an accessible alert", async () => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(
    jsonResponse({ error: { message: "trace storage is temporarily unavailable" } }, 503),
  );

  render(<App />);

  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("trace storage is temporarily unavailable");
});

test("the timeline pages through every result and opens opaque frame keys", async () => {
  window.history.replaceState({}, "", "/?run=paged-run-key");
  const run = runSummary({
    run_key: "paged-run-key",
    run_id: "multipart-run",
    frame_count: 3,
    record_count: 3,
  });
  const frameKeys = ["a".repeat(64), "b".repeat(64), "c".repeat(64)];
  const frames = frameKeys.map((frame_key, index) => ({
    frame_key,
    frame_id: `part-${index + 1}/presentation-frame`,
    frame_index: index === 2 ? null : index,
    timestamp_ns: index * 33_333_333,
    record_ids: [`record-${index}`],
    stages: ["POSE_2D"],
    statuses: ["SUCCEEDED"],
    track_ids: ["hand-0"],
    view_ids: ["left", "right"],
  }));
  const frameOffsets: number[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname === "/api/v1/runs") {
      return jsonResponse({ items: [], offset: 0, limit: 50, total: 0 });
    }
    if (url.pathname === "/api/v1/runs/paged-run-key") {
      return jsonResponse({
        run,
        manifest: { run_id: run.run_id, metadata: { item_id: run.item_id } },
        summary: { status: "COMPLETED" },
        validation: { ok: true, errors: [], warnings: [] },
        stages: ["POSE_2D"],
        track_ids: ["hand-0"],
        view_ids: ["left", "right"],
        global_records: [],
      });
    }
    if (url.pathname === "/api/v1/runs/paged-run-key/frames") {
      const offset = Number(url.searchParams.get("offset") ?? 0);
      frameOffsets.push(offset);
      const items = offset === 2 ? frames.slice(2) : frames.slice(0, 2);
      return jsonResponse({ items, offset, limit: 120, total: 3 });
    }
    if (url.pathname === `/api/v1/runs/paged-run-key/frames/${frameKeys[2]}`) {
      return jsonResponse({
        run_key: run.run_key,
        run_id: run.run_id,
        frame: frames[2],
        records: [],
      });
    }
    return jsonResponse({ error: { message: "not found" } }, 404);
  });

  render(<App />);
  expect(await screen.findByRole("button", { name: /帧 0/i })).toBeVisible();

  const next = screen.getByRole("button", { name: "下一组帧" });
  expect(next).toBeEnabled();
  await userEvent.click(next);
  const multipartFrame = await screen.findByRole("button", { name: /part-3\/presentation-frame/i });
  expect(new URL(window.location.href).searchParams.get("offset")).toBe("2");
  expect(frameOffsets).toContain(2);

  await userEvent.click(multipartFrame);
  expect(new URL(window.location.href).searchParams.get("frame")).toBe(frameKeys[2]);
  expect(await screen.findByText("#—")).toBeVisible();

  await userEvent.click(screen.getByRole("button", { name: "上一组帧" }));
  expect(await screen.findByRole("button", { name: /帧 0/i })).toBeVisible();
  expect(new URL(window.location.href).searchParams.has("offset")).toBe(false);
});

test("the run catalog pages past the first 50 results and opens the 51st run", async () => {
  const runs = Array.from({ length: 51 }, (_, index) => runSummary({
    run_key: `catalog-run-${index + 1}`,
    item_id: `catalog-item-${String(index + 1).padStart(2, "0")}`,
    run_id: `run-${index + 1}`,
  }));
  const runOffsets: number[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname === "/api/v1/runs") {
      const offset = Number(url.searchParams.get("offset") ?? 0);
      runOffsets.push(offset);
      return jsonResponse({
        items: runs.slice(offset, offset + 50),
        offset,
        limit: 50,
        total: runs.length,
      });
    }
    if (url.pathname === "/api/v1/runs/catalog-run-51") {
      return jsonResponse({
        run: runs[50],
        manifest: { run_id: runs[50].run_id, metadata: { item_id: runs[50].item_id } },
        summary: { status: "COMPLETED" },
        validation: { ok: true, errors: [], warnings: [] },
        stages: [],
        track_ids: [],
        view_ids: [],
        global_records: [],
      });
    }
    if (url.pathname === "/api/v1/runs/catalog-run-51/frames") {
      return jsonResponse({ items: [], offset: 0, limit: 120, total: 0 });
    }
    return jsonResponse({ error: { message: "not found" } }, 404);
  });

  render(<App />);
  expect(await screen.findByRole("button", { name: /catalog-item-01/i })).toBeVisible();
  expect(screen.queryByRole("button", { name: /catalog-item-51/i })).not.toBeInTheDocument();

  await userEvent.click(screen.getByRole("button", { name: "下一页运行" }));
  const run51 = await screen.findByRole("button", { name: /catalog-item-51/i });
  expect(new URL(window.location.href).searchParams.get("run_offset")).toBe("50");
  expect(runOffsets).toContain(50);

  await userEvent.click(run51);
  expect(new URL(window.location.href).searchParams.get("run")).toBe("catalog-run-51");
  expect(await within(screen.getByRole("main")).findByText("catalog-item-51")).toBeVisible();
});
