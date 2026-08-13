import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";

import { App } from "./App";

function jsonResponse(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    headers: { "Content-Type": "application/json" },
  });
}

const frameKey = "c".repeat(64);
const run = {
  run_key: "retry-run",
  item_id: "retry-item",
  run_id: "retry-run-id",
  status: "COMPLETED",
  created_at_utc: "2026-08-13T04:02:53Z",
  finalized_at_utc: "2026-08-13T04:04:31Z",
  pipeline_version: "0.1.0",
  record_count: 4,
  frame_count: 1,
  stage_counts: { POSE_2D: 1 },
  warning_count: 0,
  failure_count: 0,
};
const frame = {
  frame_key: frameKey,
  frame_id: "frame/000000",
  frame_index: 0,
  timestamp_ns: 0,
  record_ids: [],
  stages: ["POSE_2D"],
  statuses: ["SUCCEEDED"],
  track_ids: ["hand-0"],
  view_ids: ["left", "right"],
};

test("operator can retry every selected-run request after the API connection returns", async () => {
  window.history.replaceState({}, "", `/?run=${run.run_key}&frame=${frameKey}`);
  let offline = true;
  const requestedPaths: string[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = new URL(String(input), window.location.origin);
    requestedPaths.push(url.pathname);
    if (offline) throw new TypeError("Failed to fetch");
    if (url.pathname === "/api/v1/runs") {
      return jsonResponse({ items: [run], offset: 0, limit: 50, total: 1 });
    }
    if (url.pathname === `/api/v1/runs/${run.run_key}`) {
      return jsonResponse({
        run,
        manifest: { run_id: run.run_id },
        summary: { status: "COMPLETED" },
        validation: { ok: true, errors: [], warnings: [] },
        stages: ["POSE_2D"],
        track_ids: ["hand-0"],
        view_ids: ["left", "right"],
        global_records: [],
      });
    }
    if (url.pathname === `/api/v1/runs/${run.run_key}/frames`) {
      expect(url.searchParams.get("limit")).toBe("40");
      return jsonResponse({ items: [frame], offset: 0, limit: 40, total: 1 });
    }
    if (url.pathname === `/api/v1/runs/${run.run_key}/frames/${frameKey}`) {
      return jsonResponse({ run_key: run.run_key, run_id: run.run_id, frame, records: [] });
    }
    throw new Error(`unexpected request ${url.pathname}`);
  });

  render(<App />);

  expect(await screen.findByRole("status", { name: "API 已断开" })).toBeVisible();
  expect(screen.getAllByRole("alert")[0]).toHaveTextContent(
    "无法连接到追踪 API，请确认后端服务或 SSH 隧道是否正在运行。",
  );

  offline = false;
  await userEvent.click(screen.getByRole("button", { name: "重试连接" }));

  expect(await screen.findByRole("status", { name: "API 已连接" })).toBeVisible();
  expect(await screen.findByRole("heading", { name: run.run_id })).toBeVisible();
  await waitFor(() => {
    expect(requestedPaths.filter((path) => path === "/api/v1/runs")).toHaveLength(2);
    expect(requestedPaths.filter((path) => path === `/api/v1/runs/${run.run_key}`)).toHaveLength(2);
    expect(requestedPaths.filter((path) => path === `/api/v1/runs/${run.run_key}/frames`)).toHaveLength(2);
    expect(
      requestedPaths.filter(
        (path) => path === `/api/v1/runs/${run.run_key}/frames/${frameKey}`,
      ),
    ).toHaveLength(2);
  });
});

test("switching runs clears stale detail while the new request is pending", async () => {
  window.history.replaceState({}, "", "/?run=run-alpha");
  const alpha = { ...run, run_key: "run-alpha", item_id: "item-alpha", run_id: "alpha-detail" };
  const beta = { ...run, run_key: "run-beta", item_id: "item-beta", run_id: "beta-detail" };
  let resolveBeta!: (response: Response) => void;
  const betaDetail = new Promise<Response>((resolve) => { resolveBeta = resolve; });
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname === "/api/v1/runs") {
      return jsonResponse({ items: [alpha, beta], offset: 0, limit: 50, total: 2 });
    }
    if (url.pathname.endsWith("/frames")) {
      return jsonResponse({ items: [], offset: 0, limit: 40, total: 0 });
    }
    if (url.pathname === "/api/v1/runs/run-beta") return betaDetail;
    const selected = url.pathname.endsWith("run-alpha") ? alpha : beta;
    return jsonResponse({
      run: selected,
      manifest: { run_id: selected.run_id },
      summary: { status: "COMPLETED" },
      validation: { ok: true, errors: [], warnings: [] },
      stages: ["POSE_2D"],
      track_ids: [],
      view_ids: [],
      global_records: [],
    });
  });

  render(<App />);
  expect(await screen.findByRole("heading", { name: "alpha-detail" })).toBeVisible();

  await userEvent.click(screen.getByRole("button", { name: /item-beta.*beta-detail/i }));
  expect(screen.queryByRole("heading", { name: "alpha-detail" })).not.toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "正在打开运行…" })).toBeVisible();

  resolveBeta(jsonResponse({
    run: beta,
    manifest: { run_id: beta.run_id },
    summary: { status: "COMPLETED" },
    validation: { ok: true, errors: [], warnings: [] },
    stages: ["POSE_2D"],
    track_ids: [],
    view_ids: [],
    global_records: [],
  }));
  expect(await screen.findByRole("heading", { name: "beta-detail" })).toBeVisible();
  expect(screen.queryByRole("heading", { name: "正在打开运行…" })).not.toBeInTheDocument();
});

test("rapid catalog typing waits 250 ms and only requests the final query", async () => {
  const requestedQueries: string[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname === "/api/v1/runs") requestedQueries.push(url.searchParams.get("q") ?? "");
    return jsonResponse({ items: [], offset: 0, limit: 50, total: 0 });
  });

  render(<App />);
  const search = await screen.findByRole("searchbox", { name: "搜索运行" });
  fireEvent.change(search, { target: { value: "Orbb" } });
  fireEvent.change(search, { target: { value: "Orbbec_Ego" } });

  expect(requestedQueries.filter(Boolean)).toEqual([]);
  await waitFor(() => expect(requestedQueries.filter(Boolean)).toEqual(["Orbbec_Ego"]), {
    timeout: 700,
  });
});

test("a later successful request restores the connection indicator", async () => {
  window.history.replaceState({}, "", `/?run=${run.run_key}`);
  let catalogFailed = false;
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname === "/api/v1/runs") {
      if (!catalogFailed) {
        catalogFailed = true;
        throw new TypeError("Failed to fetch");
      }
      return jsonResponse({ items: [run], offset: 0, limit: 50, total: 1 });
    }
    if (url.pathname.endsWith("/frames")) {
      return jsonResponse({ items: [], offset: 0, limit: 40, total: 0 });
    }
    return jsonResponse({
      run,
      manifest: { run_id: run.run_id },
      summary: { status: "COMPLETED" },
      validation: { ok: true, errors: [], warnings: [] },
      stages: [],
      track_ids: [],
      view_ids: [],
      global_records: [],
    });
  });

  render(<App />);

  expect(await screen.findByRole("status", { name: "API 已连接" })).toBeVisible();
});

test("one successful request wins over a concurrent network failure", async () => {
  window.history.replaceState({}, "", `/?run=${run.run_key}`);
  let resolveRun!: (response: Response) => void;
  const runResponse = new Promise<Response>((resolve) => { resolveRun = resolve; });
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname === "/api/v1/runs") throw new TypeError("Failed to fetch");
    if (url.pathname.endsWith("/frames")) {
      return jsonResponse({ items: [], offset: 0, limit: 40, total: 0 });
    }
    return runResponse;
  });

  render(<App />);
  resolveRun(jsonResponse({
    run,
    manifest: { run_id: run.run_id },
    summary: { status: "COMPLETED" },
    validation: { ok: true, errors: [], warnings: [] },
    stages: [],
    track_ids: [],
    view_ids: [],
    global_records: [],
  }));

  expect(await screen.findByRole("status", { name: "API 已连接" })).toBeVisible();
});

test("a disconnected app probes the API again without requiring a click", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  window.history.replaceState({}, "", "/");
  let calls = 0;
  vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
    calls += 1;
    if (calls === 1) throw new TypeError("Failed to fetch");
    return jsonResponse({ items: [], offset: 0, limit: 50, total: 0 });
  });

  render(<App />);
  expect(await screen.findByRole("status", { name: "API 已断开" })).toBeVisible();
  await vi.advanceTimersByTimeAsync(2_000);
  expect(await screen.findByRole("status", { name: "API 已连接" })).toBeVisible();
  vi.useRealTimers();
});
