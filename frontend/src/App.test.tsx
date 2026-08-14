import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";

import { App } from "./App";

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

test("operator can choose a completed run from the run catalog", async () => {
  vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
    jsonResponse({
      items: [
        {
          run_key: "capture-item--run-a1b2c3",
          item_id: "capture-item",
          run_id: "capture-204253",
          status: "COMPLETED",
          created_at_utc: "2026-08-13T04:02:53Z",
          finalized_at_utc: "2026-08-13T04:04:31Z",
          pipeline_version: "0.1.0",
          record_count: 428,
          frame_count: 120,
          stage_counts: { POSE_2D: 120, RAW_FUSION: 118 },
          warning_count: 2,
          failure_count: 0,
        },
      ],
      offset: 0,
      limit: 50,
      total: 1,
    }),
  );

  render(<App />);

  expect(await screen.findByText("capture-item")).toBeVisible();
  expect(screen.getByText(/capture-204253/)).toBeVisible();
  expect(screen.getByText(/428 records/)).toBeVisible();
  expect(screen.getByText("2 warnings")).toBeVisible();

  await userEvent.click(screen.getByRole("button", { name: /capture-204253/i }));
  expect(new URL(window.location.href).searchParams.get("run")).toBe("capture-item--run-a1b2c3");
});

test("operator can filter the selected run by pipeline stage using URL-backed state", async () => {
  window.history.replaceState({}, "", "/?run=capture-204253");
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname === "/api/v1/runs") {
      return jsonResponse({ items: [], offset: 0, limit: 50, total: 0 });
    }
    if (url.pathname === "/api/v1/runs/capture-204253") {
      return jsonResponse({
        run: {
          run_key: "capture-204253",
          item_id: "capture-item",
          run_id: "capture-204253",
          status: "COMPLETED",
          created_at_utc: "2026-08-13T04:02:53Z",
          finalized_at_utc: "2026-08-13T04:04:31Z",
          pipeline_version: "0.1.0",
          record_count: 428,
          frame_count: 120,
          stage_counts: { POSE_2D: 120, RAW_FUSION: 118 },
          warning_count: 2,
          failure_count: 0,
        },
        manifest: { run_id: "capture-204253" },
        summary: { status: "COMPLETED" },
        validation: { ok: true, errors: [], warnings: ["two source gaps"] },
        stages: ["POSE_2D", "RAW_FUSION", "TEMPORAL_REFINEMENT"],
        track_ids: ["hand-0", "hand-1"],
        view_ids: ["left", "right"],
      });
    }
    if (url.pathname === "/api/v1/runs/capture-204253/frames") {
      const filtered = url.searchParams.get("stage") === "RAW_FUSION";
      return jsonResponse({
        items: [
          {
            frame_key: (filtered ? "b" : "a").repeat(64),
            frame_id: filtered ? "frame/000018" : "frame/000017",
            frame_index: filtered ? 18 : 17,
            timestamp_ns: filtered ? 600_000_000 : 566_666_667,
            record_ids: [filtered ? "fusion-18" : "pose-17"],
            stages: [filtered ? "RAW_FUSION" : "POSE_2D"],
            statuses: ["SUCCEEDED"],
            track_ids: ["hand-0"],
            view_ids: ["left", "right"],
          },
        ],
        offset: 0,
        limit: 120,
        total: 1,
      });
    }
    return jsonResponse({ error: { message: "not found" } }, 404);
  });

  render(<App />);

  expect(await screen.findByRole("heading", { name: "capture-204253" })).toBeVisible();
  expect(screen.getByText("120 帧")).toBeVisible();
  expect(await screen.findByRole("button", { name: /帧 17/i })).toBeVisible();

  await userEvent.selectOptions(screen.getByRole("combobox", { name: "阶段" }), "RAW_FUSION");

  expect(new URL(window.location.href).searchParams.get("stage")).toBe("RAW_FUSION");
  expect(await screen.findByRole("button", { name: /帧 18/i })).toBeVisible();
  expect(screen.queryByRole("button", { name: /帧 17/i })).not.toBeInTheDocument();
});

test("operator can inspect stereo overlays, FHP21, QA, provenance, and raw records for a URL-selected frame", async () => {
  const frameKey = "f".repeat(64);
  window.history.replaceState({}, "", `/?run=capture-204253&frame=${frameKey}`);
  const keypoints = Array.from({ length: 21 }, (_, index) => [120 + index * 4, 90 + index * 3]);
  const landmarks = Array.from(
    { length: 21 },
    (_, index) => [index * 0.004, (index % 5) * 0.009, 0.42 + index * 0.002],
  );
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname === "/api/v1/runs") {
      return jsonResponse({ items: [], offset: 0, limit: 50, total: 0 });
    }
    if (url.pathname === "/api/v1/runs/capture-204253") {
      return jsonResponse({
        run: {
          run_key: "capture-204253",
          item_id: "capture-item",
          run_id: "capture-204253",
          status: "COMPLETED",
          created_at_utc: "2026-08-13T04:02:53Z",
          finalized_at_utc: "2026-08-13T04:04:31Z",
          pipeline_version: "0.1.0",
          record_count: 428,
          frame_count: 120,
          stage_counts: { POSE_2D: 120, RAW_FUSION: 118 },
          warning_count: 1,
          failure_count: 0,
        },
        manifest: {
          run_id: "capture-204253",
        },
        summary: { status: "COMPLETED" },
        validation: { ok: true, errors: [], warnings: ["one frame gap"] },
        provenance: {
          worker_inputs: {
            record_id: "h20:system:verified",
            payload: {
              model_manifest_sha256: "abc123",
              mmpose_commit: "mmpose-5408bc76",
            },
          },
          calibration: {
            record_id: "h20:calibration:rectification",
            payload: { calibration_id: "cal-9e1d" },
          },
        },
        stages: ["DECODE", "POSE_2D", "RAW_FUSION", "QA"],
        track_ids: ["hand-0"],
        view_ids: ["left", "right"],
      });
    }
    if (url.pathname === "/api/v1/runs/capture-204253/frames") {
      return jsonResponse({
        items: [{
          frame_key: frameKey,
          frame_id: "frame/000017",
          frame_index: 17,
          timestamp_ns: 566_666_667,
          record_ids: ["decode-left-17", "decode-right-17", "pose-left-17", "pose-right-17", "fusion-17", "qa-17"],
          stages: ["DECODE", "POSE_2D", "RAW_FUSION", "QA"],
          statuses: ["SUCCEEDED", "WARNING"],
          track_ids: ["hand-0"],
          view_ids: ["left", "right"],
        }],
        offset: 0,
        limit: 120,
        total: 1,
      });
    }
    if (url.pathname === `/api/v1/runs/capture-204253/frames/${frameKey}`) {
      const frame = {
        frame_key: frameKey,
        frame_id: "frame/000017",
        frame_index: 17,
        timestamp_ns: 566_666_667,
        record_ids: [],
        stages: ["DECODE", "POSE_2D", "RAW_FUSION", "QA"],
        statuses: ["SUCCEEDED", "WARNING"],
        track_ids: ["hand-0"],
        view_ids: ["left", "right"],
      };
      return jsonResponse({
        run_id: "capture-204253",
        frame,
        records: [
          {
            record_id: "decode-left-17",
            stage: "DECODE",
            status: "SUCCEEDED",
            event: "frame_decoded",
            parent_ids: [],
            blobs: [{ role: "source_left", relative_path: "blobs/sha256/aa/left.png", media_type: "image/png" }],
            payload: { frame_id: frame.frame_id, view_id: "left", image_width: 640, image_height: 480 },
          },
          {
            record_id: "decode-right-17",
            stage: "DECODE",
            status: "SUCCEEDED",
            event: "frame_decoded",
            parent_ids: [],
            blobs: [{ role: "source_right", relative_path: "blobs/sha256/bb/right.png", media_type: "image/png" }],
            payload: { frame_id: frame.frame_id, view_id: "right", image_width: 640, image_height: 480 },
          },
          {
            record_id: "pose-left-17",
            stage: "POSE_2D",
            status: "SUCCEEDED",
            event: "view_keypoints_inferred",
            parent_ids: ["decode-left-17"],
            payload: {
              frame_id: frame.frame_id,
              track_id: "hand-0",
              view_id: "left",
              keypoints_uv: keypoints,
              keypoint_scores: Array(21).fill(0.97),
              detections: [{ bbox_xyxy: [90, 65, 250, 245], score: 0.98, label: "hand" }],
            },
          },
          {
            record_id: "pose-right-17",
            stage: "POSE_2D",
            status: "SUCCEEDED",
            event: "view_keypoints_inferred",
            parent_ids: ["decode-right-17"],
            payload: {
              frame_id: frame.frame_id,
              track_id: "hand-0",
              view_id: "right",
              keypoints_uv: keypoints.map(([x, y]) => [x - 18, y]),
              keypoint_scores: Array(21).fill(0.95),
            },
          },
          {
            record_id: "fusion-17",
            stage: "RAW_FUSION",
            status: "SUCCEEDED",
            event: "stereo_landmarks_fused",
            parent_ids: ["pose-left-17", "pose-right-17"],
            payload: {
              frame_id: frame.frame_id,
              track_id: "hand-0",
              landmarks_xyz_m: landmarks,
              validity: Array(21).fill(true),
              coordinate_frame: "rig",
              mean_reprojection_error_px: 0.74,
              mapping_id: "fhp21/v1",
            },
          },
          {
            record_id: "qa-17",
            stage: "QA",
            status: "WARNING",
            event: "epipolar_checked",
            parent_ids: ["fusion-17"],
            payload: {
              frame_id: frame.frame_id,
              metric: "epipolar_residual_px",
              value: 1.42,
              threshold: 1.0,
              message: "Residual exceeds review threshold",
            },
          },
        ],
      });
    }
    return jsonResponse({ error: { message: "not found" } }, 404);
  });

  render(<App />);

  expect(await screen.findByText("此运行没有叠加视频")).toBeVisible();
  expect(await screen.findByRole("img", { name: "左目 source_left" })).toBeVisible();
  expect(screen.getByRole("img", { name: "右目 source_right" })).toBeVisible();
  expect(screen.getAllByText("21 / 21 visible")).toHaveLength(2);
  expect(screen.getByRole("img", { name: "FHP21 三维骨架" })).toBeVisible();

  await userEvent.click(screen.getByRole("tab", { name: "QA" }));
  expect(screen.getByText("epipolar_residual_px")).toBeVisible();
  expect(screen.getByText("Residual exceeds review threshold")).toBeVisible();

  await userEvent.click(screen.getByRole("tab", { name: "来源" }));
  expect(screen.getByText("mmpose-5408bc76")).toBeVisible();
  expect(screen.getByText("pose-left-17 + pose-right-17")).toBeVisible();

  await userEvent.click(screen.getByRole("tab", { name: "JSON" }));
  expect(screen.getByText(/"record_id": "fusion-17"/)).toBeVisible();
});

test("operator can search data-item folders and preserve the catalog query in the URL", async () => {
  const fetchMock = vi.spyOn(globalThis, "fetch")
    .mockResolvedValueOnce(jsonResponse({ items: [], offset: 0, limit: 50, total: 0 }))
    .mockResolvedValueOnce(jsonResponse({
      items: [{
        run_key: "Orbbec_Ego_204253--run-20260813-120000",
        item_id: "Orbbec_Ego_204253",
        run_id: "run-20260813-120000",
        data_item_id: "Orbbec_Ego_204253",
        status: "COMPLETED",
        created_at_utc: "2026-08-13T04:02:53Z",
        finalized_at_utc: "2026-08-13T04:04:31Z",
        pipeline_version: "0.1.0",
        record_count: 428,
        frame_count: 120,
        stage_counts: {},
        warning_count: 0,
        failure_count: 0,
      }],
      offset: 0,
      limit: 50,
      total: 1,
    }));

  render(<App />);
  const search = await screen.findByRole("searchbox", { name: "搜索运行" });
  await userEvent.type(search, "Orbbec_Ego_204253{enter}");

  expect(new URL(window.location.href).searchParams.get("q")).toBe("Orbbec_Ego_204253");
  expect(await screen.findByText("Orbbec_Ego_204253")).toBeVisible();
  expect(screen.getByText(/run-20260813-120000/)).toBeVisible();
  expect(fetchMock.mock.calls.some(([input]) => new URL(String(input), window.location.origin).searchParams.get("q") === "Orbbec_Ego_204253")).toBe(true);
});
