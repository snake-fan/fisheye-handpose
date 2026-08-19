import { expect, test, vi } from "vitest";

import { traceApi } from "./client";

test("network failures are exposed as a retryable Chinese connection error", async () => {
  vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));

  await expect(traceApi.listRuns()).rejects.toMatchObject({
    name: "TraceApiError",
    code: "CONNECTION_FAILED",
    message: "无法连接到追踪 API，请确认后端服务或 SSH 隧道是否正在运行。",
    retryable: true,
  });
});

test("immutable JSON artifacts use the encoded artifact route and browser cache", async () => {
  const signal = new AbortController().signal;
  const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
    ok: true,
    json: async () => ({ schema_version: "timeline/test" }),
  } as Response);

  await expect(traceApi.getArtifactJson(
    "run/key",
    "blobs/sha256/aa/frame timeline.json",
    signal,
  )).resolves.toEqual({ schema_version: "timeline/test" });
  expect(fetchMock).toHaveBeenCalledWith(
    "/api/v1/runs/run%2Fkey/artifacts/blobs/sha256/aa/frame%20timeline.json",
    expect.objectContaining({
      cache: "force-cache",
      headers: { Accept: "application/json" },
      signal,
    }),
  );
});


test.each([
  "",
  "/blobs/frame.jpg",
  "blobs/frame.jpg/",
  "blobs//frame.jpg",
  ".",
  "..",
  "blobs/./frame.jpg",
  "blobs/../frame.jpg",
  "blobs\\frame.jpg",
  "blobs/\0frame.jpg",
])("artifact URLs reject ambiguous or unsafe paths without normalization: %j", (path) => {
  expect(() => traceApi.artifactUrl("run-key", path)).toThrow(/artifact path/i);
});
