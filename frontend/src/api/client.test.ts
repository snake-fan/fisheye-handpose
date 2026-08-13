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
