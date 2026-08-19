import type {
  FrameDetail,
  FrameQuery,
  FrameSummary,
  Page,
  RunDetail,
  RunQuery,
  RunSummary,
} from "./types";

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim() ?? "";
const apiBaseUrl = configuredBaseUrl.replace(/\/$/, "");

export type TraceApiErrorCode = "CONNECTION_FAILED" | "HTTP_ERROR";

export class TraceApiError extends Error {
  readonly code: TraceApiErrorCode;
  readonly retryable: boolean;

  constructor(message: string, code: TraceApiErrorCode, retryable: boolean) {
    super(message);
    this.name = "TraceApiError";
    this.code = code;
    this.retryable = retryable;
  }
}

function withQuery<T extends object>(path: string, query: T): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== "") params.set(key, String(value));
  }
  const suffix = params.toString();
  return `${apiBaseUrl}${path}${suffix ? `?${suffix}` : ""}`;
}

async function getJson<T>(
  url: string,
  signal?: AbortSignal,
  cache: RequestCache = "no-store",
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, {
      cache,
      headers: { Accept: "application/json" },
      signal,
    });
  } catch (reason) {
    if (reason instanceof DOMException && reason.name === "AbortError") throw reason;
    if (signal?.aborted) throw reason;
    throw new TraceApiError(
      "无法连接到追踪 API，请确认后端服务或 SSH 隧道是否正在运行。",
      "CONNECTION_FAILED",
      true,
    );
  }
  if (!response.ok) {
    let message = `请求失败（HTTP ${response.status}）`;
    try {
      const body = (await response.json()) as { error?: { message?: string } };
      if (body.error?.message) message = body.error.message;
    } catch {
      // Keep the status-based fallback for a non-JSON error response.
    }
    throw new TraceApiError(message, "HTTP_ERROR", response.status >= 500);
  }
  return response.json() as Promise<T>;
}

function segment(value: string | number): string {
  return encodeURIComponent(String(value));
}

function artifactUrl(runKey: string, relativePath: string): string {
  if (relativePath.includes("\\") || relativePath.includes("\0")) {
    throw new TypeError("artifact path contains a forbidden character");
  }
  const segments = relativePath.split("/");
  if (
    segments.length === 0
    || segments.some((value) => value === "" || value === "." || value === "..")
  ) {
    throw new TypeError("artifact path contains an unsafe segment");
  }
  return `${apiBaseUrl}/api/v1/runs/${segment(runKey)}/artifacts/${segments.map(segment).join("/")}`;
}

export const traceApi = {
  listRuns(query: RunQuery = {}, signal?: AbortSignal): Promise<Page<RunSummary>> {
    return getJson(withQuery("/api/v1/runs", query), signal);
  },

  getRun(runKey: string, signal?: AbortSignal): Promise<RunDetail> {
    return getJson(`${apiBaseUrl}/api/v1/runs/${segment(runKey)}`, signal);
  },

  listFrames(
    runKey: string,
    query: FrameQuery = {},
    signal?: AbortSignal,
  ): Promise<Page<FrameSummary>> {
    return getJson(
      withQuery(`/api/v1/runs/${segment(runKey)}/frames`, query),
      signal,
    );
  },

  getFrame(runKey: string, frameKey: string, signal?: AbortSignal): Promise<FrameDetail> {
    return getJson(
      `${apiBaseUrl}/api/v1/runs/${segment(runKey)}/frames/${segment(frameKey)}`,
      signal,
    );
  },

  artifactUrl(runKey: string, relativePath: string): string {
    return artifactUrl(runKey, relativePath);
  },

  getArtifactJson<T>(runKey: string, relativePath: string, signal?: AbortSignal): Promise<T> {
    return getJson<T>(artifactUrl(runKey, relativePath), signal, "force-cache");
  },
};
