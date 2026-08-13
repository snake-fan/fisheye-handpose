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

function withQuery<T extends object>(path: string, query: T): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== "") params.set(key, String(value));
  }
  const suffix = params.toString();
  return `${apiBaseUrl}${path}${suffix ? `?${suffix}` : ""}`;
}

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, {
    cache: "no-store",
    headers: { Accept: "application/json" },
    signal,
  });
  if (!response.ok) {
    let message = `请求失败（HTTP ${response.status}）`;
    try {
      const body = (await response.json()) as { error?: { message?: string } };
      if (body.error?.message) message = body.error.message;
    } catch {
      // Keep the status-based fallback for a non-JSON error response.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

function segment(value: string | number): string {
  return encodeURIComponent(String(value));
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
    const normalized = relativePath.split("/").filter(Boolean).map(segment).join("/");
    return `${apiBaseUrl}/api/v1/runs/${segment(runKey)}/artifacts/${normalized}`;
  },
};
