import type { ArtifactRef, TraceRecord } from "../api/types";

export const FHP21_EDGES: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
];

export const FHP21_NAMES: readonly string[] = [
  "wrist_center",
  "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
  "index_mcp", "index_pip", "index_dip", "index_tip",
  "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
  "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
  "little_mcp", "little_pip", "little_dip", "little_tip",
];

export const FINGER_COLORS = ["#75f6c4", "#63d7e5", "#7ba6ff", "#f2c66d", "#ef86b8"];

export type Point2 = [number, number];
export type Point3 = [number, number, number];

export function payloadOf(record: TraceRecord): Record<string, unknown> {
  return record.payload && typeof record.payload === "object" ? record.payload : {};
}

function objectInstances(payload: Record<string, unknown>): Record<string, unknown>[] {
  if (!Array.isArray(payload.instances)) return [];
  return payload.instances.filter(
    (instance): instance is Record<string, unknown> => Boolean(instance && typeof instance === "object"),
  );
}

export function posePayloadOf(record: TraceRecord, trackId = ""): Record<string, unknown> {
  const payload = payloadOf(record);
  if (points2(payload.keypoints_uv).length > 0) return payload;
  const instances = objectInstances(payload);
  const selected = instances.find((instance) => (
    trackId && (instance.track_id === trackId || instance.candidate_id === trackId)
  )) ?? instances[0];
  return selected ? { ...payload, ...selected } : payload;
}

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function points2(value: unknown): Point2[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((point) => {
    if (!Array.isArray(point) || !finiteNumber(point[0]) || !finiteNumber(point[1])) return [];
    return [[point[0], point[1]] as Point2];
  });
}

export function points3(value: unknown): Array<Point3 | null> {
  if (!Array.isArray(value)) return [];
  return value.map((point) => {
    if (
      !Array.isArray(point)
      || !finiteNumber(point[0])
      || !finiteNumber(point[1])
      || !finiteNumber(point[2])
    ) return null;
    return [point[0], point[1], point[2]];
  });
}

export function scoreValues(value: unknown): number[] {
  if (!Array.isArray(value)) return [];
  return value.map((score) => finiteNumber(score) ? score : 0);
}

export function validityValues(value: unknown, count: number): boolean[] {
  if (!Array.isArray(value)) return Array.from({ length: count }, () => true);
  return Array.from({ length: count }, (_, index) => {
    const flag = value[index];
    return flag === true || flag === "VALID" || flag === 1;
  });
}

export function artifactsOf(record: TraceRecord): ArtifactRef[] {
  return [...(record.blobs ?? []), ...(record.artifacts ?? [])].filter((artifact) => {
    return typeof artifact.relative_path === "string" || typeof artifact.path === "string";
  });
}

function artifactView(artifact: ArtifactRef): string {
  const role = String(artifact.role ?? "").toLowerCase();
  if (role.includes("left")) return "left";
  if (role.includes("right")) return "right";
  return "";
}

function artifactRank(artifact: ArtifactRef): number {
  const role = String(artifact.role ?? "").toLowerCase();
  if (role.startsWith("source")) return 0;
  if (role.startsWith("rectified")) return 1;
  if (role === "crop") return 2;
  if (role === "overlay") return 3;
  return 4;
}

export function viewArtifacts(records: TraceRecord[], viewId: string): ArtifactRef[] {
  const seen = new Set<string>();
  const result: ArtifactRef[] = [];
  for (const record of records) {
    const recordView = String(payloadOf(record).view_id ?? "").toLowerCase();
    for (const artifact of artifactsOf(record)) {
      const inferredView = artifactView(artifact);
      if (recordView !== viewId && inferredView !== viewId) continue;
      const path = String(artifact.relative_path ?? artifact.path ?? "");
      const key = `${path}:${String(artifact.role ?? "artifact")}`;
      if (seen.has(key)) continue;
      seen.add(key);
      result.push(artifact);
    }
  }
  return result.sort((a, b) => artifactRank(a) - artifactRank(b));
}

export function viewPoseRecord(
  records: TraceRecord[],
  viewId: string,
  trackId = "",
): TraceRecord | undefined {
  return [...records].reverse().find((record) => {
    const payload = payloadOf(record);
    if (payload.view_id !== viewId || points2(posePayloadOf(record, trackId).keypoints_uv).length === 0) {
      return false;
    }
    if (!trackId || Array.isArray(payload.instances)) return true;
    return payload.track_id === trackId;
  });
}

export function viewImageSize(records: TraceRecord[], viewId: string): [number, number] {
  for (const record of records) {
    const payload = payloadOf(record);
    if (payload.view_id !== viewId) continue;
    const width = payload.image_width ?? payload.width;
    const height = payload.image_height ?? payload.height;
    if (finiteNumber(width) && finiteNumber(height) && width > 0 && height > 0) {
      return [width, height];
    }
    const size = payload.image_size;
    if (Array.isArray(size) && finiteNumber(size[0]) && finiteNumber(size[1])) {
      return [size[0], size[1]];
    }
  }
  return [1920, 1080];
}

const THREE_D_STAGE_RANK: Record<string, number> = {
  RAW_FUSION: 1,
  KINEMATIC_REFINEMENT: 2,
  TEMPORAL_REFINEMENT: 3,
  EXPORT: 4,
};

export function best3dRecord(records: TraceRecord[], trackId = ""): TraceRecord | undefined {
  return records
    .filter((record) => {
      const payload = payloadOf(record);
      if (points3(payload.landmarks_xyz_m).length !== 21) return false;
      return !trackId || payload.track_id === trackId;
    })
    .sort((a, b) => (THREE_D_STAGE_RANK[b.stage ?? ""] ?? 0) - (THREE_D_STAGE_RANK[a.stage ?? ""] ?? 0))[0];
}

export function recordLabel(record: TraceRecord): string {
  return record.record_id ?? `${record.stage ?? "RECORD"}:${record.ordinal ?? "?"}`;
}

export function asDisplay(value: unknown, fallback = "—"): string {
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}
