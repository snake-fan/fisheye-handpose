import type { ArtifactRef, TraceRecord } from "../api/types";


export type EvidenceOutputStatus = "PRODUCED" | "NOT_PRODUCED" | "UNKNOWN";

export interface FrameEvidenceRecord extends TraceRecord {
  readonly raw: TraceRecord;
  readonly sourceIndex: number;
  readonly record_id: string;
  readonly stage: string;
  readonly status: string;
  readonly event: string;
  readonly payload: Record<string, unknown>;
  readonly viewId: string;
  readonly trackId: string;
  readonly artifactRefs: ArtifactRef[];
  readonly outputStatus: EvidenceOutputStatus;
  readonly failureReason: string;
}

export interface FrameEvidenceArtifact {
  readonly artifact: ArtifactRef;
  readonly record: FrameEvidenceRecord;
  readonly role: string;
  readonly path: string;
}

export interface FrameEvidence {
  readonly sourceRecords: readonly TraceRecord[];
  readonly records: FrameEvidenceRecord[];
  readonly artifacts: FrameEvidenceArtifact[];
  recordsForStage(stage: string): readonly FrameEvidenceRecord[];
  recordsForView(viewId: string): readonly FrameEvidenceRecord[];
  recordsForTrack(trackId: string): readonly FrameEvidenceRecord[];
  artifactsForRole(role: string): readonly FrameEvidenceArtifact[];
  latestForStage(stage: string): FrameEvidenceRecord | undefined;
  hasArtifactRole(role: string): boolean;
}

const EMPTY_RECORDS: readonly FrameEvidenceRecord[] = Object.freeze([]);
const EMPTY_ARTIFACTS: readonly FrameEvidenceArtifact[] = Object.freeze([]);

function objectValue(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function text(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function firstText(value: Record<string, unknown>, keys: readonly string[]): string {
  for (const key of keys) {
    const candidate = text(value[key]);
    if (candidate) return candidate;
  }
  return "";
}

export function payloadFor(record: TraceRecord): Record<string, unknown> {
  return objectValue(record.payload) ?? {};
}

export function artifactPathFor(artifact: ArtifactRef): string {
  const relativePath = text(artifact.relative_path);
  return relativePath || text(artifact.path);
}

export function artifactRefsFor(record: TraceRecord): ArtifactRef[] {
  const normalized = (record as Partial<FrameEvidenceRecord>).artifactRefs;
  if (Array.isArray(normalized)) return normalized;
  return [...(record.blobs ?? []), ...(record.artifacts ?? [])].filter((artifact) => (
    Boolean(artifact)
    && typeof artifact === "object"
    && Boolean(artifactPathFor(artifact))
  ));
}

export function outputStatusFor(record: TraceRecord): EvidenceOutputStatus {
  const normalized = (record as Partial<FrameEvidenceRecord>).outputStatus;
  if (normalized === "PRODUCED" || normalized === "NOT_PRODUCED" || normalized === "UNKNOWN") {
    return normalized;
  }
  const payload = payloadFor(record);
  if (payload.output_status === "PRODUCED") return "PRODUCED";
  if (payload.output_status === "NOT_PRODUCED") return "NOT_PRODUCED";
  if (record.status === "SUCCEEDED") return "PRODUCED";
  if (
    record.status === "FAILED"
    || record.status === "SKIPPED"
    || text(record.event).toLowerCase().includes("not_produced")
  ) return "NOT_PRODUCED";
  return "UNKNOWN";
}

/**
 * Older visual-evidence records did not carry an explicit output status. Keep
 * those records renderable unless the trace explicitly says no output exists.
 */
export function outputAvailableFor(record: TraceRecord): boolean {
  return outputStatusFor(record) !== "NOT_PRODUCED";
}

export function failureReasonFor(record: TraceRecord): string {
  const normalized = (record as Partial<FrameEvidenceRecord>).failureReason;
  if (typeof normalized === "string") return normalized;
  const payload = payloadFor(record);
  const direct = firstText(payload, ["reason", "hand_reason", "failure_reason"]);
  if (direct) return direct;
  const selection = objectValue(payload.selection);
  const gate = objectValue(selection?.gate);
  const gateReason = text(gate?.reason);
  if (gateReason) return gateReason;
  const decision = text(selection?.decision);
  if (decision) return decision;
  const error = objectValue(payload.error);
  const errorReason = firstText(error ?? {}, ["reason", "code", "message", "type"]);
  if (errorReason) return errorReason;
  if (record.status === "FAILED") return "STAGE_FAILED";
  return outputStatusFor(record) === "NOT_PRODUCED" ? "OUTPUT_NOT_PRODUCED" : "";
}

function appendIndex<T>(index: Map<string, T[]>, key: string, value: T): void {
  if (!key) return;
  const existing = index.get(key);
  if (existing) existing.push(value);
  else index.set(key, [value]);
}

function normalizeRecord(raw: TraceRecord, sourceIndex: number): FrameEvidenceRecord {
  const sourcePayload = payloadFor(raw);
  const viewId = firstText(sourcePayload, ["view_id", "view", "side"]).toLowerCase();
  const trackId = firstText(sourcePayload, ["track_id", "track"]);
  const payload = (
    (viewId && sourcePayload.view_id !== viewId)
    || (trackId && sourcePayload.track_id !== trackId)
  )
    ? {
        ...sourcePayload,
        ...(viewId ? { view_id: viewId } : {}),
        ...(trackId ? { track_id: trackId } : {}),
      }
    : sourcePayload;
  const stage = text(raw.stage);
  const status = text(raw.status);
  const event = text(raw.event);
  const artifactRefs = artifactRefsFor(raw);
  const partial: TraceRecord = {
    ...raw,
    record_id: text(raw.record_id) || `${stage || "RECORD"}:${raw.ordinal ?? sourceIndex}`,
    stage,
    status,
    event,
    payload,
  };
  return {
    ...partial,
    raw,
    sourceIndex,
    record_id: partial.record_id as string,
    stage,
    status,
    event,
    payload,
    viewId,
    trackId,
    artifactRefs,
    outputStatus: outputStatusFor(partial),
    failureReason: failureReasonFor(partial),
  };
}

export function createFrameEvidence(sourceRecords: readonly TraceRecord[]): FrameEvidence {
  const byStage = new Map<string, FrameEvidenceRecord[]>();
  const byView = new Map<string, FrameEvidenceRecord[]>();
  const byTrack = new Map<string, FrameEvidenceRecord[]>();
  const byRole = new Map<string, FrameEvidenceArtifact[]>();
  const artifacts: FrameEvidenceArtifact[] = [];
  const records = sourceRecords.map((record, index) => normalizeRecord(record, index));

  for (const record of records) {
    appendIndex(byStage, record.stage, record);
    appendIndex(byView, record.viewId, record);
    appendIndex(byTrack, record.trackId, record);
    for (const artifact of record.artifactRefs) {
      const owned = {
        artifact,
        record,
        role: text(artifact.role),
        path: artifactPathFor(artifact),
      };
      artifacts.push(owned);
      appendIndex(byRole, owned.role, owned);
    }
  }

  return {
    sourceRecords,
    records,
    artifacts,
    recordsForStage: (stage) => byStage.get(stage) ?? EMPTY_RECORDS,
    recordsForView: (viewId) => byView.get(viewId.toLowerCase()) ?? EMPTY_RECORDS,
    recordsForTrack: (trackId) => byTrack.get(trackId) ?? EMPTY_RECORDS,
    artifactsForRole: (role) => byRole.get(role) ?? EMPTY_ARTIFACTS,
    latestForStage: (stage) => byStage.get(stage)?.at(-1),
    hasArtifactRole: (role) => Boolean(byRole.get(role)?.length),
  };
}
