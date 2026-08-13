import type { ArtifactRef, RunDetail, TraceRecord } from "../api/types";
import { artifactsOf, payloadOf } from "./trace";

export interface ProvenanceFacts {
  requestSha256?: string;
  modelManifestSha256?: string;
  mmposeCommit?: string;
  calibrationId?: string;
  calibrationSha256?: string;
  manoManifestSha256?: string;
  manoMappingId?: string;
}

function objectValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function textValue(...values: unknown[]): string | undefined {
  return values.find((value): value is string => typeof value === "string" && value.length > 0);
}

function facetPayload(value: unknown): Record<string, unknown> {
  const facet = objectValue(value);
  const payload = objectValue(facet.payload);
  return Object.keys(payload).length > 0 ? payload : facet;
}

function eventPayload(records: TraceRecord[], ...events: string[]): Record<string, unknown> {
  const record = records.find((candidate) => events.includes(String(candidate.event ?? "")));
  return record ? payloadOf(record) : {};
}

function firstPayloadWith(records: TraceRecord[], key: string): Record<string, unknown> {
  const record = records.find((candidate) => payloadOf(candidate)[key] !== undefined);
  return record ? payloadOf(record) : {};
}

function nestedCalibration(payload: Record<string, unknown>): Record<string, unknown> {
  const calibration = objectValue(payload.calibration);
  return Object.keys(calibration).length > 0 ? calibration : payload;
}

export function provenanceFacts(detail: RunDetail, records: TraceRecord[]): ProvenanceFacts {
  const evidence = [...(detail.global_records ?? []), ...records];
  const structuredWorker = facetPayload(detail.provenance?.worker_inputs);
  const workerRecord = eventPayload(evidence, "worker_inputs_verified");
  const worker = { ...workerRecord, ...structuredWorker };

  const structuredCalibration = nestedCalibration(facetPayload(detail.provenance?.calibration));
  const calibrationRecord = nestedCalibration(eventPayload(
    evidence,
    "worker_rectification_loaded",
    "calibration_normalized",
  ));
  const calibration = { ...calibrationRecord, ...structuredCalibration };

  const structuredMano = facetPayload(
    detail.provenance?.mano ?? detail.provenance?.mano_configuration,
  );
  const manoRecord = eventPayload(evidence, "mano_models_loaded", "mano_not_configured");
  const manoFallback = firstPayloadWith(
    evidence.filter((record) => String(record.event ?? "").includes("mano")),
    "mapping_id",
  );
  const mano = { ...manoFallback, ...manoRecord, ...structuredMano };

  const calibrationId = textValue(calibration.calibration_id, calibration.id);
  const calibrationSha256 = textValue(
    calibration.calibration_sha256,
    calibration.calibration_hash,
    calibration.sha256,
    calibrationId?.startsWith("sha256:") ? calibrationId.slice("sha256:".length) : undefined,
  );

  return {
    requestSha256: textValue(worker.request_sha256, worker.worker_request_sha256),
    modelManifestSha256: textValue(
      worker.model_manifest_sha256,
      objectValue(worker.models).manifest_sha256,
    ),
    mmposeCommit: textValue(worker.mmpose_commit),
    calibrationId,
    calibrationSha256,
    manoManifestSha256: textValue(
      mano.manifest_sha256,
      mano.mano_manifest_sha256,
      worker.mano_manifest_sha256,
    ),
    manoMappingId: textValue(mano.mapping_id, mano.mano_mapping_id),
  };
}

function artifactPath(artifact: ArtifactRef): string {
  return String(artifact.relative_path ?? artifact.path ?? "");
}

function artifactRank(artifact: ArtifactRef): number {
  const role = String(artifact.role ?? "");
  if (role === "worker_fhp21_output") return 0;
  if (role.startsWith("worker_")) return 1;
  return 2;
}

export function runArtifacts(detail: RunDetail, records: TraceRecord[]): ArtifactRef[] {
  const seen = new Set<string>();
  const result: ArtifactRef[] = [];
  for (const record of [...(detail.global_records ?? []), ...records]) {
    for (const artifact of artifactsOf(record)) {
      const path = artifactPath(artifact);
      const sha256 = artifact.sha256;
      if (!path || typeof sha256 !== "string" || !sha256) continue;
      const key = `${path}:${sha256}:${String(artifact.role ?? "artifact")}`;
      if (seen.has(key)) continue;
      seen.add(key);
      result.push(artifact);
    }
  }
  return result.sort((left, right) => artifactRank(left) - artifactRank(right));
}

export function bytesDisplay(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return "size —";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}
