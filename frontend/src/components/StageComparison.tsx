import { useState } from "react";
import { ImageIcon } from "lucide-react";

import { traceApi } from "../api/client";
import type { ArtifactRef, TraceRecord } from "../api/types";
import { artifactsOf, FHP21_EDGES, FHP21_NAMES, payloadOf } from "../domain/trace";
import type { PipelineNodeId } from "./PipelineNodeRail";

interface StageComparisonProps {
  runKey: string;
  records: TraceRecord[];
  selectedNodeId: PipelineNodeId;
  selectedTrack: string;
}

type Side = "left" | "right";
type Point2 = [number, number];
type NullablePoint2 = Point2 | null;

interface PoseLayer {
  id: string;
  points: NullablePoint2[];
  candidateId?: string;
}

interface DetectionLayer {
  id: string;
  bbox: [number, number, number, number];
  score: number | null;
  classification?: "SEED" | "RECOVERY" | "REJECTED";
  reason?: string;
  eligible?: boolean;
  sourceIndex?: number;
  inPool?: boolean;
}

interface CandidateAudit {
  decisions: DetectionLayer[];
  pool: DetectionLayer[];
}

interface AssociationMatch {
  matchId: string;
  leftCandidateId: string;
  rightCandidateId: string;
  trackId: string | null;
}

const SIDES: readonly Side[] = ["left", "right"];
// Kept byte-for-byte aligned with worker.visualization.track_color_rgb.
const TRACK_PALETTE = ["#75f6c4", "#ffb454", "#7ba6ff", "#ef86b8", "#c4f06a"];

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function pointList(value: unknown): NullablePoint2[] {
  if (!Array.isArray(value) || value.length !== 21) return [];
  return value.map((point) => {
    if (!Array.isArray(point) || !finiteNumber(point[0]) || !finiteNumber(point[1])) return null;
    return [point[0], point[1]];
  });
}

function hasVisiblePoint(layer: PoseLayer): boolean {
  return layer.points.some((point) => point !== null);
}

function artifactPath(artifact: ArtifactRef): string {
  return String(artifact.relative_path ?? artifact.path ?? "");
}

function artifactByRole(records: TraceRecord[], role: string): ArtifactRef | undefined {
  return records.flatMap(artifactsOf).find((artifact) => artifact.role === role);
}

function sideLabel(side: Side): string {
  return side === "left" ? "左目" : "右目";
}

function outputProduced(record: TraceRecord): boolean {
  const status = payloadOf(record).output_status;
  if (status === "NOT_PRODUCED") return false;
  if (status === "PRODUCED") return true;
  return record.status !== "SKIPPED" && record.status !== "FAILED";
}

function failureReason(record: TraceRecord): string {
  const payload = payloadOf(record);
  if (typeof payload.reason === "string" && payload.reason) return payload.reason;
  if (typeof payload.hand_reason === "string" && payload.hand_reason) return payload.hand_reason;
  if (payload.selection && typeof payload.selection === "object") {
    const selection = payload.selection as Record<string, unknown>;
    const gate = recordObject(selection.gate);
    if (typeof gate?.reason === "string" && gate.reason) return gate.reason;
    const decision = selection.decision;
    if (typeof decision === "string" && decision) return decision;
  }
  return record.status === "FAILED" ? "STAGE_FAILED" : "OUTPUT_NOT_PRODUCED";
}

function trackColor(identity: string): string {
  const match = /^track-(\d+)$/.exec(identity);
  if (match) return TRACK_PALETTE[Number(match[1]) % TRACK_PALETTE.length];
  let hash = 0;
  for (const character of identity) hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  return TRACK_PALETTE[hash % TRACK_PALETTE.length];
}

function EvidenceImage({
  runKey,
  artifact,
  label,
}: {
  runKey: string;
  artifact: ArtifactRef | undefined;
  label: string;
}) {
  if (!artifact) {
    return (
      <div className="comparison-empty">
        <ImageIcon aria-hidden="true" />
        <span>NOT_PRODUCED</span>
      </div>
    );
  }
  return <img src={traceApi.artifactUrl(runKey, artifactPath(artifact))} alt={label} />;
}

function StereoImageComparison({
  runKey,
  records,
  beforePrefix,
  afterPrefix,
}: {
  runKey: string;
  records: TraceRecord[];
  beforePrefix: string;
  afterPrefix: string;
}) {
  return (
    <div className="debug-branch-comparison">
      <header className="debug-only-banner">DEBUG_ONLY QA BRANCH · DETECTION INPUT REMAINS NATIVE FISHEYE RGB</header>
      <div className="stage-comparison-stereo">
        {SIDES.map((side) => {
        const label = sideLabel(side);
        const beforeRole = `${beforePrefix}_${side}`;
        const afterRole = `${afterPrefix}_${side}`;
        return (
          <article className="stage-comparison-view" key={side}>
            <header><strong>{label}</strong><span>BEFORE → AFTER</span></header>
            <div className="stage-comparison-pair">
              <figure>
                <EvidenceImage
                  runKey={runKey}
                  artifact={artifactByRole(records, beforeRole)}
                  label={`${label} before ${beforeRole}`}
                />
                <figcaption>BEFORE · {beforeRole}</figcaption>
              </figure>
              <figure>
                <EvidenceImage
                  runKey={runKey}
                  artifact={artifactByRole(records, afterRole)}
                  label={`${label} after ${afterRole}`}
                />
                <figcaption>AFTER · {afterRole}</figcaption>
              </figure>
            </div>
          </article>
        );
        })}
      </div>
    </div>
  );
}

function SourceComparison({ runKey, records }: { runKey: string; records: TraceRecord[] }) {
  return (
    <div className="source-comparison">
      <header className="coordinate-space-banner">PIPELINE INPUT · NATIVE FISHEYE PIXELS</header>
      <div className="stage-comparison-stereo">
        {SIDES.map((side) => {
          const label = sideLabel(side);
          const role = `source_${side}`;
          return (
            <figure className="stage-comparison-view" key={side}>
              <EvidenceImage
                runKey={runKey}
                artifact={artifactByRole(records, role)}
                label={`${label} Pipeline ${role}`}
              />
              <figcaption>{label} · {role}</figcaption>
            </figure>
          );
        })}
      </div>
    </div>
  );
}

function imageSize(records: TraceRecord[], side: Side): [number, number] {
  for (const record of records) {
    const payload = payloadOf(record);
    if (payload.view_id !== side) continue;
    if (
      finiteNumber(payload.image_width)
      && finiteNumber(payload.image_height)
      && payload.image_width > 0
      && payload.image_height > 0
    ) return [payload.image_width, payload.image_height];
  }
  return [1920, 1080];
}

function rectifiedImageSize(records: TraceRecord[]): [number, number] {
  for (const record of records) {
    if (record.stage !== "RECTIFICATION") continue;
    const payload = payloadOf(record);
    const width = payload.image_width ?? payload.output_width;
    const height = payload.image_height ?? payload.output_height;
    if (finiteNumber(width) && finiteNumber(height) && width > 0 && height > 0) {
      return [width, height];
    }
    if (
      Array.isArray(payload.output_size)
      && finiteNumber(payload.output_size[0])
      && finiteNumber(payload.output_size[1])
    ) return [payload.output_size[0], payload.output_size[1]];
  }
  return [1920, 1080];
}

function objectValues(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) return [];
  return value.filter(
    (item): item is Record<string, unknown> => Boolean(item && typeof item === "object"),
  );
}

function classificationOf(value: unknown): DetectionLayer["classification"] {
  return value === "SEED" || value === "RECOVERY" || value === "REJECTED"
    ? value
    : undefined;
}

function detectionLayer(
  value: Record<string, unknown>,
  fallbackId: string,
): DetectionLayer | null {
  const bbox = value.bbox_xyxy;
  if (!Array.isArray(bbox) || bbox.length !== 4 || !bbox.every(finiteNumber)) return null;
  const score = finiteNumber(value.bbox_score)
    ? value.bbox_score
    : finiteNumber(value.score) ? value.score : null;
  return {
    id: String(value.candidate_id ?? fallbackId),
    bbox: [bbox[0], bbox[1], bbox[2], bbox[3]],
    score,
    classification: classificationOf(value.classification),
    reason: typeof value.reason === "string" ? value.reason : undefined,
    eligible: typeof value.eligible_for_association === "boolean"
      ? value.eligible_for_association
      : undefined,
    sourceIndex: finiteNumber(value.source_index) ? value.source_index : undefined,
  };
}

function detectionsForSide(records: TraceRecord[], side: Side): DetectionLayer[] {
  const found = new Map<string, DetectionLayer>();
  for (const record of records) {
    if (record.stage !== "DETECTION") continue;
    const payload = payloadOf(record);
    if (payload.view_id !== side) continue;
    const detections = objectValues(payload.detections).length
      ? objectValues(payload.detections)
      : objectValues(payload.instances);
    detections.forEach((detection, index) => {
      const parsed = detectionLayer(detection, `${side}-candidate-${index}`);
      if (parsed) found.set(parsed.id, parsed);
    });
  }
  return [...found.values()].sort((left, right) => left.id.localeCompare(right.id));
}

function candidateAuditForSide(records: TraceRecord[], side: Side): CandidateAudit | null {
  const record = [...records].reverse().find((candidate) => {
    const payload = payloadOf(candidate);
    return candidate.stage === "DETECTION"
      && payload.view_id === side
      && Array.isArray(payload.candidate_decisions);
  });
  if (!record) return null;
  const payload = payloadOf(record);
  const decisions = objectValues(payload.candidate_decisions).flatMap((value, index) => {
    const parsed = detectionLayer(value, `${side}-candidate-${index}`);
    return parsed ? [parsed] : [];
  });
  const pool = objectValues(payload.candidate_pool).flatMap((value, index) => {
    const parsed = detectionLayer(value, `${side}-pool-${index}`);
    return parsed ? [parsed] : [];
  });
  const poolIds = new Set(pool.map((candidate) => candidate.id));
  return {
    decisions: decisions.map((candidate) => ({ ...candidate, inPool: poolIds.has(candidate.id) })),
    pool: pool.map((candidate) => ({ ...candidate, inPool: true })),
  };
}

function poseLayers(records: TraceRecord[], side: Side, space: "native" | "rectified"): PoseLayer[] {
  const candidates = new Map<string, PoseLayer>();
  const candidateTracks = new Map<string, string>();
  const direct: PoseLayer[] = [];
  const field = space === "rectified" ? "keypoints_uv_rectified" : "keypoints_uv";

  for (const record of records) {
    if (record.stage !== "POSE_2D") continue;
    const payload = payloadOf(record);
    if (payload.view_id !== side) continue;
    const candidateId = typeof payload.candidate_id === "string" ? payload.candidate_id : "";
    const trackId = typeof payload.track_id === "string" ? payload.track_id : "";
    if (candidateId && trackId) candidateTracks.set(candidateId, trackId);
    const points = pointList(payload[field]);
    if (points.length === 21 && points.some(Boolean)) {
      direct.push({
        id: trackId || candidateId || `${side}-pose-${direct.length}`,
        points,
        candidateId: candidateId || undefined,
      });
    }
    objectValues(payload.instances).forEach((instance, index) => {
      const instancePoints = pointList(instance[field]);
      if (instancePoints.length !== 21 || !instancePoints.some(Boolean)) return;
      const id = String(instance.candidate_id ?? instance.track_id ?? `${side}-candidate-${index}`);
      candidates.set(id, { id, points: instancePoints, candidateId: id });
    });
  }

  const layers: PoseLayer[] = [];
  for (const [candidateId, layer] of candidates) {
    layers.push({ ...layer, id: candidateTracks.get(candidateId) ?? candidateId, candidateId });
  }
  for (const layer of direct) {
    if (!layers.some((candidate) => candidate.id === layer.id)) layers.push(layer);
  }
  return [...new Map(layers.map((layer) => [layer.id, layer])).values()]
    .sort((left, right) => left.id.localeCompare(right.id));
}

function projectedLayers(records: TraceRecord[], stage: string, side: Side): PoseLayer[] {
  const layers: PoseLayer[] = [];
  for (const record of records) {
    if (record.stage !== stage || !outputProduced(record)) continue;
    const payload = payloadOf(record);
    if (payload.projected_keypoints_space !== "rectified") continue;
    if (!payload.projected_keypoints_uv || typeof payload.projected_keypoints_uv !== "object") continue;
    const points = pointList((payload.projected_keypoints_uv as Record<string, unknown>)[side]);
    if (points.length !== 21 || !points.some(Boolean)) continue;
    const identity = payload.track_id;
    if (typeof identity !== "string" || !identity) continue;
    layers.push({ id: identity, points });
  }
  return [...new Map(layers.map((layer) => [layer.id, layer])).values()]
    .sort((left, right) => left.id.localeCompare(right.id));
}

function SkeletonGraphic({
  layers,
  selectedTrack,
  width,
  height,
  label,
  layerLabel,
}: {
  layers: PoseLayer[];
  selectedTrack: string;
  width: number;
  height: number;
  label: string;
  layerLabel?: string;
}) {
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={label}
    >
      {layers.filter(hasVisiblePoint).map((layer) => {
        const highlighted = !selectedTrack || selectedTrack === layer.id;
        const color = trackColor(layer.id);
        return (
          <g
            key={layer.id}
            aria-label={`${layer.id} ${layerLabel ?? label.replace(/^.*?\s/, "")}`}
            data-track-id={layer.id}
            opacity={highlighted ? 1 : 0.24}
            stroke={color}
            fill={color}
          >
            {FHP21_EDGES.map(([from, to]) => {
              const start = layer.points[from];
              const end = layer.points[to];
              if (!start || !end) return null;
              return (
                <line
                  key={`${from}-${to}`}
                  x1={start[0]}
                  y1={start[1]}
                  x2={end[0]}
                  y2={end[1]}
                  vectorEffect="non-scaling-stroke"
                />
              );
            })}
            {layer.points.map((point, index) => point && (
              <circle
                key={`${index}:${point[0]}:${point[1]}`}
                data-joint-index={index}
                cx={point[0]}
                cy={point[1]}
                r="3.5"
              />
            ))}
          </g>
        );
      })}
    </svg>
  );
}

function TrackLegend({ layers, selectedTrack }: { layers: PoseLayer[]; selectedTrack: string }) {
  return (
    <ul className="track-legend" aria-label="可见手轨迹">
      {layers.map((layer) => (
        <li key={layer.id}>
          <i style={{ backgroundColor: trackColor(layer.id) }} />
          <span aria-current={selectedTrack === layer.id ? "true" : "false"}>{layer.id}</span>
        </li>
      ))}
    </ul>
  );
}

function OverlayPanel({
  runKey,
  background,
  backgroundLabel,
  layers,
  selectedTrack,
  width,
  height,
  graphicLabel,
  layerLabel,
}: {
  runKey: string;
  background: ArtifactRef | undefined;
  backgroundLabel: string;
  layers: PoseLayer[];
  selectedTrack: string;
  width: number;
  height: number;
  graphicLabel: string;
  layerLabel?: string;
}) {
  return (
    <div className="comparison-overlay-stage">
      <EvidenceImage runKey={runKey} artifact={background} label={backgroundLabel} />
      <SkeletonGraphic
        layers={layers}
        selectedTrack={selectedTrack}
        width={width}
        height={height}
        label={graphicLabel}
        layerLabel={layerLabel}
      />
      {!layers.some(hasVisiblePoint) && <div className="overlay-empty-label">NOT_PRODUCED</div>}
    </div>
  );
}

function DetectionGraphic({
  detections,
  width,
  height,
  label,
  variant = "legacy",
}: {
  detections: DetectionLayer[];
  width: number;
  height: number;
  label: string;
  variant?: "legacy" | "raw" | "pool";
}) {
  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label={label}>
      {detections.map((detection) => {
        const [x1, y1, x2, y2] = detection.bbox;
        const color = detection.classification === "SEED"
          ? "#74f6c4"
          : detection.classification === "RECOVERY"
            ? "#f4bd6a"
            : detection.classification === "REJECTED"
              ? "#ff6d78"
              : trackColor(detection.id);
        const accessibleLabel = detection.classification && variant !== "legacy"
          ? `${detection.id} ${detection.classification} ${variant === "raw" ? "raw proposal" : "pool candidate"}`
          : `${detection.id} detection`;
        return (
          <g
            key={detection.id}
            aria-label={accessibleLabel}
            className={detection.classification
              ? `candidate-box ${detection.classification.toLowerCase()}`
              : undefined}
            data-classification={detection.classification}
            stroke={color}
          >
            <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="none" vectorEffect="non-scaling-stroke" />
            <text x={x1 + 4} y={Math.max(12, y1 - 5)} fill={color} stroke="none">
              {detection.id}{detection.score === null ? "" : ` ${(detection.score * 100).toFixed(0)}%`}
              {detection.classification ? ` · ${detection.classification}` : ""}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function CandidateDecisionList({ decisions, label }: { decisions: DetectionLayer[]; label: string }) {
  return (
    <ul className="candidate-decision-list" aria-label={`${label} detector candidate decisions`}>
      {decisions.map((candidate) => (
        <li
          aria-label={`${candidate.id} candidate decision`}
          className={candidate.classification?.toLowerCase()}
          data-classification={candidate.classification}
          data-in-pool={String(candidate.inPool === true)}
          key={candidate.id}
        >
          <div className="candidate-decision-identity">
            <strong>{candidate.id}</strong>
            <span>{candidate.classification ?? "UNKNOWN"}</span>
          </div>
          <div className="candidate-decision-facts">
            <span>{candidate.score === null ? "SCORE —" : `SCORE ${(candidate.score * 100).toFixed(1)}%`}</span>
            <span>{candidate.sourceIndex === undefined ? "SOURCE —" : `SOURCE #${candidate.sourceIndex}`}</span>
            <span>{candidate.eligible === undefined ? "ELIGIBLE —" : `ELIGIBLE ${candidate.eligible ? "YES" : "NO"}`}</span>
            <span>{candidate.inPool ? "POOL IN" : "POOL OUT"}</span>
          </div>
          <p>{candidate.reason ?? "REASON_NOT_RECORDED"}</p>
        </li>
      ))}
    </ul>
  );
}

function DetectionComparison({ runKey, records }: { runKey: string; records: TraceRecord[] }) {
  const audits = new Map(SIDES.map((side) => [side, candidateAuditForSide(records, side)]));
  const candidateAware = [...audits.values()].some(Boolean);
  return (
    <div className="native-comparison">
      <header className="coordinate-space-banner">
        {candidateAware
          ? "RAW DETECTOR PROPOSALS → BOUNDED ASSOCIATION POOL"
          : "SOURCE → DETECTION · NATIVE FISHEYE PIXELS"}
      </header>
      <div className="stage-comparison-stereo">
        {SIDES.map((side) => {
          const label = sideLabel(side);
          const source = artifactByRole(records, `source_${side}`);
          const detections = detectionsForSide(records, side);
          const [width, height] = imageSize(records, side);
          const audit = audits.get(side);
          if (audit) {
            return (
              <section className="stage-comparison-view candidate-audit-view" key={side} aria-label={`${label} HAND_DETECTION`}>
                <header><strong>{label}</strong><span>{audit.decisions.length} RAW → {audit.pool.length} POOL</span></header>
                <div className="stage-comparison-pair">
                  <figure>
                    <div className="comparison-overlay-stage">
                      <EvidenceImage runKey={runKey} artifact={source} label={`${label} raw proposal background`} />
                      <DetectionGraphic
                        detections={audit.decisions}
                        width={width}
                        height={height}
                        label={`${label} raw detector proposals`}
                        variant="raw"
                      />
                    </div>
                    <figcaption>BEFORE · {audit.decisions.length} RAW PROPOSALS</figcaption>
                  </figure>
                  <figure>
                    <div className="comparison-overlay-stage">
                      <EvidenceImage runKey={runKey} artifact={source} label={`${label} bounded pool background`} />
                      <DetectionGraphic
                        detections={audit.pool}
                        width={width}
                        height={height}
                        label={`${label} bounded association pool`}
                        variant="pool"
                      />
                    </div>
                    <figcaption>AFTER · {audit.pool.length} CANDIDATES FOR ASSOCIATION</figcaption>
                  </figure>
                </div>
                <CandidateDecisionList decisions={audit.decisions} label={label} />
              </section>
            );
          }
          return (
            <section className="stage-comparison-view" key={side} aria-label={`${label} HAND_DETECTION`}>
              <header><strong>{label}</strong><span>{detections.length} HAND CANDIDATES</span></header>
              <div className="stage-comparison-pair">
                <figure>
                  <EvidenceImage runKey={runKey} artifact={source} label={`${label} detection input`} />
                  <figcaption>BEFORE · source_{side}</figcaption>
                </figure>
                <figure>
                  <div className="comparison-overlay-stage">
                    <EvidenceImage runKey={runKey} artifact={source} label={`${label} detection output background`} />
                    <DetectionGraphic detections={detections} width={width} height={height} label={`${label} all detection boxes`} />
                  </div>
                  <figcaption>AFTER · ALL DETECTIONS</figcaption>
                </figure>
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}

const VIRTUAL_CROP_EVENTS = new Set([
  "virtual_crop_pose_inferred",
  "virtual_crop_pose_not_produced",
]);

interface VirtualCropCandidate {
  key: string;
  record: TraceRecord;
  candidateId: string;
  side: Side;
  cropArtifact: ArtifactRef | undefined;
  maskArtifact: ArtifactRef | undefined;
  cropPoints: NullablePoint2[];
  nativePoints: NullablePoint2[];
  cropSize: [number, number];
  nativeSize: [number, number];
  policyId: string;
  cameraId: string;
  validFraction: number | null;
  fov: [number, number] | null;
}

function objectValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function positiveSize(value: unknown, fallback: [number, number]): [number, number] {
  if (
    Array.isArray(value)
    && value.length >= 2
    && finiteNumber(value[0])
    && finiteNumber(value[1])
    && value[0] > 0
    && value[1] > 0
  ) return [value[0], value[1]];
  return fallback;
}

function virtualFov(camera: Record<string, unknown> | null, size: [number, number]): [number, number] | null {
  const intrinsics = camera?.K_virtual;
  if (!Array.isArray(intrinsics) || !Array.isArray(intrinsics[0]) || !Array.isArray(intrinsics[1])) {
    return null;
  }
  const fx = intrinsics[0][0];
  const fy = intrinsics[1][1];
  if (!finiteNumber(fx) || !finiteNumber(fy) || fx <= 0 || fy <= 0) return null;
  const toDegrees = 180 / Math.PI;
  return [
    2 * Math.atan(size[0] / (2 * fx)) * toDegrees,
    2 * Math.atan(size[1] / (2 * fy)) * toDegrees,
  ];
}

function virtualCropCandidates(records: TraceRecord[]): VirtualCropCandidate[] {
  return records.flatMap((record, index) => {
    if (!VIRTUAL_CROP_EVENTS.has(record.event ?? "")) return [];
    const payload = payloadOf(record);
    if (payload.view_id !== "left" && payload.view_id !== "right") return [];
    const side = payload.view_id as Side;
    const candidateId = typeof payload.candidate_id === "string" && payload.candidate_id
      ? payload.candidate_id
      : `${side}-candidate-${index}`;
    const camera = objectValue(payload.virtual_camera);
    const cropSize = positiveSize(camera?.output_size, [256, 256]);
    const artifacts = artifactsOf(record);
    return [{
      key: record.record_id ?? `${side}:${candidateId}:${index}`,
      record,
      candidateId,
      side,
      cropArtifact: artifacts.find((artifact) => artifact.role === "virtual_crop"),
      maskArtifact: artifacts.find((artifact) => artifact.role === "virtual_crop_valid_mask"),
      cropPoints: pointList(payload.keypoints_uv_crop),
      nativePoints: pointList(payload.keypoints_uv_native ?? payload.keypoints_uv),
      cropSize,
      nativeSize: imageSize([record], side),
      policyId: String(camera?.crop_policy_id ?? payload.crop_policy_id ?? "UNKNOWN_POLICY"),
      cameraId: String(camera?.virtual_camera_id ?? payload.virtual_camera_id ?? "UNKNOWN_CAMERA"),
      validFraction: finiteNumber(camera?.valid_fraction) ? camera.valid_fraction : null,
      fov: virtualFov(camera, cropSize),
    }];
  }).sort((left, right) => (
    left.side.localeCompare(right.side) || left.candidateId.localeCompare(right.candidateId)
  ));
}

function VirtualCropDiagnostics({ runKey, records }: { runKey: string; records: TraceRecord[] }) {
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  const candidates = virtualCropCandidates(records);
  if (!candidates.length) return null;
  const sourceBackgrounds: Record<Side, ArtifactRef | undefined> = {
    left: artifactByRole(records, "source_left"),
    right: artifactByRole(records, "source_right"),
  };
  return (
    <section className="virtual-crop-diagnostics" aria-label="Virtual crop RTMPose diagnostics">
      <header className="virtual-crop-heading">
        <div>
          <span>V2 · PHYSICAL POSE INPUT</span>
          <strong>VIRTUAL PERSPECTIVE CROP DIAGNOSTICS</strong>
        </div>
        <span>{candidates.length} CANDIDATES · CROP → NATIVE</span>
      </header>
      <div className="virtual-crop-candidate-list">
        {candidates.map((candidate) => {
          const label = sideLabel(candidate.side);
          const produced = outputProduced(candidate.record);
          const expanded = expandedKey === candidate.key;
          const cropLayer: PoseLayer = { id: candidate.candidateId, points: candidate.cropPoints };
          const nativeLayer: PoseLayer = { id: candidate.candidateId, points: candidate.nativePoints };
          return (
            <article
              className={`virtual-crop-candidate ${produced ? "produced" : "not-produced"}`}
              aria-label={`${candidate.candidateId} virtual crop diagnostic`}
              key={candidate.key}
            >
              <header>
                <div className="virtual-crop-candidate-identity">
                  <strong>{candidate.candidateId}</strong>
                  <span>{label.toUpperCase()} · VIRTUAL PINHOLE</span>
                </div>
                <div className="virtual-crop-candidate-controls">
                  <span>{produced ? "PRODUCED" : "NOT_PRODUCED"}</span>
                  <button
                    type="button"
                    aria-expanded={expanded}
                    aria-label={`${expanded ? "收起" : "展开"} ${candidate.candidateId} 图像证据`}
                    onClick={() => setExpandedKey(expanded ? null : candidate.key)}
                  >
                    {expanded ? "HIDE EVIDENCE" : "LOAD EVIDENCE"}
                  </button>
                </div>
              </header>
              <dl className="virtual-camera-metrics">
                <div><dt>POLICY</dt><dd>{candidate.policyId}</dd></div>
                <div>
                  <dt>FOV</dt>
                  <dd>{candidate.fov
                    ? `FOV ${candidate.fov[0].toFixed(1)}° × ${candidate.fov[1].toFixed(1)}°`
                    : "FOV —"}</dd>
                </div>
                <div>
                  <dt>VALID RAYS</dt>
                  <dd>{candidate.validFraction === null
                    ? "VALID —"
                    : `VALID ${(candidate.validFraction * 100).toFixed(1)}%`}</dd>
                </div>
                <div><dt>CAMERA</dt><dd title={candidate.cameraId}>{candidate.cameraId}</dd></div>
              </dl>
              {expanded && (
                <div className={`virtual-crop-space-grid ${candidate.maskArtifact ? "with-mask" : ""}`}>
                  <figure>
                    <OverlayPanel
                      runKey={runKey}
                      background={candidate.cropArtifact}
                      backgroundLabel={`${candidate.candidateId} virtual crop`}
                      layers={[cropLayer]}
                      selectedTrack=""
                      width={candidate.cropSize[0]}
                      height={candidate.cropSize[1]}
                      graphicLabel={`${candidate.candidateId} crop-space keypoints`}
                      layerLabel="crop-space keypoints"
                    />
                    <figcaption>BEFORE · RTMPOSE CROP PIXELS</figcaption>
                  </figure>
                  <figure>
                    <OverlayPanel
                      runKey={runKey}
                      background={sourceBackgrounds[candidate.side]}
                      backgroundLabel={`${candidate.candidateId} native fisheye source`}
                      layers={[nativeLayer]}
                      selectedTrack=""
                      width={candidate.nativeSize[0]}
                      height={candidate.nativeSize[1]}
                      graphicLabel={`${candidate.candidateId} native-space keypoints`}
                      layerLabel="native-space keypoints"
                    />
                    <figcaption>AFTER · NATIVE FISHEYE PIXELS</figcaption>
                  </figure>
                  {candidate.maskArtifact && (
                    <figure className="virtual-crop-mask">
                      <EvidenceImage
                        runKey={runKey}
                        artifact={candidate.maskArtifact}
                        label={`${candidate.candidateId} valid mask`}
                      />
                      <figcaption>VALID_MASK · PHYSICAL RAYS</figcaption>
                    </figure>
                  )}
                </div>
              )}
              {!produced && (
                <p className="virtual-crop-failure">
                  <span>REASON</span>{failureReason(candidate.record)}
                </p>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}

function PoseComparison({
  runKey,
  records,
  selectedTrack,
}: {
  runKey: string;
  records: TraceRecord[];
  selectedTrack: string;
}) {
  return (
    <>
      <VirtualCropDiagnostics runKey={runKey} records={records} />
      <div className="native-comparison">
        <header className="coordinate-space-banner">DETECTION → ALL RTMPOSE HANDS · NATIVE FISHEYE PIXELS</header>
        <div className="stage-comparison-stereo">
          {SIDES.map((side) => {
            const label = sideLabel(side);
            const source = artifactByRole(records, `source_${side}`);
            const detections = detectionsForSide(records, side);
            const layers = poseLayers(records, side, "native");
            const [width, height] = imageSize(records, side);
            return (
              <section className="stage-comparison-view pose-stage-comparison" key={side} aria-label={`${label} HAND_POSE_2D`}>
                <header><strong>{label}</strong><span>DETECTION → ALL HAND KEYPOINTS</span></header>
                <div className="stage-comparison-pair">
                  <figure>
                    <div className="comparison-overlay-stage">
                      <EvidenceImage runKey={runKey} artifact={source} label={`${label} detection background`} />
                      <DetectionGraphic detections={detections} width={width} height={height} label={`${label} detection input boxes`} />
                    </div>
                    <figcaption>BEFORE · DETECTION</figcaption>
                  </figure>
                  <figure>
                    <OverlayPanel
                      runKey={runKey}
                      background={source}
                      backgroundLabel={`${label} RTMPose background`}
                      layers={layers}
                      selectedTrack={selectedTrack}
                      width={width}
                      height={height}
                      graphicLabel={`${label} RTMPose 全手叠加`}
                      layerLabel="2D 骨架"
                    />
                    <figcaption>AFTER · ALL RTMPOSE HANDS</figcaption>
                  </figure>
                </div>
                <TrackLegend layers={layers} selectedTrack={selectedTrack} />
              </section>
            );
          })}
        </div>
      </div>
    </>
  );
}

function associationMatches(records: TraceRecord[]): AssociationMatch[] {
  const association = records.find((record) => (
    record.stage === "CROSS_VIEW_ASSOCIATION" && Array.isArray(payloadOf(record).matches)
  ));
  if (!association) return [];
  const assignments = records
    .filter((record) => record.stage === "CROSS_VIEW_ASSOCIATION")
    .flatMap((record) => objectValues(payloadOf(record).assignments));
  return objectValues(payloadOf(association).matches).flatMap((match, index) => {
    const matchId = String(match.match_id ?? `match-${index}`);
    const assignment = assignments.find((candidate) => (
      candidate.match_id === matchId
      || (typeof candidate.observation_id === "string" && candidate.observation_id.endsWith(`:${matchId}`))
    ));
    const leftCandidateId = match.left_candidate_id;
    const rightCandidateId = match.right_candidate_id;
    if (typeof leftCandidateId !== "string" || typeof rightCandidateId !== "string") return [];
    return [{
      matchId,
      leftCandidateId,
      rightCandidateId,
      trackId: typeof assignment?.track_id === "string"
        ? assignment.track_id
        : typeof match.track_id === "string" ? match.track_id : null,
    }];
  });
}

function AssociationComparison({
  runKey,
  records,
  selectedTrack,
}: {
  runKey: string;
  records: TraceRecord[];
  selectedTrack: string;
}) {
  const matches = associationMatches(records);
  const [width, height] = rectifiedImageSize(records);
  return (
    <div className="projection-comparison">
      <header className="coordinate-space-banner">CANDIDATES → MATCHED / TRACKED</header>
      <div className="association-summary" aria-label="跨视角匹配结果">
        {matches.length ? matches.map((match) => (
          <span key={match.matchId}>
            {match.trackId
              ? `${match.matchId} → ${match.trackId}`
              : `${match.matchId} · MATCHED · UNTRACKED`}
          </span>
        )) : <span>NOT_PRODUCED · NO_CROSS_VIEW_MATCH</span>}
      </div>
      <div className="stage-comparison-stereo">
        {SIDES.map((side) => {
          const label = sideLabel(side);
          const candidates = poseLayers(records, side, "rectified").map((layer) => ({
            ...layer,
            id: layer.candidateId ?? layer.id,
          }));
          const associationLayers = candidates.map((candidate) => {
            const match = matches.find((value) => (
              (side === "left" ? value.leftCandidateId : value.rightCandidateId)
                === (candidate.candidateId ?? candidate.id)
            ));
            if (!match) return { ...candidate, id: `${candidate.id} · UNMATCHED` };
            return {
              ...candidate,
              id: match.trackId ?? `${match.matchId} · UNTRACKED`,
            };
          });
          return (
            <section className="stage-comparison-view" key={side} aria-label={`${label} CROSS_VIEW_ASSOCIATION`}>
              <header><strong>{label}</strong><span>RECTIFIED PIXEL SPACE</span></header>
              <div className="stage-comparison-pair">
                <figure>
                  <OverlayPanel
                    runKey={runKey}
                    background={artifactByRole(records, `rectified_${side}`)}
                    backgroundLabel={`${label} association candidate background`}
                    layers={candidates}
                    selectedTrack=""
                    width={width}
                    height={height}
                    graphicLabel={`${label} candidate keypoints`}
                  />
                  <figcaption>BEFORE · ALL RECTIFIED CANDIDATES</figcaption>
                </figure>
                <figure>
                  <OverlayPanel
                    runKey={runKey}
                    background={artifactByRole(records, `rectified_${side}`)}
                    backgroundLabel={`${label} association matched background`}
                    layers={associationLayers}
                    selectedTrack={selectedTrack}
                    width={width}
                    height={height}
                    graphicLabel={`${label} association`}
                  />
                  <figcaption>AFTER · MATCH / TRACK ID</figcaption>
                </figure>
              </div>
              <TrackLegend layers={associationLayers} selectedTrack={selectedTrack} />
            </section>
          );
        })}
      </div>
    </div>
  );
}

function ProjectionTransition({
  runKey,
  records,
  selectedTrack,
  heading,
  beforeLabel,
  afterLabel,
  beforeLayers,
  afterLayers,
}: {
  runKey: string;
  records: TraceRecord[];
  selectedTrack: string;
  heading: string;
  beforeLabel: string;
  afterLabel: string;
  beforeLayers: (side: Side) => PoseLayer[];
  afterLayers: (side: Side) => PoseLayer[];
}) {
  const [width, height] = rectifiedImageSize(records);
  return (
    <div className="projection-comparison">
      <header className="coordinate-space-banner">{heading}</header>
      <div className="coordinate-space-tag">RECTIFIED PIXEL SPACE</div>
      <div className="stage-comparison-stereo">
        {SIDES.map((side) => {
          const label = sideLabel(side);
          const before = beforeLayers(side);
          const after = afterLayers(side);
          return (
            <section className="stage-comparison-view" key={side} aria-label={`${label} ${afterLabel}`}>
              <header><strong>{label}</strong><span>BEFORE → AFTER</span></header>
              <div className="stage-comparison-pair">
                <figure>
                  <OverlayPanel
                    runKey={runKey}
                    background={artifactByRole(records, `rectified_${side}`)}
                    backgroundLabel={`${label} ${beforeLabel} rectified background`}
                    layers={before}
                    selectedTrack={selectedTrack}
                    width={width}
                    height={height}
                    graphicLabel={`${label} ${beforeLabel}`}
                  />
                  <figcaption>BEFORE · {beforeLabel}</figcaption>
                </figure>
                <figure>
                  <OverlayPanel
                    runKey={runKey}
                    background={artifactByRole(records, `rectified_${side}`)}
                    backgroundLabel={`${label} ${afterLabel} rectified background`}
                    layers={after}
                    selectedTrack={selectedTrack}
                    width={width}
                    height={height}
                    graphicLabel={`${label} ${afterLabel} rectified projection`}
                  />
                  <figcaption>AFTER · {afterLabel}</figcaption>
                </figure>
              </div>
              <TrackLegend layers={after.length ? after : before} selectedTrack={selectedTrack} />
            </section>
          );
        })}
      </div>
    </div>
  );
}

function NotProducedReasons({ records, stage }: { records: TraceRecord[]; stage: string }) {
  const failed = records.filter((record) => record.stage === stage && !outputProduced(record));
  if (!failed.length) return null;
  return (
    <ul className="stage-failure-list" aria-label={`${stage} 未产出原因`}>
      {failed.map((record, index) => {
        const payload = payloadOf(record);
        const track = typeof payload.track_id === "string" ? payload.track_id : "NO_TRACK";
        return <li key={record.record_id ?? `${stage}-${index}`}>{track} · NOT_PRODUCED · {failureReason(record)}</li>;
      })}
    </ul>
  );
}

function RawDownstreamRejectionReasons({ records }: { records: TraceRecord[] }) {
  const downstreamStages = new Set(["KINEMATIC_REFINEMENT", "TEMPORAL_REFINEMENT", "EXPORT"]);
  const reasons = [...new Set(records.flatMap((record) => {
    if (!downstreamStages.has(record.stage ?? "") || outputProduced(record)) return [];
    const payload = payloadOf(record);
    if (typeof payload.track_id === "string" && payload.track_id) return [];
    const reason = failureReason(record);
    return reason === "OUTPUT_NOT_PRODUCED" || reason === "STAGE_FAILED" ? [] : [reason];
  }))];
  if (!reasons.length) return null;
  return (
    <ul className="stage-failure-list" aria-label="Raw downstream rejection reasons">
      {reasons.map((reason) => (
        <li key={reason}>DOWNSTREAM · NOT_PRODUCED · {reason}</li>
      ))}
    </ul>
  );
}

function RawComparison({ runKey, records, selectedTrack }: Omit<StageComparisonProps, "selectedNodeId">) {
  const matches = associationMatches(records);
  return (
    <>
      <ProjectionTransition
        runKey={runKey}
        records={records}
        selectedTrack={selectedTrack}
        heading="ASSOCIATED 2D → STEREO TRIANGULATION · RECTIFIED PIXEL SPACE"
        beforeLabel="ASSOCIATION"
        afterLabel="Raw 3D"
        beforeLayers={(side) => poseLayers(records, side, "rectified").flatMap((layer) => {
          const match = matches.find((value) => (
            (side === "left" ? value.leftCandidateId : value.rightCandidateId)
              === (layer.candidateId ?? layer.id)
          ));
          return match?.trackId ? [{ ...layer, id: match.trackId }] : [];
        })}
        afterLayers={(side) => projectedLayers(records, "RAW_FUSION", side)}
      />
      <NotProducedReasons records={records} stage="RAW_FUSION" />
      <RawDownstreamRejectionReasons records={records} />
    </>
  );
}

function recordObject(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function manoGateOf(record: TraceRecord): Record<string, unknown> | null {
  const payload = payloadOf(record);
  const fitQuality = recordObject(payload.fit_quality);
  const selection = recordObject(payload.selection);
  const selectionGate = recordObject(selection?.gate);
  if (!fitQuality && !selectionGate) return null;
  const gate = { ...(selectionGate ?? {}), ...(fitQuality ?? {}) };
  const diagnosticFields = [
    "method",
    "first_pass_rmse_m",
    "full_rmse_m",
    "inlier_rmse_m",
    "joint_weights",
    "inlier_mask",
  ];
  return diagnosticFields.some((field) => Object.hasOwn(gate, field)) ? gate : null;
}

function metricMillimetres(value: unknown): string {
  return finiteNumber(value) ? `${(value * 1000).toFixed(2)} mm` : "—";
}

function integerDisplay(value: unknown): string {
  return finiteNumber(value) ? String(Math.trunc(value)) : "—";
}

function ManoGateDiagnostics({ records, selectedTrack }: { records: TraceRecord[]; selectedTrack: string }) {
  const diagnostics = records.flatMap((record) => {
    if (record.stage !== "KINEMATIC_REFINEMENT") return [];
    const gate = manoGateOf(record);
    return gate ? [{ record, gate }] : [];
  });
  if (!diagnostics.length) return null;

  return (
    <section className="mano-gate-diagnostics" aria-label="MANO robust gate diagnostics">
      <header className="mano-gate-heading">
        <div>
          <span>FRAME-WISE FIT AUDIT</span>
          <strong>MANO ROBUST GATE · METRIC RESIDUALS</strong>
        </div>
        <span>HEURISTIC DIAGNOSTIC · NOT CALIBRATED</span>
      </header>
      <div className="mano-gate-list">
        {diagnostics.map(({ record, gate }, index) => {
          const payload = payloadOf(record);
          const trackId = typeof payload.track_id === "string" && payload.track_id
            ? payload.track_id
            : "NO_TRACK";
          const stages = objectValues(gate.stage_iterations);
          const weightedRefit = gate.triggered === true
            || stages.some((stage) => stage.stage === "WEIGHTED_REFIT");
          const weights = Array.isArray(gate.joint_weights) ? gate.joint_weights : [];
          const mask = Array.isArray(gate.inlier_mask) ? gate.inlier_mask : [];
          const trimmedIndices = Array.isArray(gate.trimmed_joint_indices)
            ? [...new Set(gate.trimmed_joint_indices.filter(
              (value): value is number => Number.isInteger(value) && value >= 0 && value < FHP21_NAMES.length,
            ))].sort((left, right) => left - right)
            : [];
          const accepted = gate.accepted === true;
          const produced = outputProduced(record);
          return (
            <article
              key={record.record_id ?? `mano-gate-${index}`}
              className={`mano-gate-card ${produced ? "produced" : "not-produced"} ${selectedTrack === trackId ? "selected" : ""}`}
              aria-label={`${trackId} MANO robust gate diagnostic`}
              aria-current={selectedTrack === trackId}
            >
              <header>
                <div>
                  <strong>{trackId}</strong>
                  <span>{produced ? "PRODUCED" : "NOT_PRODUCED"}</span>
                </div>
                <div className="mano-gate-identity">
                  <span>{typeof gate.method === "string" ? gate.method : "METHOD_UNRECORDED"}</span>
                  <span>{typeof gate.status === "string" ? gate.status : "STATUS_UNRECORDED"}</span>
                  <strong>{accepted ? "GATE ACCEPTED" : "GATE REJECTED"}</strong>
                </div>
              </header>
              <p className="mano-gate-reason">
                <span>GATE REASON</span>
                {typeof gate.reason === "string" && gate.reason ? gate.reason : failureReason(record)}
              </p>
              <dl className="mano-gate-metrics">
                <div><dt>首遍 RMSE</dt><dd>{metricMillimetres(gate.first_pass_rmse_m)}</dd></div>
                <div><dt>普通 / FULL RMSE</dt><dd>{metricMillimetres(gate.full_rmse_m)}</dd></div>
                <div><dt>INLIER RMSE</dt><dd>{metricMillimetres(gate.inlier_rmse_m)}</dd></div>
                <div><dt>WEIGHTED RMSE</dt><dd>{metricMillimetres(gate.weighted_rmse_m)}</dd></div>
                <div><dt>INLIER GATE ≤</dt><dd>{metricMillimetres(gate.rmse_gate_m ?? 0.02)}</dd></div>
                <div><dt>FULL CEILING ≤</dt><dd>{metricMillimetres(gate.full_rmse_ceiling_m ?? 0.04)}</dd></div>
                <div>
                  <dt>有效支持 / 最少</dt>
                  <dd>{integerDisplay(gate.effective_joint_count)} / {integerDisplay(gate.minimum_effective_joint_count)}</dd>
                </div>
                <div>
                  <dt>WEIGHTED REFIT</dt>
                  <dd>{weightedRefit ? "YES" : "NO"} · {stages.length} STAGES</dd>
                </div>
              </dl>
              {stages.length > 0 && (
                <div className="mano-gate-iterations" aria-label={`${trackId} MANO stage iterations`}>
                  {stages.map((stage, stageIndex) => (
                    <span key={`${String(stage.stage ?? "STAGE")}-${stageIndex}`}>
                      {String(stage.stage ?? "STAGE_UNRECORDED")} · {integerDisplay(stage.iterations_run)}
                    </span>
                  ))}
                </div>
              )}
              <div className="mano-gate-trimmed">
                <span>TRIMMED FHP21</span>
                {trimmedIndices.length
                  ? trimmedIndices.map((jointIndex) => (
                    <strong key={jointIndex}>{jointIndex} · {FHP21_NAMES[jointIndex]}</strong>
                  ))
                  : <em>NONE</em>}
              </div>
              {(weights.length > 0 || mask.length > 0) && (
                <ul className="mano-joint-weights" aria-label={`${trackId} FHP21 joint weights`}>
                  {FHP21_NAMES.map((name, jointIndex) => {
                    const weight = finiteNumber(weights[jointIndex]) ? weights[jointIndex] : null;
                    const inlier = typeof mask[jointIndex] === "boolean"
                      ? mask[jointIndex]
                      : weight === null || weight > 0;
                    const explicitlyTrimmed = trimmedIndices.includes(jointIndex);
                    const jointState = explicitlyTrimmed
                      ? "trimmed"
                      : inlier ? "inlier" : "unsupported";
                    const stateLabel = explicitlyTrimmed
                      ? "TRIMMED"
                      : inlier ? "INLIER" : "NO RAW SUPPORT";
                    return (
                      <li
                        key={name}
                        className={jointState}
                        title={`${jointIndex} · ${name} · weight ${weight === null ? "—" : weight.toFixed(2)} · ${stateLabel}`}
                      >
                        <span>{jointIndex}</span>
                        <i style={{ opacity: weight === null ? 0.35 : Math.max(0.18, Math.min(1, weight)) }} />
                      </li>
                    );
                  })}
                </ul>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}

function ManoComparison({ runKey, records, selectedTrack }: Omit<StageComparisonProps, "selectedNodeId">) {
  return (
    <>
      <ProjectionTransition
        runKey={runKey}
        records={records}
        selectedTrack={selectedTrack}
        heading="RAW_FUSION → MANO v1.2"
        beforeLabel="RAW_FUSION"
        afterLabel="KINEMATIC_REFINEMENT"
        beforeLayers={(side) => projectedLayers(records, "RAW_FUSION", side)}
        afterLayers={(side) => projectedLayers(records, "KINEMATIC_REFINEMENT", side)}
      />
      <ManoGateDiagnostics records={records} selectedTrack={selectedTrack} />
      <NotProducedReasons records={records} stage="KINEMATIC_REFINEMENT" />
    </>
  );
}

function temporalInputLayers(records: TraceRecord[], side: Side): PoseLayer[] {
  const layers: PoseLayer[] = [];
  for (const record of records) {
    if (record.stage !== "TEMPORAL_REFINEMENT") continue;
    const payload = payloadOf(record);
    if (typeof payload.track_id !== "string") continue;
    const inputStage = payload.input_stage === "KINEMATIC_REFINEMENT"
      ? "KINEMATIC_REFINEMENT"
      : payload.input_stage === "RAW_FUSION" ? "RAW_FUSION" : "";
    if (!inputStage) continue;
    const layer = projectedLayers(records, inputStage, side).find((value) => value.id === payload.track_id);
    if (layer) layers.push(layer);
  }
  return [...new Map(layers.map((layer) => [layer.id, layer])).values()]
    .sort((left, right) => left.id.localeCompare(right.id));
}

function TemporalComparison({ runKey, records, selectedTrack }: Omit<StageComparisonProps, "selectedNodeId">) {
  const temporalRecords = records.filter((record) => record.stage === "TEMPORAL_REFINEMENT");
  const trackedTemporalRecords = temporalRecords.filter((record) => {
    const trackId = payloadOf(record).track_id;
    return typeof trackId === "string" && Boolean(trackId);
  });
  return (
    <>
      <div className="stage-path-list" aria-label="每手实际时序输入">
        {trackedTemporalRecords.map((record, index) => {
          const payload = payloadOf(record);
          const track = payload.track_id as string;
          const input = payload.input_stage === "KINEMATIC_REFINEMENT"
            ? "KINEMATIC_REFINEMENT"
            : payload.input_stage === "RAW_FUSION" ? "RAW_FUSION" : "UNKNOWN_INPUT";
          const provenance = input === "RAW_FUSION"
            ? "RAW → EMA"
            : input === "KINEMATIC_REFINEMENT" ? "MANO → EMA" : "来源未记录";
          return (
            <span key={record.record_id ?? `temporal-path-${index}`}>
              {track} · {input} → TEMPORAL_REFINEMENT · {provenance}
            </span>
          );
        })}
      </div>
      <ProjectionTransition
        runKey={runKey}
        records={records}
        selectedTrack={selectedTrack}
        heading="ACTUAL INPUT_STAGE → TEMPORAL · RECTIFIED PIXEL SPACE"
        beforeLabel="ACTUAL INPUT_STAGE"
        afterLabel="TEMPORAL_REFINEMENT"
        beforeLayers={(side) => temporalInputLayers(records, side)}
        afterLayers={(side) => projectedLayers(records, "TEMPORAL_REFINEMENT", side)}
      />
      <NotProducedReasons records={records} stage="TEMPORAL_REFINEMENT" />
    </>
  );
}

function ExportComparison({ runKey, records, selectedTrack }: Omit<StageComparisonProps, "selectedNodeId">) {
  return (
    <>
      <ProjectionTransition
        runKey={runKey}
        records={records}
        selectedTrack={selectedTrack}
        heading="TEMPORAL_REFINEMENT → STABLE FHP21 EXPORT"
        beforeLabel="TEMPORAL_REFINEMENT"
        afterLabel="EXPORT"
        beforeLayers={(side) => projectedLayers(records, "TEMPORAL_REFINEMENT", side)}
        afterLayers={(side) => projectedLayers(records, "EXPORT", side)}
      />
      <NotProducedReasons records={records} stage="EXPORT" />
    </>
  );
}

export function StageComparison({
  runKey,
  records,
  selectedNodeId,
  selectedTrack,
}: StageComparisonProps) {
  let content;
  switch (selectedNodeId) {
    case "SOURCE_RGB":
      content = <SourceComparison runKey={runKey} records={records} />;
      break;
    case "FISHEYE_UNDISTORTION":
      content = (
        <StereoImageComparison runKey={runKey} records={records} beforePrefix="source" afterPrefix="undistorted" />
      );
      break;
    case "STEREO_RECTIFICATION":
      content = (
        <StereoImageComparison runKey={runKey} records={records} beforePrefix="undistorted" afterPrefix="rectified" />
      );
      break;
    case "HAND_DETECTION":
      content = <DetectionComparison runKey={runKey} records={records} />;
      break;
    case "HAND_POSE_2D":
      content = <PoseComparison runKey={runKey} records={records} selectedTrack={selectedTrack} />;
      break;
    case "CROSS_VIEW_ASSOCIATION":
      content = <AssociationComparison runKey={runKey} records={records} selectedTrack={selectedTrack} />;
      break;
    case "STEREO_TRIANGULATION_RAW_3D":
      content = <RawComparison runKey={runKey} records={records} selectedTrack={selectedTrack} />;
      break;
    case "MANO_FRAMEWISE":
      content = <ManoComparison runKey={runKey} records={records} selectedTrack={selectedTrack} />;
      break;
    case "TEMPORAL_REFINEMENT":
      content = <TemporalComparison runKey={runKey} records={records} selectedTrack={selectedTrack} />;
      break;
    case "STABLE_FHP21_EXPORT":
      content = <ExportComparison runKey={runKey} records={records} selectedTrack={selectedTrack} />;
      break;
  }
  return <section className="stage-comparison" aria-label="阶段前后对比">{content}</section>;
}
