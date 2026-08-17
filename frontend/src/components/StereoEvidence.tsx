import { Eye, ImageIcon, Layers2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { traceApi } from "../api/client";
import type { ArtifactRef, TraceRecord } from "../api/types";
import {
  artifactsOf,
  FHP21_EDGES,
  FINGER_COLORS,
  payloadOf,
  scoreValues,
  viewImageSize,
} from "../domain/trace";
import { PreviewableImage } from "./PreviewableImage";

interface StereoEvidenceProps {
  runKey: string;
  records: TraceRecord[];
  trackId: string;
}

const DEFAULT_KEYPOINT_SCORE_THRESHOLD = 0.2;
type Side = "left" | "right";
type Point2 = [number, number];
type NullablePoint2 = Point2 | null;

interface PoseLayer {
  id: string;
  candidateId: string;
  ownerRecordId: string;
  points: NullablePoint2[];
  scores: number[];
  scoreThreshold: number;
  boxes: Array<[number, number, number, number]>;
}

interface EvidenceLayer {
  id: string;
  selectionKey: string;
  label: string;
  artifact: ArtifactRef | undefined;
  artifactRole: string;
  coordinateSpace: "native" | "undistorted" | "rectified" | "crop";
  stageLabel: string;
  poses: PoseLayer[];
  width: number;
  height: number;
  noOverlayReason: string;
}

type ProjectedStage = "RAW_FUSION" | "KINEMATIC_REFINEMENT" | "TEMPORAL_REFINEMENT" | "EXPORT";

const PROJECTED_STAGE_LABELS: Record<ProjectedStage, string> = {
  RAW_FUSION: "RAW_FUSION",
  KINEMATIC_REFINEMENT: "MANO",
  TEMPORAL_REFINEMENT: "TEMPORAL",
  EXPORT: "EXPORT",
};

function keypointScoreThreshold(payload: Record<string, unknown>): number {
  const direct = payload.keypoint_score_threshold;
  if (typeof direct === "number" && Number.isFinite(direct) && direct >= 0 && direct <= 1) {
    return direct;
  }
  const thresholds = payload.thresholds;
  const nested = thresholds && typeof thresholds === "object"
    ? (thresholds as Record<string, unknown>).keypoint_score
    : undefined;
  return typeof nested === "number" && Number.isFinite(nested) && nested >= 0 && nested <= 1
    ? nested
    : DEFAULT_KEYPOINT_SCORE_THRESHOLD;
}

function artifactPath(artifact: ArtifactRef): string {
  return String(artifact.relative_path ?? artifact.path ?? "");
}

function isAllowedBackground(artifact: ArtifactRef): boolean {
  const mediaType = String(artifact.media_type ?? "").toLowerCase();
  const path = artifactPath(artifact);
  return !mediaType.startsWith("video/")
    && mediaType !== "application/json"
    && !/\.(mp4|webm|mov|json)$/i.test(path);
}

function detections(payload: Record<string, unknown>): Array<[number, number, number, number]> {
  const candidates = Array.isArray(payload.detections)
    ? payload.detections
    : payload.bbox_xyxy
      ? [{ bbox_xyxy: payload.bbox_xyxy }]
      : [];
  return candidates.flatMap((detection) => {
    if (!detection || typeof detection !== "object") return [];
    const box = (detection as { bbox_xyxy?: unknown }).bbox_xyxy;
    if (!Array.isArray(box) || box.length < 4 || box.some((value) => typeof value !== "number")) return [];
    return [[box[0], box[1], box[2], box[3]] as [number, number, number, number]];
  });
}

interface ArtifactOwner {
  artifact: ArtifactRef;
  recordId: string;
}

function artifactOwnerByRole(records: TraceRecord[], role: string): ArtifactOwner | undefined {
  for (const [index, record] of records.entries()) {
    const artifact = artifactsOf(record).find((candidate) => (
      candidate.role === role && isAllowedBackground(candidate)
    ));
    if (artifact) return { artifact, recordId: record.record_id ?? `${record.stage ?? "record"}-${index}` };
  }
  return undefined;
}

function rectifiedImageSize(records: TraceRecord[]): [number, number] {
  for (const record of records) {
    if (record.stage !== "RECTIFICATION") continue;
    const payload = payloadOf(record);
    const width = payload.image_width ?? payload.output_width;
    const height = payload.image_height ?? payload.output_height;
    if (
      typeof width === "number"
      && Number.isFinite(width)
      && width > 0
      && typeof height === "number"
      && Number.isFinite(height)
      && height > 0
    ) return [width, height];
    const outputSize = payload.output_size;
    if (
      Array.isArray(outputSize)
      && typeof outputSize[0] === "number"
      && Number.isFinite(outputSize[0])
      && outputSize[0] > 0
      && typeof outputSize[1] === "number"
      && Number.isFinite(outputSize[1])
      && outputSize[1] > 0
    ) return [outputSize[0], outputSize[1]];
  }
  return [1920, 1080];
}

function positiveSize(value: unknown, fallback: [number, number]): [number, number] {
  if (
    Array.isArray(value)
    && typeof value[0] === "number"
    && Number.isFinite(value[0])
    && value[0] > 0
    && typeof value[1] === "number"
    && Number.isFinite(value[1])
    && value[1] > 0
  ) return [value[0], value[1]];
  return fallback;
}

function objectValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function nullablePoints2(value: unknown): NullablePoint2[] {
  if (!Array.isArray(value) || value.length !== 21) return [];
  return value.map((point) => {
    if (
      !Array.isArray(point)
      || typeof point[0] !== "number"
      || !Number.isFinite(point[0])
      || typeof point[1] !== "number"
      || !Number.isFinite(point[1])
    ) return null;
    return [point[0], point[1]];
  });
}

function posePoints2(value: unknown): NullablePoint2[] {
  if (!Array.isArray(value) || value.length === 0 || value.length > 21) return [];
  const points = Array.from({ length: 21 }, (_, index): NullablePoint2 => {
    const point = value[index];
    if (
      !Array.isArray(point)
      || typeof point[0] !== "number"
      || !Number.isFinite(point[0])
      || typeof point[1] !== "number"
      || !Number.isFinite(point[1])
    ) return null;
    return [point[0], point[1]];
  });
  return points.some(Boolean) ? points : [];
}

function outputProduced(record: TraceRecord): boolean {
  const outputStatus = payloadOf(record).output_status;
  if (outputStatus === "NOT_PRODUCED") return false;
  if (outputStatus === "PRODUCED") return true;
  return record.status !== "FAILED" && record.status !== "SKIPPED";
}

function objectInstances(payload: Record<string, unknown>): Record<string, unknown>[] {
  if (!Array.isArray(payload.instances)) return [];
  return payload.instances.filter(
    (instance): instance is Record<string, unknown> => Boolean(instance && typeof instance === "object"),
  );
}

function poseLayer(
  payload: Record<string, unknown>,
  side: Side,
  ownerRecordId: string,
  field: "keypoints_uv" | "keypoints_uv_rectified",
  includeBoxes: boolean,
): PoseLayer | null {
  const points = posePoints2(payload[field]);
  if (points.length !== 21) return null;
  return {
    id: String(payload.track_id ?? payload.candidate_id ?? `${side}-pose`),
    candidateId: typeof payload.candidate_id === "string" ? payload.candidate_id : "",
    ownerRecordId,
    points,
    scores: scoreValues(payload.keypoint_scores),
    scoreThreshold: keypointScoreThreshold(payload),
    boxes: includeBoxes ? detections(payload) : [],
  };
}

function selectedPoseLayers(
  records: TraceRecord[],
  side: Side,
  trackId: string,
  field: "keypoints_uv" | "keypoints_uv_rectified",
  includeBoxes: boolean,
): PoseLayer[] {
  const tracked = new Map<string, PoseLayer>();
  const untracked = new Map<string, PoseLayer>();
  for (const record of records) {
    if (record.stage !== "POSE_2D") continue;
    const payload = payloadOf(record);
    if (payload.view_id !== side) continue;
    const recordTrack = typeof payload.track_id === "string" ? payload.track_id : "";
    const ownerRecordId = record.record_id ?? `${record.stage ?? "POSE_2D"}:${record.event ?? "record"}`;
    if (!trackId || recordTrack === trackId) {
      const direct = poseLayer(payload, side, ownerRecordId, field, includeBoxes);
      if (direct) (recordTrack ? tracked : untracked).set(direct.id, direct);
    }
    if (trackId) continue;
    objectInstances(payload).forEach((instance) => {
      const merged = { ...payload, ...instance };
      const layer = poseLayer(merged, side, ownerRecordId, field, includeBoxes);
      if (layer) untracked.set(layer.id, layer);
    });
  }
  if (trackId) return [...tracked.values()].sort((left, right) => left.id.localeCompare(right.id));
  const trackedCandidates = new Set([...tracked.values()].map((layer) => layer.candidateId).filter(Boolean));
  const combined = new Map(tracked);
  for (const layer of untracked.values()) {
    if (!trackedCandidates.has(layer.candidateId)) combined.set(layer.id, layer);
  }
  return [...combined.values()].sort((left, right) => left.id.localeCompare(right.id));
}

function cropEvidenceLayers(
  records: TraceRecord[],
  side: Side,
  trackId: string,
): EvidenceLayer[] {
  const selectedCandidates = new Set<string>();
  if (trackId) {
    for (const record of records) {
      const payload = payloadOf(record);
      if (
        record.stage === "POSE_2D"
        && payload.view_id === side
        && payload.track_id === trackId
        && typeof payload.candidate_id === "string"
      ) selectedCandidates.add(payload.candidate_id);
    }
  }
  return records.flatMap((record, index) => {
    if (
      record.stage !== "POSE_2D"
      || (record.event !== "virtual_crop_pose_inferred" && record.event !== "virtual_crop_pose_not_produced")
    ) return [];
    const payload = payloadOf(record);
    if (payload.view_id !== side || typeof payload.candidate_id !== "string") return [];
    const candidateId = payload.candidate_id;
    if (trackId && !selectedCandidates.has(candidateId)) return [];
    const cropArtifact = artifactsOf(record).find((artifact) => (
      artifact.role === "virtual_crop" && isAllowedBackground(artifact)
    ));
    if (!cropArtifact) return [];
    const camera = objectValue(payload.virtual_camera);
    const [width, height] = positiveSize(camera?.output_size, [256, 256]);
    const points = outputProduced(record) ? nullablePoints2(payload.keypoints_uv_crop) : [];
    const pose: PoseLayer[] = points.length === 21 ? [{
      id: trackId || candidateId,
      candidateId,
      ownerRecordId: record.record_id ?? `${side}-crop-${index}`,
      points,
      scores: scoreValues(payload.keypoint_scores),
      scoreThreshold: keypointScoreThreshold(payload),
      boxes: [],
    }] : [];
    return [{
      id: `crop:${side}:${record.record_id ?? index}:${candidateId}`,
      selectionKey: `crop:${side}:${candidateId}`,
      label: `VIRTUAL CROP · ${candidateId}`,
      artifact: cropArtifact,
      artifactRole: `virtual_crop · ${candidateId}`,
      coordinateSpace: "crop" as const,
      stageLabel: `POSE_2D · CROP · ${candidateId}`,
      poses: pose,
      width,
      height,
      noOverlayReason: `${candidateId} 未提供 CROP UV`,
    }];
  });
}

function projectedEvidenceLayer(
  records: TraceRecord[],
  side: Side,
  trackId: string,
  stage: ProjectedStage,
  rectified: ArtifactRef | undefined,
  size: [number, number],
): EvidenceLayer | null {
  const stageRecords = records.filter((record) => record.stage === stage);
  if (stageRecords.length === 0) return null;
  const poses = new Map<string, PoseLayer>();
  for (const record of stageRecords) {
    const payload = payloadOf(record);
    const recordTrack = typeof payload.track_id === "string" ? payload.track_id : "";
    if (!recordTrack || (trackId && recordTrack !== trackId) || !outputProduced(record)) continue;
    if (payload.projected_keypoints_space !== "rectified") continue;
    const projected = objectValue(payload.projected_keypoints_uv);
    const points = nullablePoints2(projected?.[side]);
    if (points.length !== 21 || !points.some(Boolean)) continue;
    poses.set(recordTrack, {
      id: recordTrack,
      candidateId: "",
      ownerRecordId: record.record_id ?? `${stage}:${recordTrack}`,
      points,
      scores: [],
      scoreThreshold: DEFAULT_KEYPOINT_SCORE_THRESHOLD,
      boxes: [],
    });
  }
  const ownerIds = stageRecords.map((record, index) => record.record_id ?? `${stage}-${index}`).join(",");
  return {
    id: `rectified:${stage}:${side}:${ownerIds}`,
    selectionKey: `rectified:${stage}:${side}`,
    label: `RECTIFIED · ${PROJECTED_STAGE_LABELS[stage]}`,
    artifact: rectified,
    artifactRole: `rectified_${side}`,
    coordinateSpace: "rectified",
    stageLabel: stage,
    poses: [...poses.values()].sort((left, right) => left.id.localeCompare(right.id)),
    width: size[0],
    height: size[1],
    noOverlayReason: trackId
      ? `${stage} 未提供 ${trackId} 的 RECTIFIED 投影`
      : `${stage} 未提供 RECTIFIED 投影`,
  };
}

function evidenceLayers(
  records: TraceRecord[],
  side: Side,
  trackId: string,
): EvidenceLayer[] {
  const sourceOwner = artifactOwnerByRole(records, `source_${side}`);
  const undistortedOwner = artifactOwnerByRole(records, `undistorted_${side}`);
  const rectifiedOwner = artifactOwnerByRole(records, `rectified_${side}`);
  const source = sourceOwner?.artifact;
  const undistorted = undistortedOwner?.artifact;
  const rectified = rectifiedOwner?.artifact;
  const [nativeWidth, nativeHeight] = viewImageSize(records, side);
  const [rectifiedWidth, rectifiedHeight] = rectifiedImageSize(records);
  const nativePoses = selectedPoseLayers(records, side, trackId, "keypoints_uv", true);
  const rectifiedPoses = selectedPoseLayers(records, side, trackId, "keypoints_uv_rectified", false);
  const nativePoseOwners = nativePoses.map((pose) => pose.ownerRecordId).join(",") || "no-pose";
  const rectifiedPoseOwners = rectifiedPoses.map((pose) => pose.ownerRecordId).join(",") || "no-pose";
  const layers: EvidenceLayer[] = [{
    id: `source:${side}:${sourceOwner?.recordId ?? "no-frame"}:${nativePoseOwners}`,
    selectionKey: `source:${side}`,
    label: "SOURCE · NATIVE POSE_2D",
    artifact: source,
    artifactRole: `source_${side}`,
    coordinateSpace: "native",
    stageLabel: "POSE_2D · NATIVE",
    poses: nativePoses,
    width: nativeWidth,
    height: nativeHeight,
    noOverlayReason: "POSE_2D 未提供 NATIVE UV",
  }];
  if (undistorted) {
    layers.push({
      id: `undistorted:${side}:${undistortedOwner?.recordId ?? "no-frame"}`,
      selectionKey: `undistorted:${side}`,
      label: "UNDISTORTED · NO UV",
      artifact: undistorted,
      artifactRole: `undistorted_${side}`,
      coordinateSpace: "undistorted",
      stageLabel: "UNDISTORTED",
      poses: [],
      width: rectifiedWidth,
      height: rectifiedHeight,
      noOverlayReason: "UNDISTORTED 无严格 UV 映射",
    });
  }
  if (rectified) {
    layers.push({
      id: `rectified:pose_2d:${side}:${rectifiedOwner?.recordId ?? "no-frame"}:${rectifiedPoseOwners}`,
      selectionKey: `rectified:pose_2d:${side}`,
      label: "RECTIFIED · POSE_2D",
      artifact: rectified,
      artifactRole: `rectified_${side}`,
      coordinateSpace: "rectified",
      stageLabel: "POSE_2D · RECTIFIED",
      poses: rectifiedPoses,
      width: rectifiedWidth,
      height: rectifiedHeight,
      noOverlayReason: "POSE_2D 未提供 RECTIFIED UV",
    });
  }
  const projectedStages: ProjectedStage[] = [
    "RAW_FUSION",
    "KINEMATIC_REFINEMENT",
    "TEMPORAL_REFINEMENT",
    "EXPORT",
  ];
  for (const stage of projectedStages) {
    const layer = projectedEvidenceLayer(
      records,
      side,
      trackId,
      stage,
      rectified,
      [rectifiedWidth, rectifiedHeight],
    );
    if (layer) layers.push(layer);
  }
  layers.push(...cropEvidenceLayers(records, side, trackId));
  return layers;
}

interface ViewPanelProps {
  side: Side;
  runKey: string;
  records: TraceRecord[];
  trackId: string;
  overlay: boolean;
}

function PoseOverlayGraphic({
  sideLabel,
  selected,
  poses,
}: {
  sideLabel: string;
  selected: EvidenceLayer;
  poses: PoseLayer[];
}) {
  return (
    <svg
      className="pose-overlay"
      viewBox={`0 0 ${selected.width} ${selected.height}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={`${sideLabel} 2D 叠加层`}
    >
      {poses.map((pose) => (
        <g key={pose.id} data-track-id={pose.id} aria-label={`${pose.id} ${selected.stageLabel}`}>
          {pose.boxes.map(([x1, y1, x2, y2], index) => (
            <rect
              key={`bbox-${index}`}
              x={x1}
              y={y1}
              width={x2 - x1}
              height={y2 - y1}
              className="detection-box"
              vectorEffect="non-scaling-stroke"
            />
          ))}
          {FHP21_EDGES.map(([from, to], index) => {
            const start = pose.points[from];
            const end = pose.points[to];
            if (!start || !end) return null;
            return (
              <line
                key={`${from}-${to}`}
                x1={start[0]}
                y1={start[1]}
                x2={end[0]}
                y2={end[1]}
                stroke={FINGER_COLORS[Math.floor(index / 4)]}
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
          {pose.points.map((point, index) => {
            if (!point) return null;
            const [x, y] = point;
            const isVisible = pose.scores.length === 0
              || (pose.scores[index] ?? Number.NEGATIVE_INFINITY) >= pose.scoreThreshold;
            return (
              <circle
                key={`${x}:${y}:${index}`}
                data-joint-index={index}
                cx={x}
                cy={y}
                r="3.8"
                className={isVisible ? "" : "hidden-keypoint"}
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
        </g>
      ))}
    </svg>
  );
}

function ViewPanel({ side, runKey, records, trackId, overlay }: ViewPanelProps) {
  const layers = useMemo(() => evidenceLayers(records, side, trackId), [records, side, trackId]);
  const [selectedKey, setSelectedKey] = useState("");
  useEffect(() => {
    setSelectedKey((current) => {
      if (layers.some((layer) => layer.selectionKey === current)) return current;
      return layers[0]?.selectionKey ?? "";
    });
  }, [layers]);
  const selected = layers.find((layer) => layer.selectionKey === selectedKey) ?? layers[0];
  const selectedArtifact = selected?.artifact;
  const poses = selected?.poses ?? [];
  const visible = poses.reduce((total, pose) => total + pose.points.filter((point, index) => (
    point !== null
    && (
      pose.scores.length === 0
      || (pose.scores[index] ?? Number.NEGATIVE_INFINITY) >= pose.scoreThreshold
    )
  )).length, 0);
  const scores = poses.flatMap((pose) => pose.scores);
  const sideLabel = side === "left" ? "左目" : "右目";
  const renderPoseOverlay = () => (
    overlay && selected && selectedArtifact && poses.length > 0
      ? <PoseOverlayGraphic sideLabel={sideLabel} selected={selected} poses={poses} />
      : undefined
  );

  return (
    <article className="view-panel">
      <header>
        <div>
          <span className="camera-index">{side === "left" ? "L" : "R"}</span>
          <strong>{sideLabel}</strong>
          <small>{String(selected?.coordinateSpace ?? side).toUpperCase()} SPACE</small>
        </div>
        {layers.length > 1 ? (
          <select
            className="artifact-select"
            aria-label={`${sideLabel}证据层`}
            value={selectedKey}
            onChange={(event) => setSelectedKey(event.target.value)}
          >
            {layers.map((layer) => (
              <option key={layer.id} value={layer.selectionKey}>{layer.label}</option>
            ))}
          </select>
        ) : (
          <span className="artifact-role"><ImageIcon /> {selected?.label ?? "NO EVIDENCE"}</span>
        )}
      </header>

      <div className="image-stage">
        {selectedArtifact ? (
          <PreviewableImage
            src={traceApi.artifactUrl(runKey, artifactPath(selectedArtifact))}
            alt={`${sideLabel} ${selected.artifactRole}`}
            previewOverlay={renderPoseOverlay()}
          />
        ) : (
          <div className="no-artifact"><ImageIcon /><span>此视角没有图像工件</span></div>
        )}

        {renderPoseOverlay()}
        <div className="image-reticle" aria-hidden="true"><i /><i /></div>
      </div>

      <footer>
        {selected && !selectedArtifact ? (
          <>
            <span className="overlay-binding-gap">无可叠加证据 · 本帧未保存 {selected.artifactRole}</span>
            {poses.length > 0 && (
              <span className={visible === poses.length * 21 ? "" : "landmark-gap"}>
                {visible} / {poses.length * 21} visible
              </span>
            )}
          </>
        ) : selected && poses.length === 0 ? (
          <span className="overlay-binding-gap">无对应骨骼 · {selected.noOverlayReason}</span>
        ) : (
          <>
            <span><i className="dot success" /> {selected?.stageLabel ?? "NO POSE"}</span>
            <span className={visible === poses.length * 21 ? "" : "landmark-gap"}>
              {poses.length ? `${visible} / ${poses.length * 21} visible` : "no landmarks"}
            </span>
            <span>{scores.length ? `${(Math.max(...scores) * 100).toFixed(1)}% max` : "score —"}</span>
          </>
        )}
      </footer>
    </article>
  );
}

export function StereoEvidence({ runKey, records, trackId }: StereoEvidenceProps) {
  const [overlay, setOverlay] = useState(true);
  return (
    <section className="evidence-card stereo-card">
      <header className="card-header">
        <div>
          <span className="section-index">03</span>
          <div><h2>双目证据</h2><p>图像、坐标空间与阶段骨骼原子绑定</p></div>
        </div>
        <button
          type="button"
          className={`overlay-toggle ${overlay ? "active" : ""}`}
          aria-pressed={overlay}
          onClick={() => setOverlay((value) => !value)}
        >
          {overlay ? <Eye /> : <Layers2 />} 2D OVERLAY
        </button>
      </header>
      <div className="stereo-grid">
        <ViewPanel side="left" runKey={runKey} records={records} trackId={trackId} overlay={overlay} />
        <ViewPanel side="right" runKey={runKey} records={records} trackId={trackId} overlay={overlay} />
      </div>
    </section>
  );
}
