import { ImageIcon } from "lucide-react";

import { traceApi } from "../api/client";
import type { ArtifactRef, TraceRecord } from "../api/types";
import { artifactsOf, FHP21_EDGES, payloadOf } from "../domain/trace";
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
}

interface AssociationMatch {
  matchId: string;
  leftCandidateId: string;
  rightCandidateId: string;
  trackId: string;
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
  if (payload.selection && typeof payload.selection === "object") {
    const decision = (payload.selection as Record<string, unknown>).decision;
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
      const bbox = detection.bbox_xyxy;
      if (
        !Array.isArray(bbox)
        || bbox.length !== 4
        || !bbox.every(finiteNumber)
      ) return;
      const id = String(detection.candidate_id ?? `${side}-candidate-${index}`);
      const score = finiteNumber(detection.bbox_score)
        ? detection.bbox_score
        : finiteNumber(detection.score) ? detection.score : null;
      found.set(id, { id, bbox: [bbox[0], bbox[1], bbox[2], bbox[3]], score });
    });
  }
  return [...found.values()].sort((left, right) => left.id.localeCompare(right.id));
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
}: {
  detections: DetectionLayer[];
  width: number;
  height: number;
  label: string;
}) {
  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label={label}>
      {detections.map((detection) => {
        const [x1, y1, x2, y2] = detection.bbox;
        return (
          <g key={detection.id} aria-label={`${detection.id} detection`} stroke={trackColor(detection.id)}>
            <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="none" vectorEffect="non-scaling-stroke" />
            <text x={x1 + 4} y={Math.max(12, y1 - 5)} fill={trackColor(detection.id)} stroke="none">
              {detection.id}{detection.score === null ? "" : ` ${(detection.score * 100).toFixed(0)}%`}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function DetectionComparison({ runKey, records }: { runKey: string; records: TraceRecord[] }) {
  return (
    <div className="native-comparison">
      <header className="coordinate-space-banner">SOURCE → DETECTION · NATIVE FISHEYE PIXELS</header>
      <div className="stage-comparison-stereo">
        {SIDES.map((side) => {
          const label = sideLabel(side);
          const source = artifactByRole(records, `source_${side}`);
          const detections = detectionsForSide(records, side);
          const [width, height] = imageSize(records, side);
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
      trackId: typeof assignment?.track_id === "string" ? assignment.track_id : matchId,
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
          <span key={match.matchId}>{match.matchId} → {match.trackId}</span>
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
            return { ...candidate, id: match?.trackId ?? `${candidate.id} · UNMATCHED` };
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

function RawComparison({ runKey, records, selectedTrack }: Omit<StageComparisonProps, "selectedNodeId">) {
  const matches = associationMatches(records);
  return (
    <ProjectionTransition
      runKey={runKey}
      records={records}
      selectedTrack={selectedTrack}
      heading="ASSOCIATED 2D → STEREO TRIANGULATION · RECTIFIED PIXEL SPACE"
      beforeLabel="ASSOCIATION"
      afterLabel="Raw 3D"
      beforeLayers={(side) => poseLayers(records, side, "rectified").map((layer) => {
        const match = matches.find((value) => (
          (side === "left" ? value.leftCandidateId : value.rightCandidateId)
            === (layer.candidateId ?? layer.id)
        ));
        return { ...layer, id: match?.trackId ?? layer.id };
      })}
      afterLayers={(side) => projectedLayers(records, "RAW_FUSION", side)}
    />
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
  return (
    <>
      <div className="stage-path-list" aria-label="每手实际时序输入">
        {temporalRecords.map((record, index) => {
          const payload = payloadOf(record);
          const track = typeof payload.track_id === "string" ? payload.track_id : "NO_TRACK";
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
