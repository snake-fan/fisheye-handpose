import { Eye, ImageIcon, Layers2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { traceApi } from "../api/client";
import type { ArtifactRef, TraceRecord } from "../api/types";
import {
  FHP21_EDGES,
  FINGER_COLORS,
  posePayloadOf,
  points2,
  scoreValues,
  viewArtifacts,
  viewImageSize,
  viewPoseRecord,
} from "../domain/trace";

interface StereoEvidenceProps {
  runKey: string;
  records: TraceRecord[];
  trackId: string;
}

const DEFAULT_KEYPOINT_SCORE_THRESHOLD = 0.2;

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

function artifactRole(artifact: ArtifactRef): string {
  return String(artifact.role ?? "artifact");
}

function isVideo(artifact: ArtifactRef): boolean {
  const mediaType = String(artifact.media_type ?? "").toLowerCase();
  return mediaType.startsWith("video/") || /\.(mp4|webm|mov)$/i.test(artifactPath(artifact));
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

interface ViewPanelProps {
  side: "left" | "right";
  runKey: string;
  records: TraceRecord[];
  trackId: string;
  overlay: boolean;
}

function ViewPanel({ side, runKey, records, trackId, overlay }: ViewPanelProps) {
  const artifacts = useMemo(() => viewArtifacts(records, side), [records, side]);
  const [selectedPath, setSelectedPath] = useState("");
  useEffect(() => {
    setSelectedPath((current) => {
      if (artifacts.some((artifact) => artifactPath(artifact) === current)) return current;
      return artifacts[0] ? artifactPath(artifacts[0]) : "";
    });
  }, [artifacts]);
  const selected = artifacts.find((artifact) => artifactPath(artifact) === selectedPath) ?? artifacts[0];
  const pose = viewPoseRecord(records, side, trackId);
  const posePayload = pose ? posePayloadOf(pose, trackId) : {};
  const keypoints = points2(posePayload.keypoints_uv);
  const scores = scoreValues(posePayload.keypoint_scores);
  const scoreThreshold = keypointScoreThreshold(posePayload);
  const keypointVisible = (index: number) => (
    scores.length === 0 || (scores[index] ?? Number.NEGATIVE_INFINITY) >= scoreThreshold
  );
  const visible = keypoints.filter((_, index) => keypointVisible(index)).length;
  const [width, height] = viewImageSize(records, side);
  const sideLabel = side === "left" ? "左目" : "右目";

  return (
    <article className="view-panel">
      <header>
        <div>
          <span className="camera-index">{side === "left" ? "L" : "R"}</span>
          <strong>{sideLabel}</strong>
          <small>{String(posePayload.view_id ?? side).toUpperCase()} CAMERA</small>
        </div>
        {artifacts.length > 1 ? (
          <select
            className="artifact-select"
            aria-label={`${sideLabel}工件`}
            value={selectedPath}
            onChange={(event) => setSelectedPath(event.target.value)}
          >
            {artifacts.map((artifact) => (
              <option key={`${artifactPath(artifact)}:${artifactRole(artifact)}`} value={artifactPath(artifact)}>
                {artifactRole(artifact)}
              </option>
            ))}
          </select>
        ) : (
          <span className="artifact-role"><ImageIcon /> {selected ? artifactRole(selected) : "NO ARTIFACT"}</span>
        )}
      </header>

      <div className="image-stage">
        {selected ? (
          isVideo(selected) ? (
            <video
              src={traceApi.artifactUrl(runKey, artifactPath(selected))}
              aria-label={`${sideLabel} ${artifactRole(selected)}`}
              controls
              preload="metadata"
            />
          ) : (
            <img
              src={traceApi.artifactUrl(runKey, artifactPath(selected))}
              alt={`${sideLabel} ${artifactRole(selected)}`}
            />
          )
        ) : (
          <div className="no-artifact"><ImageIcon /><span>此视角没有图像工件</span></div>
        )}

        {overlay && keypoints.length > 0 && (
          <svg
            className="pose-overlay"
            viewBox={`0 0 ${width} ${height}`}
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label={`${sideLabel} 2D 叠加层`}
          >
            {detections(posePayload).map(([x1, y1, x2, y2], index) => (
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
              const start = keypoints[from];
              const end = keypoints[to];
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
            {keypoints.map(([x, y], index) => (
              <circle
                key={`${x}:${y}:${index}`}
                cx={x}
                cy={y}
                r="3.8"
                className={keypointVisible(index) ? "" : "hidden-keypoint"}
                vectorEffect="non-scaling-stroke"
              />
            ))}
          </svg>
        )}
        <div className="image-reticle" aria-hidden="true"><i /><i /></div>
      </div>

      <footer>
        <span><i className="dot success" /> {pose?.stage ?? "NO POSE"}</span>
        <span className={visible === 21 ? "" : "landmark-gap"}>
          {keypoints.length ? `${visible} / 21 visible` : "no landmarks"}
        </span>
        <span>{scores.length ? `${(Math.max(...scores) * 100).toFixed(1)}% max` : "score —"}</span>
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
          <div><h2>双目证据</h2><p>同步影像与物理像素空间 2D 关键点</p></div>
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
