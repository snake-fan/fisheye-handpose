import { Download, Film } from "lucide-react";

import { traceApi } from "../api/client";
import type { ArtifactRef, RunDetail, TraceRecord } from "../api/types";
import { artifactsOf, payloadOf } from "../domain/trace";

interface OverlayVideoPlayerProps {
  runKey: string;
  detail: RunDetail;
}

const VIDEO_ROLE = "overlay_video_raw_vs_stable_stereo_rectified";
const TIMELINE_ROLE = "overlay_video_timeline";

function artifactPath(artifact: ArtifactRef): string {
  return String(artifact.relative_path ?? artifact.path ?? "");
}

function recordWithRole(records: TraceRecord[], role: string) {
  for (const record of records) {
    const artifact = artifactsOf(record).find((candidate) => candidate.role === role);
    if (artifact) return { record, artifact };
  }
  return undefined;
}

export function OverlayVideoPlayer({ runKey, detail }: OverlayVideoPlayerProps) {
  const records = detail.global_records ?? [];
  const videoEvidence = recordWithRole(records, VIDEO_ROLE);
  if (!videoEvidence) {
    return (
      <section className="overlay-video-card unavailable" aria-label="骨架抖动视频">
        <Film aria-hidden="true" />
        <div><strong>此运行没有叠加视频</strong><span>逐帧节点对比仍然可用</span></div>
      </section>
    );
  }
  const timeline = recordWithRole(records, TIMELINE_ROLE)?.artifact;
  const payload = payloadOf(videoEvidence.record);
  const stableInputStages = Array.isArray(payload.stable_input_stages)
    ? [...new Set(payload.stable_input_stages.filter(
      (value): value is string => typeof value === "string" && Boolean(value),
    ))].sort((left, right) => {
      const rank: Record<string, number> = { RAW_FUSION: 0, KINEMATIC_REFINEMENT: 1 };
      return (rank[left] ?? 99) - (rank[right] ?? 99) || left.localeCompare(right);
    })
    : [];
  const legacyInput = payload.actual_input_stage ?? payload.input_stage;
  const inputStages = stableInputStages.length
    ? stableInputStages
    : typeof legacyInput === "string" && legacyInput ? [legacyInput] : [];
  const inputLabel = inputStages.length ? inputStages.join(" + ") : "UNKNOWN_INPUT";
  const temporalMethod = String(payload.temporal_method ?? payload.method ?? "TEMPORAL_REFINEMENT");
  const frameCount = typeof payload.frame_count === "number"
    ? payload.frame_count
    : detail.run.frame_count;
  const pixelSpace = String(payload.image_space ?? payload.projected_keypoints_space ?? "rectified").toUpperCase();
  const rawInput = inputStages.includes("RAW_FUSION");
  const manoInput = inputStages.includes("KINEMATIC_REFINEMENT");
  const sourceLabel = rawInput && manoInput
    ? "混合输入 · RAW / MANO → EMA"
    : rawInput
      ? "MANO 未产出 · RAW → EMA"
      : manoInput
        ? "MANO → Temporal"
        : "输入来源未记录";

  return (
    <section className="overlay-video-card" aria-label="骨架抖动视频">
      <header className="card-header">
        <div>
          <span className="section-index">PLAY</span>
          <div><h2>Raw vs Stable · 双目抖动检查</h2><p>{inputLabel} → {temporalMethod}</p></div>
        </div>
        <span className="video-space-label">{frameCount} frames · {pixelSpace}</span>
      </header>
      <video
        src={traceApi.artifactUrl(runKey, artifactPath(videoEvidence.artifact))}
        aria-label="Raw 与 Stable 双目骨架抖动对比视频"
        controls
        preload="metadata"
      />
      <footer className="overlay-video-meta">
        <span className={rawInput || !manoInput ? "video-source-warning" : "video-source-ok"}>
          {sourceLabel}
        </span>
        {timeline && (
          <a href={traceApi.artifactUrl(runKey, artifactPath(timeline))} download>
            <Download aria-hidden="true" /> 下载帧时间映射
          </a>
        )}
      </footer>
    </section>
  );
}
