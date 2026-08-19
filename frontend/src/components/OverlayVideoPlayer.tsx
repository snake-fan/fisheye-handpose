import { ChevronLeft, ChevronRight, Download, Film, LoaderCircle } from "lucide-react";
import {
  type CSSProperties,
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";

import { traceApi } from "../api/client";
import type {
  ArtifactRef,
  OverlayVideoTimeline,
  RunDetail,
  TraceRecord,
} from "../api/types";
import {
  artifactPathFor,
  createFrameEvidence,
  type FrameEvidenceRecord,
} from "../domain/frameEvidence";
import { SCHEMA_IDS } from "../domain/projectContract.generated";
import { artifactsOf, payloadOf } from "../domain/trace";

interface OverlayVideoPlayerProps {
  runKey: string;
  detail: RunDetail;
}

interface ArtifactEvidence {
  record: FrameEvidenceRecord;
  artifact: ArtifactRef;
  timelineArtifact?: ArtifactRef;
}

interface ParsedTimelineFrame {
  sourceFrameIndex: number;
  startSeconds: number;
  seekSeconds: number;
}

type TimelineLoadState = "loading" | "ready" | "error" | "missing";
type TimelineMode = "exact" | "estimated";

type TimelineTrackStyle = CSSProperties & {
  "--frame-progress": string;
  "--frame-tick-gap": string;
};

const VIDEO_ROLE = "overlay_video_raw_vs_stable_stereo_rectified";
const TIMELINE_ROLE = "overlay_video_timeline";
const MEDIA_TIME_EPSILON_SECONDS = 0.000001;

function latestOverlayEvidence(records: TraceRecord[]): ArtifactEvidence | undefined {
  const evidence = createFrameEvidence(records);
  for (let index = evidence.records.length - 1; index >= 0; index -= 1) {
    const record = evidence.records[index];
    if (
      record.stage !== "EXPORT"
      || record.status !== "SUCCEEDED"
      || record.payload.output_status !== "PRODUCED"
    ) continue;
    const artifact = artifactsOf(record).find((candidate) => candidate.role === VIDEO_ROLE);
    if (!artifact) continue;
    const timelineArtifact = artifactsOf(record).find((candidate) => candidate.role === TIMELINE_ROLE);
    return { record, artifact, timelineArtifact };
  }
  return undefined;
}

function objectValue(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function finiteInteger(value: unknown, minimum: number): number | undefined {
  return typeof value === "number"
    && Number.isSafeInteger(value)
    && value >= minimum
    ? value
    : undefined;
}

function parseTimeline(value: unknown): ParsedTimelineFrame[] | undefined {
  const document = objectValue(value);
  if (
    document?.schema_version !== SCHEMA_IDS.OVERLAY_VIDEO_TIMELINE
    || !Array.isArray(document.frames)
  ) {
    return undefined;
  }
  const timeBase = objectValue(document.time_base);
  const numerator = finiteInteger(timeBase?.numerator, 1);
  const denominator = finiteInteger(timeBase?.denominator, 1);
  if (numerator === undefined || denominator === undefined || document.frames.length === 0) {
    return undefined;
  }
  const parsed: ParsedTimelineFrame[] = [];
  let previousEndPoints = 0;
  for (let index = 0; index < document.frames.length; index += 1) {
    const frame = objectValue(document.frames[index]);
    const videoFrameIndex = finiteInteger(frame?.video_frame_index, 0);
    const sourceFrameIndex = finiteInteger(frame?.frame_index, 0);
    const videoPoints = finiteInteger(frame?.video_pts, 0);
    const durationPoints = finiteInteger(frame?.duration_pts, 1);
    const frameId = frame?.frame_id;
    if (
      videoFrameIndex !== index
      || sourceFrameIndex === undefined
      || videoPoints === undefined
      || durationPoints === undefined
      || typeof frameId !== "string"
      || !frameId
      || videoPoints !== previousEndPoints
    ) {
      return undefined;
    }
    const startSeconds = videoPoints * numerator / denominator;
    const endSeconds = (videoPoints + durationPoints) * numerator / denominator;
    const seekSeconds = (videoPoints + durationPoints / 2) * numerator / denominator;
    if (![startSeconds, endSeconds, seekSeconds].every(Number.isFinite)) return undefined;
    parsed.push({
      sourceFrameIndex,
      startSeconds,
      seekSeconds,
    });
    previousEndPoints = videoPoints + durationPoints;
  }
  return parsed;
}

function fallbackTimeline(
  frameCount: number,
  frameRateValue: unknown,
  videoDuration: number,
): ParsedTimelineFrame[] | undefined {
  if (frameCount <= 0) return undefined;
  const frameRate = objectValue(frameRateValue);
  const rateNumerator = finiteInteger(frameRate?.numerator, 1);
  const rateDenominator = finiteInteger(frameRate?.denominator, 1);
  const frameDuration = rateNumerator !== undefined && rateDenominator !== undefined
    ? rateDenominator / rateNumerator
    : Number.isFinite(videoDuration) && videoDuration > 0
      ? videoDuration / frameCount
      : undefined;
  if (frameDuration === undefined || !Number.isFinite(frameDuration) || frameDuration <= 0) {
    return undefined;
  }
  return Array.from({ length: frameCount }, (_, index) => ({
    sourceFrameIndex: index,
    startSeconds: index * frameDuration,
    seekSeconds: (index + 0.5) * frameDuration,
  }));
}

function frameIndexAtTime(frames: ParsedTimelineFrame[], seconds: number): number {
  if (!frames.length || !Number.isFinite(seconds) || seconds <= frames[0].startSeconds) return 0;
  let low = 0;
  let high = frames.length - 1;
  let match = 0;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    if (frames[middle].startSeconds <= seconds + MEDIA_TIME_EPSILON_SECONDS) {
      match = middle;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  return match;
}

function displayVideoTime(seconds: number): string {
  const totalMilliseconds = Math.max(0, Math.round(seconds * 1000));
  const minutes = Math.floor(totalMilliseconds / 60_000);
  const remaining = totalMilliseconds - minutes * 60_000;
  const wholeSeconds = Math.floor(remaining / 1000);
  const milliseconds = remaining % 1000;
  return `${String(minutes).padStart(2, "0")}:${String(wholeSeconds).padStart(2, "0")}.${String(milliseconds).padStart(3, "0")}`;
}

function AvailableOverlayVideoPlayer({
  runKey,
  detail,
  videoEvidence,
  timelineArtifact,
}: {
  runKey: string;
  detail: RunDetail;
  videoEvidence: ArtifactEvidence;
  timelineArtifact?: ArtifactRef;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const sliderId = useId();
  const timelinePath = timelineArtifact ? artifactPathFor(timelineArtifact) : "";
  const payload = payloadOf(videoEvidence.record);
  const declaredFrameCount = typeof payload.frame_count === "number"
    ? payload.frame_count
    : detail.run.frame_count;
  const frameCount = finiteInteger(declaredFrameCount, 1) ?? 0;
  const [timelineState, setTimelineState] = useState<TimelineLoadState>(
    timelinePath ? "loading" : "missing",
  );
  const [timelineMode, setTimelineMode] = useState<TimelineMode>("exact");
  const [timelineFrames, setTimelineFrames] = useState<ParsedTimelineFrame[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [mediaReady, setMediaReady] = useState(false);
  const [mediaDuration, setMediaDuration] = useState(Number.NaN);

  useEffect(() => {
    if (!timelinePath) {
      setTimelineFrames([]);
      setTimelineState("missing");
      return undefined;
    }
    const controller = new AbortController();
    setTimelineFrames([]);
    setTimelineState("loading");
    void traceApi.getArtifactJson<OverlayVideoTimeline>(
      runKey,
      timelinePath,
      controller.signal,
    ).then((value) => {
      const frames = parseTimeline(value);
      if (!frames) throw new Error("invalid overlay video timeline");
      setTimelineFrames(frames);
      setTimelineMode("exact");
      setCurrentFrame(0);
      setTimelineState("ready");
    }).catch(() => {
      if (controller.signal.aborted) return;
      setTimelineFrames([]);
      setTimelineState("error");
    });
    return () => controller.abort();
  }, [runKey, timelinePath]);

  useEffect(() => {
    if (
      !mediaReady
      || timelineFrames.length > 0
      || (timelineState !== "missing" && timelineState !== "error")
    ) {
      return;
    }
    const frames = fallbackTimeline(frameCount, payload.frame_rate, mediaDuration);
    if (!frames) return;
    setTimelineFrames(frames);
    setTimelineMode("estimated");
    setCurrentFrame(frameIndexAtTime(frames, videoRef.current?.currentTime ?? 0));
    setTimelineState("ready");
  }, [
    frameCount,
    mediaDuration,
    mediaReady,
    payload.frame_rate,
    timelineFrames.length,
    timelineState,
  ]);

  const syncCurrentFrame = useCallback((seconds: number) => {
    if (!timelineFrames.length) return;
    setCurrentFrame(frameIndexAtTime(timelineFrames, seconds));
  }, [timelineFrames]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || timelineState !== "ready" || !timelineFrames.length) return undefined;
    const syncFromElement = () => syncCurrentFrame(video.currentTime);
    video.addEventListener("timeupdate", syncFromElement);
    video.addEventListener("seeked", syncFromElement);
    video.addEventListener("ended", syncFromElement);

    let callbackId: number | undefined;
    let stopped = false;
    if (typeof video.requestVideoFrameCallback === "function") {
      const scheduleFrameCallback = () => {
        if (stopped) return;
        callbackId = video.requestVideoFrameCallback((_now, metadata) => {
          if (stopped) return;
          syncCurrentFrame(metadata.mediaTime);
          scheduleFrameCallback();
        });
      };
      scheduleFrameCallback();
    }
    syncFromElement();
    return () => {
      stopped = true;
      video.removeEventListener("timeupdate", syncFromElement);
      video.removeEventListener("seeked", syncFromElement);
      video.removeEventListener("ended", syncFromElement);
      if (callbackId !== undefined && typeof video.cancelVideoFrameCallback === "function") {
        video.cancelVideoFrameCallback(callbackId);
      }
    };
  }, [syncCurrentFrame, timelineFrames, timelineState]);

  const seekToFrame = useCallback((requestedFrame: number) => {
    const video = videoRef.current;
    if (!video || !mediaReady || !timelineFrames.length) return;
    const boundedFrame = Math.max(0, Math.min(timelineFrames.length - 1, requestedFrame));
    video.pause();
    video.currentTime = timelineFrames[boundedFrame].seekSeconds;
    setCurrentFrame(boundedFrame);
  }, [mediaReady, timelineFrames]);

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
  const activeFrame = timelineFrames[currentFrame];
  const displayedFrameCount = timelineFrames.length || frameCount;
  const frameProgress = timelineFrames.length <= 1
    ? 0
    : currentFrame / (timelineFrames.length - 1) * 100;
  const visibleTickDivisions = Math.max(1, Math.min(80, timelineFrames.length - 1));
  const trackStyle: TimelineTrackStyle = {
    "--frame-progress": `${frameProgress}%`,
    "--frame-tick-gap": `${100 / visibleTickDivisions}%`,
  };

  return (
    <section className="overlay-video-card" aria-label="骨架抖动视频">
      <header className="card-header">
        <div>
          <span className="section-index">PLAY</span>
          <div><h2>Raw vs Stable · 双目抖动检查</h2><p>{inputLabel} → {temporalMethod}</p></div>
        </div>
        <span className="video-space-label">{displayedFrameCount} frames · {pixelSpace}</span>
      </header>
      <video
        ref={videoRef}
        src={traceApi.artifactUrl(runKey, artifactPathFor(videoEvidence.artifact))}
        aria-label="Raw 与 Stable 双目骨架抖动对比视频"
        controls
        preload="metadata"
        onLoadedMetadata={(event) => {
          setMediaReady(true);
          setMediaDuration(event.currentTarget.duration);
          syncCurrentFrame(event.currentTarget.currentTime);
        }}
      />
      {timelineState !== "missing" && (
        <section className={`video-frame-timeline ${timelineState}`} aria-label="视频帧时间轴">
          {timelineState === "loading" && (
            <div className="video-frame-message" role="status">
              <LoaderCircle className="spin" aria-hidden="true" /> 正在加载精确帧映射
            </div>
          )}
          {timelineState === "error" && (
            <div className="video-frame-message error" role="status">
              帧时间轴不可用，仍可使用视频进度条
            </div>
          )}
          {timelineState === "ready" && activeFrame && (
            <div className="video-frame-controls">
              <button
                type="button"
                aria-label="上一帧"
                disabled={!mediaReady || currentFrame === 0}
                onClick={() => seekToFrame(currentFrame - 1)}
              ><ChevronLeft aria-hidden="true" /></button>
              <div className="video-frame-scrubber">
                <div className="video-frame-readout">
                  <label htmlFor={sliderId}>
                    帧时间轴{timelineMode === "estimated" && <small> · CFR 估算</small>}
                  </label>
                  <output htmlFor={sliderId}>
                    {currentFrame + 1} / {timelineFrames.length}
                    <span>
                      {timelineMode === "exact"
                        ? `源帧 ${String(activeFrame.sourceFrameIndex).padStart(6, "0")} · `
                        : "视频时间 "}
                      {displayVideoTime(activeFrame.startSeconds)}
                    </span>
                  </output>
                </div>
                <div className="video-frame-track" style={trackStyle}>
                  <input
                    id={sliderId}
                    className="video-frame-slider"
                    type="range"
                    min="0"
                    max={timelineFrames.length - 1}
                    step="1"
                    value={currentFrame}
                    disabled={!mediaReady}
                    aria-label="帧时间轴"
                    aria-valuetext={`第 ${currentFrame + 1} 帧，共 ${timelineFrames.length} 帧`}
                    onChange={(event) => seekToFrame(Number(event.currentTarget.value))}
                  />
                </div>
                <div className="video-frame-scale" aria-hidden="true">
                  <span>1</span><span>拖动或按 ← → 逐帧</span><span>{timelineFrames.length}</span>
                </div>
              </div>
              <button
                type="button"
                aria-label="下一帧"
                disabled={!mediaReady || currentFrame === timelineFrames.length - 1}
                onClick={() => seekToFrame(currentFrame + 1)}
              ><ChevronRight aria-hidden="true" /></button>
            </div>
          )}
        </section>
      )}
      <footer className="overlay-video-meta">
        <span className={rawInput || !manoInput ? "video-source-warning" : "video-source-ok"}>
          {sourceLabel}
        </span>
        {timelineArtifact && (
          <a href={traceApi.artifactUrl(runKey, artifactPathFor(timelineArtifact))} download>
            <Download aria-hidden="true" /> 下载帧时间映射
          </a>
        )}
      </footer>
    </section>
  );
}

export function OverlayVideoPlayer({ runKey, detail }: OverlayVideoPlayerProps) {
  const records = detail.global_records ?? [];
  const videoEvidence = latestOverlayEvidence(records);
  if (!videoEvidence) {
    return (
      <section className="overlay-video-card unavailable" aria-label="骨架抖动视频">
        <Film aria-hidden="true" />
        <div><strong>此运行没有叠加视频</strong><span>逐帧节点对比仍然可用</span></div>
      </section>
    );
  }
  const { timelineArtifact } = videoEvidence;
  const videoPath = artifactPathFor(videoEvidence.artifact);
  return (
    <AvailableOverlayVideoPlayer
      key={`${runKey}:${videoPath}`}
      runKey={runKey}
      detail={detail}
      videoEvidence={videoEvidence}
      timelineArtifact={timelineArtifact}
    />
  );
}
