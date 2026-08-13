import { ChevronLeft, ChevronRight, LoaderCircle } from "lucide-react";

import type { FrameSummary } from "../api/types";

interface FrameTimelineProps {
  frames: FrameSummary[];
  total: number;
  offset: number;
  limit: number;
  selectedFrameKey: string;
  loading: boolean;
  error: string;
  onSelect: (frameKey: string) => void;
  onPage: (offset: number) => void;
}

function displayTime(timestampNs: number | null): string {
  if (timestampNs === null) return "—";
  return `${(timestampNs / 1_000_000_000).toFixed(3)} s`;
}

export function FrameTimeline({
  frames,
  total,
  offset,
  limit,
  selectedFrameKey,
  loading,
  error,
  onSelect,
  onPage,
}: FrameTimelineProps) {
  const pageEnd = offset + frames.length;
  const hasPrevious = offset > 0;
  const hasNext = frames.length > 0 && pageEnd < total;
  return (
    <section className="timeline" aria-label="帧时间轴">
      <div className="timeline-head">
        <div>
          <span className="section-index">02</span>
          <h2>帧时间轴</h2>
          <span>{total.toLocaleString()} MATCHES</span>
        </div>
        <div className="timeline-legend">
          <span><i className="dot success" /> SUCCEEDED</span>
          <span><i className="dot warning" /> WARNING</span>
          {total > 0 && (
            <span className="page-range">{offset + 1}–{Math.min(pageEnd, total)} / {total}</span>
          )}
          <button
            type="button"
            aria-label="上一组帧"
            disabled={loading || !hasPrevious}
            onClick={() => onPage(Math.max(0, offset - limit))}
          ><ChevronLeft /></button>
          <button
            type="button"
            aria-label="下一组帧"
            disabled={loading || !hasNext}
            onClick={() => onPage(pageEnd)}
          ><ChevronRight /></button>
        </div>
      </div>
      <div className="frame-strip">
        {loading && <div className="timeline-message"><LoaderCircle className="spin" /> 正在索引帧</div>}
        {error && <div className="timeline-message error" role="alert">{error}</div>}
        {!loading && !error && frames.length === 0 && (
          <div className="timeline-message">当前筛选没有帧证据</div>
        )}
        {frames.map((frame) => {
          const hasFailure = frame.statuses.includes("FAILED");
          const hasWarning = frame.statuses.includes("WARNING");
          const tone = hasFailure ? "failed" : hasWarning ? "warning" : "success";
          return (
            <button
              key={frame.frame_key}
              type="button"
              className={`frame-tick ${tone} ${selectedFrameKey === frame.frame_key ? "selected" : ""}`}
              onClick={() => onSelect(frame.frame_key)}
              aria-pressed={selectedFrameKey === frame.frame_key}
              aria-label={`帧 ${frame.frame_index ?? frame.frame_id}，${frame.stages.join("、") || "无阶段"}`}
            >
              <span className="frame-number">{frame.frame_index === null ? "NO INDEX" : String(frame.frame_index).padStart(6, "0")}</span>
              <span className="frame-stage">{frame.stages.at(-1) ?? "NO STAGE"}</span>
              <span className="frame-time">{displayTime(frame.timestamp_ns)}</span>
              <i />
            </button>
          );
        })}
      </div>
    </section>
  );
}
