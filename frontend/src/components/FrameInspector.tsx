import { Clock3, LoaderCircle } from "lucide-react";
import { useState } from "react";

import type { FrameDetail, RunDetail } from "../api/types";
import { InspectorTabs } from "./InspectorTabs";
import { PipelineNodeRail, type PipelineNodeId } from "./PipelineNodeRail";
import { SkeletonCanvas } from "./SkeletonCanvas";
import { StageComparison } from "./StageComparison";
import { StereoEvidence } from "./StereoEvidence";

interface FrameInspectorProps {
  runKey: string;
  runDetail: RunDetail;
  frameDetail: FrameDetail | null;
  selectedTrack: string;
  loading: boolean;
  error: string;
}

export function FrameInspector({
  runKey,
  runDetail,
  frameDetail,
  selectedTrack,
  loading,
  error,
}: FrameInspectorProps) {
  const [selectedNodeId, setSelectedNodeId] = useState<PipelineNodeId>("SOURCE_RGB");
  if (loading && !frameDetail) {
    return <section className="inspection-state"><LoaderCircle className="spin" /><span>正在组装帧证据</span></section>;
  }
  if (error) {
    return <section className="inspection-state error" role="alert"><strong>帧加载失败</strong><span>{error}</span></section>;
  }
  if (!frameDetail) {
    const globalRecords = runDetail.global_records ?? [];
    return (
      <div className="global-inspection">
        <section className="inspection-placeholder">
          <span className="eyebrow">FRAME EVIDENCE</span>
          <p>从时间轴选择一帧，进入双目与骨架检查。</p>
        </section>
        {globalRecords.length > 0 && (
          <div className="global-record-inspector">
            <div className="global-record-heading">
              <span className="section-index">GLOBAL</span>
              <h2>运行级阶段记录</h2>
              <span>{globalRecords.length} RECORDS</span>
            </div>
            <InspectorTabs detail={runDetail} records={globalRecords} />
          </div>
        )}
      </div>
    );
  }

  const { frame, records } = frameDetail;
  return (
    <div className="frame-inspector">
      <div className="frame-context">
        <div>
          <span className="section-index">FRAME</span>
          <strong>{frame.frame_index === null ? "#—" : `#${String(frame.frame_index).padStart(6, "0")}`}</strong>
          <span>{frame.frame_id}</span>
        </div>
        <div><Clock3 aria-hidden="true" /> {frame.timestamp_ns === null ? "timestamp —" : `${frame.timestamp_ns.toLocaleString()} ns`}</div>
        <div className="context-chips">
          {frame.track_ids.map((track) => <span key={track}>{track}</span>)}
          {frame.view_ids.map((view) => <span key={view}>{view}</span>)}
        </div>
      </div>
      <section className="pipeline-comparison-card" aria-label="逐节点过程检查">
        <header className="card-header">
          <div>
            <span className="section-index">FLOW</span>
            <div><h2>逐节点前后对比</h2><p>同一帧 · 双目 · 所有手轨迹</p></div>
          </div>
        </header>
        <PipelineNodeRail
          records={records}
          selectedNodeId={selectedNodeId}
          onSelect={setSelectedNodeId}
        />
        <StageComparison
          runKey={runKey}
          records={records}
          selectedNodeId={selectedNodeId}
          selectedTrack={selectedTrack}
        />
      </section>
      <div className="evidence-grid">
        <StereoEvidence runKey={runKey} records={records} trackId={selectedTrack} />
        <SkeletonCanvas records={records} trackId={selectedTrack} />
      </div>
      <InspectorTabs detail={runDetail} records={records} />
    </div>
  );
}
