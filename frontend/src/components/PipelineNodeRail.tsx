import type { TraceRecord } from "../api/types";
import {
  createFrameEvidence,
  type FrameEvidence,
  type FrameEvidenceRecord,
} from "../domain/frameEvidence";

export type PipelineNodeId =
  | "SOURCE_RGB"
  | "FISHEYE_UNDISTORTION"
  | "STEREO_RECTIFICATION"
  | "HAND_DETECTION"
  | "HAND_POSE_2D"
  | "CROSS_VIEW_ASSOCIATION"
  | "STEREO_TRIANGULATION_RAW_3D"
  | "MANO_FRAMEWISE"
  | "TEMPORAL_REFINEMENT"
  | "STABLE_FHP21_EXPORT";

interface PipelineNodeDefinition {
  id: PipelineNodeId;
  label: string;
  stage?: string;
  roles?: readonly string[];
  missing: string;
}

export const PIPELINE_NODES: readonly PipelineNodeDefinition[] = [
  { id: "SOURCE_RGB", label: "Stereo Fisheye RGB", roles: ["source_left", "source_right"], missing: "此帧未保存双目源图像" },
  { id: "FISHEYE_UNDISTORTION", label: "OpenCV Fisheye Undistortion · DEBUG_ONLY", roles: ["undistorted_left", "undistorted_right"], missing: "此帧未产生去畸变图像" },
  { id: "STEREO_RECTIFICATION", label: "Stereo Rectification · DEBUG_ONLY", roles: ["rectified_left", "rectified_right"], missing: "此帧未产生双目校正图像" },
  { id: "HAND_DETECTION", label: "Hand Detection · NATIVE INPUT", stage: "DETECTION", missing: "此帧未产生手部候选" },
  { id: "HAND_POSE_2D", label: "RTMPose-m Hand5", stage: "POSE_2D", missing: "此帧未产生 2D 关键点" },
  { id: "CROSS_VIEW_ASSOCIATION", label: "Cross-view Association", stage: "CROSS_VIEW_ASSOCIATION", missing: "此帧未产生跨视角匹配" },
  { id: "STEREO_TRIANGULATION_RAW_3D", label: "Stereo Triangulation · Raw 3D", stage: "RAW_FUSION", missing: "此帧未产生 Raw Metric 3D" },
  { id: "MANO_FRAMEWISE", label: "MANO v1.2 Fitting", stage: "KINEMATIC_REFINEMENT", missing: "此帧未产生 MANO 拟合" },
  { id: "TEMPORAL_REFINEMENT", label: "Temporal Refinement", stage: "TEMPORAL_REFINEMENT", missing: "此帧未产生时序结果" },
  { id: "STABLE_FHP21_EXPORT", label: "Stable Metric FHP21", stage: "EXPORT", missing: "此帧未产生最终骨架" },
] as const;

interface PipelineNodeRailProps {
  evidence?: FrameEvidence;
  records: TraceRecord[];
  selectedNodeId: PipelineNodeId;
  onSelect: (nodeId: PipelineNodeId) => void;
}

type PipelineNodeStatus = "PRODUCED" | "PARTIAL" | "NOT_PRODUCED";

function outputState(record: FrameEvidenceRecord): Exclude<PipelineNodeStatus, "PARTIAL"> | null {
  return record.outputStatus === "UNKNOWN" ? null : record.outputStatus;
}

function nodeState(definition: PipelineNodeDefinition, evidence: FrameEvidence) {
  if (definition.roles) {
    const produced = definition.roles.every((role) => evidence.hasArtifactRole(role));
    return {
      status: (produced ? "PRODUCED" : "NOT_PRODUCED") as PipelineNodeStatus,
      reason: produced ? "" : definition.missing,
    };
  }
  const matching = evidence.recordsForStage(definition.stage ?? "");
  const produced = matching.filter((record) => outputState(record) === "PRODUCED");
  const notProduced = matching.filter((record) => outputState(record) === "NOT_PRODUCED");
  const status: PipelineNodeStatus = produced.length
    ? notProduced.length ? "PARTIAL" : "PRODUCED"
    : "NOT_PRODUCED";
  const reason = notProduced.map((record) => record.failureReason).find(Boolean)
    ?? (status === "PARTIAL" ? "部分手未产出" : definition.missing);
  return { status, reason: status === "PRODUCED" ? "" : reason };
}

export function PipelineNodeRail({ evidence, records, selectedNodeId, onSelect }: PipelineNodeRailProps) {
  const frameEvidence = evidence ?? createFrameEvidence(records);
  return (
    <nav className="pipeline-node-rail" aria-label="Pipeline 节点">
      {PIPELINE_NODES.map((definition, index) => {
        const state = nodeState(definition, frameEvidence);
        const statusClass = state.status.toLowerCase().replace("_", "-");
        return (
          <button
            key={definition.id}
            type="button"
            data-node-id={definition.id}
            className={`${statusClass} ${selectedNodeId === definition.id ? "selected" : ""}`}
            aria-pressed={selectedNodeId === definition.id}
            onClick={() => onSelect(definition.id)}
          >
            <span className="pipeline-node-index">{String(index + 1).padStart(2, "0")}</span>
            <strong>{definition.label}</strong>
            <span className="pipeline-node-status">{state.status}</span>
            {state.status !== "PRODUCED" && <small>{state.reason}</small>}
          </button>
        );
      })}
    </nav>
  );
}
