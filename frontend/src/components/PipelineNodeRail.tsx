import type { TraceRecord } from "../api/types";
import { artifactsOf, payloadOf } from "../domain/trace";

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
  records: TraceRecord[];
  selectedNodeId: PipelineNodeId;
  onSelect: (nodeId: PipelineNodeId) => void;
}

function outputProduced(record: TraceRecord): boolean {
  const payload = payloadOf(record);
  if (payload.output_status === "NOT_PRODUCED") return false;
  if (payload.output_status === "PRODUCED") return true;
  return record.status === "SUCCEEDED";
}

function recordReason(record: TraceRecord): string {
  const payload = payloadOf(record);
  if (typeof payload.reason === "string" && payload.reason) return payload.reason;
  const selection = payload.selection;
  if (selection && typeof selection === "object") {
    const decision = (selection as Record<string, unknown>).decision;
    if (typeof decision === "string" && decision) return decision;
  }
  return "";
}

function nodeState(definition: PipelineNodeDefinition, records: TraceRecord[]) {
  if (definition.roles) {
    const roles = new Set(records.flatMap(artifactsOf).map((artifact) => String(artifact.role ?? "")));
    const produced = definition.roles.every((role) => roles.has(role));
    return { produced, reason: produced ? "" : definition.missing };
  }
  const matching = records.filter((record) => record.stage === definition.stage);
  const produced = matching.some(outputProduced);
  const reason = matching.map(recordReason).find(Boolean) ?? definition.missing;
  return { produced, reason: produced ? "" : reason };
}

export function PipelineNodeRail({ records, selectedNodeId, onSelect }: PipelineNodeRailProps) {
  return (
    <nav className="pipeline-node-rail" aria-label="Pipeline 节点">
      {PIPELINE_NODES.map((definition, index) => {
        const state = nodeState(definition, records);
        return (
          <button
            key={definition.id}
            type="button"
            data-node-id={definition.id}
            className={`${state.produced ? "produced" : "not-produced"} ${selectedNodeId === definition.id ? "selected" : ""}`}
            aria-pressed={selectedNodeId === definition.id}
            onClick={() => onSelect(definition.id)}
          >
            <span className="pipeline-node-index">{String(index + 1).padStart(2, "0")}</span>
            <strong>{definition.label}</strong>
            <span className="pipeline-node-status">{state.produced ? "PRODUCED" : "NOT_PRODUCED"}</span>
            {!state.produced && <small>{state.reason}</small>}
          </button>
        );
      })}
    </nav>
  );
}
