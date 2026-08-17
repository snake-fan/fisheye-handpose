import type { TraceRecord } from "../api/types";

const TRACE_STAGE_ORDER = [
  "SYSTEM",
  "DISCOVERY",
  "CALIBRATION",
  "DECODE",
  "SYNCHRONIZATION",
  "RECTIFICATION",
  "DETECTION",
  "POSE_2D",
  "CROSS_VIEW_ASSOCIATION",
  "RAW_FUSION",
  "KINEMATIC_REFINEMENT",
  "TEMPORAL_REFINEMENT",
  "QA",
  "EXPORT",
] as const;

const TRACE_STAGE_RANK = new Map<string, number>(
  TRACE_STAGE_ORDER.map((stage, index) => [stage, index]),
);

interface StageVocabularySource {
  stages: readonly string[];
  run: { stage_counts?: Record<string, number> };
  global_records?: readonly TraceRecord[];
}

export function frameFilterStages(detail: StageVocabularySource): string[] {
  const globalCounts = new Map<string, number>();
  for (const record of detail.global_records ?? []) {
    if (!record.stage) continue;
    globalCounts.set(record.stage, (globalCounts.get(record.stage) ?? 0) + 1);
  }

  const stages = detail.stages.filter((stage) => {
    const total = detail.run.stage_counts?.[stage];
    if (typeof total !== "number" || !Number.isFinite(total)) return true;
    return total > (globalCounts.get(stage) ?? 0);
  });

  return [...new Set(stages)].sort((left, right) => {
    const fallback = TRACE_STAGE_ORDER.length;
    return (TRACE_STAGE_RANK.get(left) ?? fallback) - (TRACE_STAGE_RANK.get(right) ?? fallback)
      || left.localeCompare(right);
  });
}
