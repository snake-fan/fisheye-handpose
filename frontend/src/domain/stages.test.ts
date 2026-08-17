import { expect, test } from "vitest";

import { frameFilterStages } from "./stages";

test("legacy run details expose only frame-backed stages in pipeline order", () => {
  const stages = frameFilterStages({
    stages: [
      "CALIBRATION",
      "CROSS_VIEW_ASSOCIATION",
      "DECODE",
      "DETECTION",
      "DISCOVERY",
      "EXPORT",
      "KINEMATIC_REFINEMENT",
      "POSE_2D",
      "QA",
      "RAW_FUSION",
      "RECTIFICATION",
      "SYNCHRONIZATION",
      "SYSTEM",
      "TEMPORAL_REFINEMENT",
    ],
    run: {
      stage_counts: {
        SYSTEM: 2,
        DISCOVERY: 1,
        CALIBRATION: 1,
        DECODE: 2,
        SYNCHRONIZATION: 123,
        RECTIFICATION: 122,
        DETECTION: 240,
        POSE_2D: 1216,
        CROSS_VIEW_ASSOCIATION: 240,
        RAW_FUSION: 240,
        KINEMATIC_REFINEMENT: 241,
        TEMPORAL_REFINEMENT: 240,
        QA: 2,
        EXPORT: 241,
      },
    },
    global_records: [
      ...Array.from({ length: 2 }, () => ({ stage: "SYSTEM" })),
      { stage: "DISCOVERY" },
      { stage: "CALIBRATION" },
      ...Array.from({ length: 2 }, () => ({ stage: "DECODE" })),
      ...Array.from({ length: 3 }, () => ({ stage: "SYNCHRONIZATION" })),
      ...Array.from({ length: 2 }, () => ({ stage: "RECTIFICATION" })),
      { stage: "KINEMATIC_REFINEMENT" },
      ...Array.from({ length: 2 }, () => ({ stage: "QA" })),
      { stage: "EXPORT" },
    ],
  });

  expect(stages).toEqual([
    "SYNCHRONIZATION",
    "RECTIFICATION",
    "DETECTION",
    "POSE_2D",
    "CROSS_VIEW_ASSOCIATION",
    "RAW_FUSION",
    "KINEMATIC_REFINEMENT",
    "TEMPORAL_REFINEMENT",
    "EXPORT",
  ]);
});
