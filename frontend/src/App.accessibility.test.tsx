import { render, screen } from "@testing-library/react";
import axe from "axe-core";
import { expect, test, vi } from "vitest";

import { App } from "./App";

test("the run catalog has no automatically detectable accessibility violations", async () => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({
    items: [{
      run_key: "item-a--run-a",
      item_id: "item-a",
      run_id: "run-a",
      status: "COMPLETED",
      created_at_utc: "2026-08-13T04:02:53Z",
      finalized_at_utc: "2026-08-13T04:04:31Z",
      pipeline_version: "0.1.0",
      record_count: 24,
      frame_count: 4,
      stage_counts: { POSE_2D: 4 },
      warning_count: 0,
      failure_count: 0,
    }],
    offset: 0,
    limit: 50,
    total: 1,
  }), { headers: { "Content-Type": "application/json" } }));

  const { container } = render(<App />);
  await screen.findByRole("button", { name: /item-a.*run-a/i });

  const result = await axe.run(container, {
    rules: { "color-contrast": { enabled: false } },
  });
  expect(result.violations).toEqual([]);
});
