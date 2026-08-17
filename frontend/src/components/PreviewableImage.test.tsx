import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import axe from "axe-core";
import { expect, test } from "vitest";

import { ImagePreviewProvider, PreviewableImage } from "./PreviewableImage";

test("opens a large image preview and restores focus when Escape closes it", async () => {
  const user = userEvent.setup();
  render(<PreviewableImage src="/evidence.jpg" alt="左目证据" />);

  const trigger = screen.getByRole("button", { name: "放大预览：左目证据" });
  expect(trigger).toContainElement(screen.getByRole("img", { name: "左目证据" }));

  await user.click(screen.getByRole("img", { name: "左目证据" }));

  expect(screen.getByRole("dialog", { name: "左目证据" })).toBeVisible();
  expect(screen.getByRole("img", { name: "左目证据（大图预览）" })).toHaveAttribute(
    "src",
    "/evidence.jpg",
  );
  expect(screen.getByRole("button", { name: "关闭图片预览" })).toHaveFocus();
  expect(document.body.style.overflow).toBe("hidden");

  await user.keyboard("{Escape}");

  expect(screen.queryByRole("dialog", { name: "左目证据" })).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
  expect(document.body.style.overflow).toBe("");
});

test("closes the preview when the backdrop is clicked", async () => {
  const user = userEvent.setup();
  render(<PreviewableImage src="/evidence.jpg" alt="右目证据" />);

  await user.click(screen.getByRole("button", { name: "放大预览：右目证据" }));
  await user.click(screen.getByRole("dialog", { name: "右目证据" }));

  expect(screen.queryByRole("dialog", { name: "右目证据" })).not.toBeInTheDocument();
});

test("returns to the page when the enlarged image is clicked again", async () => {
  const user = userEvent.setup();
  render(<PreviewableImage src="/evidence.jpg" alt="可返回证据" />);

  const trigger = screen.getByRole("button", { name: "放大预览：可返回证据" });
  await user.click(trigger);
  await user.click(screen.getByRole("img", { name: "可返回证据（大图预览）" }));

  expect(screen.queryByRole("dialog", { name: "可返回证据" })).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
});

test("supports native keyboard opening, traps focus, and closes from the close button", async () => {
  const user = userEvent.setup();
  render(<PreviewableImage src="/evidence.jpg" alt="键盘证据" />);

  const trigger = screen.getByRole("button", { name: "放大预览：键盘证据" });
  trigger.focus();
  await user.keyboard("{Enter}");

  const closeButton = screen.getByRole("button", { name: "关闭图片预览" });
  const returnButton = screen.getByRole("button", { name: "返回：键盘证据" });
  expect(closeButton).toHaveFocus();
  await user.tab();
  expect(returnButton).toHaveFocus();
  await user.tab();
  expect(closeButton).toHaveFocus();
  await user.tab({ shift: true });
  expect(returnButton).toHaveFocus();

  await user.click(closeButton);
  expect(screen.queryByRole("dialog", { name: "键盘证据" })).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
});

test("uses one modal for every image, makes the background inert, and preserves overlays", async () => {
  const user = userEvent.setup();
  render(
    <ImagePreviewProvider>
      <PreviewableImage src="/left.jpg" alt="左图" />
      <PreviewableImage
        src="/right.jpg"
        alt="右图"
        previewOverlay={<svg role="img" aria-label="右图骨架叠加" />}
      />
    </ImagePreviewProvider>,
  );

  const leftTrigger = screen.getByRole("button", { name: "放大预览：左图" });
  const rightTrigger = screen.getByRole("button", { name: "放大预览：右图" });
  await user.click(leftTrigger);
  expect(screen.getAllByRole("dialog")).toHaveLength(1);
  expect(screen.getByRole("dialog", { name: "左图" })).toBeVisible();
  expect(document.querySelector("[data-image-preview-root]")).toHaveAttribute("inert");

  await user.keyboard("{Escape}");
  expect(leftTrigger).toHaveFocus();
  await user.click(rightTrigger);
  expect(screen.getByRole("dialog", { name: "右图" })).toBeVisible();
  expect(screen.getByRole("img", { name: "右图骨架叠加" })).toBeVisible();
  await user.keyboard("{Escape}");
  expect(rightTrigger).toHaveFocus();
  expect(document.body.style.overflow).toBe("");
});

test("the open image preview has no automatically detectable accessibility violations", async () => {
  const user = userEvent.setup();
  render(<PreviewableImage src="/evidence.jpg" alt="可访问证据" />);
  await user.click(screen.getByRole("button", { name: "放大预览：可访问证据" }));

  const result = await axe.run(document.body, {
    rules: { "color-contrast": { enabled: false } },
  });
  expect(result.violations).toEqual([]);
});
