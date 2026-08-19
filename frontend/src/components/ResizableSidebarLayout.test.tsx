import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, test } from "vitest";

import { ResizableSidebarLayout } from "./ResizableSidebarLayout";

const STORAGE_KEY = "fisheye-handpose.sidebar-width.v1";
const originalInnerWidth = window.innerWidth;

function renderLayout() {
  render(
    <ResizableSidebarLayout
      sidebarId="test-sidebar-pane"
      sidebar={<aside id="test-sidebar-pane">运行列表</aside>}
    >
      <main>检查器</main>
    </ResizableSidebarLayout>,
  );
  const separator = screen.getByRole("separator", { name: "调整侧边栏宽度" });
  const shell = separator.parentElement as HTMLDivElement;
  return { separator, shell };
}

function dispatchPointer(
  target: Element,
  type: "pointerdown" | "pointermove" | "pointerup" | "pointercancel",
  {
    button = 0,
    clientX,
    isPrimary = true,
    pointerId,
  }: { button?: number; clientX: number; isPrimary?: boolean; pointerId: number },
) {
  const event = new Event(type, { bubbles: true, cancelable: true });
  Object.defineProperties(event, {
    button: { value: button },
    clientX: { value: clientX },
    isPrimary: { value: isPrimary },
    pointerId: { value: pointerId },
  });
  fireEvent(target, event);
}

beforeEach(() => {
  window.localStorage.clear();
  Object.defineProperty(window, "innerWidth", {
    configurable: true,
    value: 1200,
  });
});

afterEach(() => {
  window.localStorage.clear();
  Object.defineProperty(window, "innerWidth", {
    configurable: true,
    value: originalInnerWidth,
  });
});

test("uses the intended default width when no saved preference exists", () => {
  const { separator, shell } = renderLayout();

  expect(shell).toHaveAttribute("data-sidebar-width", "292");
  expect(shell).toHaveStyle({ "--sidebar-width": "292px" });
  expect(separator).toHaveAttribute("aria-valuemin", "240");
  expect(separator).toHaveAttribute("aria-valuemax", "520");
  expect(separator).toHaveAttribute("aria-valuenow", "292");
  expect(separator).toHaveAttribute("aria-controls", "test-sidebar-pane");
  expect(separator).toHaveAttribute("aria-keyshortcuts", "Enter Space");
  expect(separator).toHaveAttribute(
    "aria-description",
    "拖动或使用方向键调整；按 Enter 或空格恢复默认宽度",
  );
});

test("restores a saved width and supports bounded keyboard resizing", () => {
  window.localStorage.setItem(STORAGE_KEY, "410");
  const { separator, shell } = renderLayout();

  expect(shell).toHaveAttribute("data-sidebar-width", "410");

  fireEvent.keyDown(separator, { key: "ArrowRight" });
  expect(shell).toHaveAttribute("data-sidebar-width", "426");
  expect(window.localStorage.getItem(STORAGE_KEY)).toBe("426");

  fireEvent.keyDown(separator, { key: "Home" });
  expect(shell).toHaveAttribute("data-sidebar-width", "240");

  fireEvent.keyDown(separator, { key: "End" });
  expect(shell).toHaveAttribute("data-sidebar-width", "520");
});

test("Enter, Space, and double click restore the default width", () => {
  window.localStorage.setItem(STORAGE_KEY, "410");
  const { separator, shell } = renderLayout();

  fireEvent.keyDown(separator, { key: "Enter" });
  expect(shell).toHaveAttribute("data-sidebar-width", "292");
  expect(window.localStorage.getItem(STORAGE_KEY)).toBe("292");

  fireEvent.keyDown(separator, { key: "ArrowRight" });
  expect(shell).toHaveAttribute("data-sidebar-width", "308");
  expect(fireEvent.keyDown(separator, { key: " " })).toBe(false);
  expect(shell).toHaveAttribute("data-sidebar-width", "292");
  expect(window.localStorage.getItem(STORAGE_KEY)).toBe("292");

  fireEvent.keyDown(separator, { key: "ArrowRight" });
  fireEvent.doubleClick(separator);
  expect(shell).toHaveAttribute("data-sidebar-width", "292");
  expect(window.localStorage.getItem(STORAGE_KEY)).toBe("292");
});

test("drags from the grab point without jumping and persists on release", () => {
  const { separator, shell } = renderLayout();
  shell.getBoundingClientRect = () => ({
    bottom: 900,
    height: 900,
    left: 100,
    right: 1300,
    top: 0,
    width: 1200,
    x: 100,
    y: 0,
    toJSON: () => ({}),
  });

  // The pointer starts four pixels inside the nine-pixel grab handle.
  dispatchPointer(separator, "pointerdown", { clientX: 396, pointerId: 7 });
  expect(shell).toHaveClass("sidebar-resizing");

  dispatchPointer(separator, "pointermove", { clientX: 524, pointerId: 7 });
  expect(shell).toHaveAttribute("data-sidebar-width", "420");

  dispatchPointer(separator, "pointerup", { clientX: 524, pointerId: 7 });
  expect(shell).not.toHaveClass("sidebar-resizing");
  expect(window.localStorage.getItem(STORAGE_KEY)).toBe("420");
});

test("ignores competing pointers and safely finishes a cancelled drag", () => {
  const { separator, shell } = renderLayout();

  dispatchPointer(separator, "pointerdown", { clientX: 296, pointerId: 7 });
  dispatchPointer(separator, "pointerdown", {
    clientX: 296,
    isPrimary: false,
    pointerId: 8,
  });
  dispatchPointer(separator, "pointermove", { clientX: 500, pointerId: 8 });
  dispatchPointer(separator, "pointerup", { clientX: 500, pointerId: 8 });
  expect(shell).toHaveClass("sidebar-resizing");
  expect(shell).toHaveAttribute("data-sidebar-width", "292");

  dispatchPointer(separator, "pointermove", { clientX: 404, pointerId: 7 });
  expect(shell).toHaveAttribute("data-sidebar-width", "400");
  dispatchPointer(separator, "pointercancel", { clientX: 404, pointerId: 7 });
  expect(shell).not.toHaveClass("sidebar-resizing");
  expect(window.localStorage.getItem(STORAGE_KEY)).toBe("400");

  dispatchPointer(separator, "pointermove", { clientX: 500, pointerId: 7 });
  expect(shell).toHaveAttribute("data-sidebar-width", "400");
});

test("keeps enough room for the main panel when the desktop narrows", () => {
  window.localStorage.setItem(STORAGE_KEY, "500");
  const { separator, shell } = renderLayout();

  expect(separator).toHaveAttribute("aria-valuemax", "520");
  expect(shell).toHaveAttribute("data-sidebar-width", "500");

  Object.defineProperty(window, "innerWidth", {
    configurable: true,
    value: 800,
  });
  fireEvent.resize(window);

  expect(separator).toHaveAttribute("aria-valuemax", "431");
  expect(shell).toHaveAttribute("data-sidebar-width", "431");

  fireEvent.keyDown(separator, { key: "ArrowRight" });
  expect(shell).toHaveAttribute("data-sidebar-width", "431");
});
