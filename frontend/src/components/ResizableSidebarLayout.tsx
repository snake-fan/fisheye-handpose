import {
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

const DEFAULT_SIDEBAR_WIDTH = 292;
const MINIMUM_SIDEBAR_WIDTH = 240;
const MAXIMUM_SIDEBAR_WIDTH = 520;
const MINIMUM_MAIN_WIDTH = 360;
const RESIZER_WIDTH = 9;
const MOBILE_BREAKPOINT = 760;
const KEYBOARD_STEP = 16;
const STORAGE_KEY = "fisheye-handpose.sidebar-width.v1";

type SidebarStyle = CSSProperties & { "--sidebar-width": string };

function maximumWidth(viewportWidth: number): number {
  if (viewportWidth <= MOBILE_BREAKPOINT) return MAXIMUM_SIDEBAR_WIDTH;
  return Math.max(
    MINIMUM_SIDEBAR_WIDTH,
    Math.min(
      MAXIMUM_SIDEBAR_WIDTH,
      viewportWidth - MINIMUM_MAIN_WIDTH - RESIZER_WIDTH,
    ),
  );
}

function clampWidth(value: number, viewportWidth: number): number {
  return Math.round(Math.max(
    MINIMUM_SIDEBAR_WIDTH,
    Math.min(maximumWidth(viewportWidth), value),
  ));
}

function storedWidth(viewportWidth: number): number {
  try {
    if (typeof window === "undefined") return clampWidth(DEFAULT_SIDEBAR_WIDTH, viewportWidth);
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === null) return clampWidth(DEFAULT_SIDEBAR_WIDTH, viewportWidth);
    const value = Number(stored);
    if (Number.isFinite(value)) return clampWidth(value, viewportWidth);
  } catch {
    // Storage can be unavailable in privacy-restricted browser contexts.
  }
  return clampWidth(DEFAULT_SIDEBAR_WIDTH, viewportWidth);
}

function persistWidth(value: number): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, String(value));
  } catch {
    // Resizing remains available even when persistence is blocked.
  }
}

export function ResizableSidebarLayout({
  sidebar,
  sidebarId,
  children,
}: {
  sidebar: ReactNode;
  sidebarId: string;
  children: ReactNode;
}) {
  const initialViewportWidth = typeof window === "undefined" ? 1024 : window.innerWidth;
  const [viewportWidth, setViewportWidth] = useState(initialViewportWidth);
  const [sidebarWidth, setSidebarWidth] = useState(() => storedWidth(initialViewportWidth));
  const [resizing, setResizing] = useState(false);
  const shellRef = useRef<HTMLDivElement>(null);
  const widthRef = useRef(sidebarWidth);
  const viewportWidthRef = useRef(viewportWidth);
  const resizingRef = useRef(false);
  const activePointerIdRef = useRef<number | null>(null);
  const dragOffsetRef = useRef(0);

  const applyWidth = useCallback((value: number) => {
    const next = clampWidth(value, viewportWidthRef.current);
    widthRef.current = next;
    setSidebarWidth(next);
    return next;
  }, []);

  const resetWidth = useCallback(() => {
    persistWidth(applyWidth(DEFAULT_SIDEBAR_WIDTH));
  }, [applyWidth]);

  const finishResize = useCallback((handle?: HTMLDivElement, pointerId?: number) => {
    if (!resizingRef.current) return;
    resizingRef.current = false;
    activePointerIdRef.current = null;
    setResizing(false);
    if (
      handle
      && pointerId !== undefined
      && typeof handle.hasPointerCapture === "function"
      && handle.hasPointerCapture(pointerId)
    ) {
      handle.releasePointerCapture(pointerId);
    }
    persistWidth(widthRef.current);
  }, []);

  useEffect(() => {
    const handleResize = () => {
      viewportWidthRef.current = window.innerWidth;
      setViewportWidth(window.innerWidth);
      if (window.innerWidth > MOBILE_BREAKPOINT) applyWidth(widthRef.current);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [applyWidth]);

  useEffect(() => () => {
    resizingRef.current = false;
    activePointerIdRef.current = null;
  }, []);

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || event.isPrimary === false || resizingRef.current) return;
    event.preventDefault();
    const shellLeft = shellRef.current?.getBoundingClientRect().left ?? 0;
    dragOffsetRef.current = event.clientX - shellLeft - widthRef.current;
    activePointerIdRef.current = event.pointerId;
    resizingRef.current = true;
    setResizing(true);
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!resizingRef.current || activePointerIdRef.current !== event.pointerId) return;
    const shellLeft = shellRef.current?.getBoundingClientRect().left ?? 0;
    applyWidth(event.clientX - shellLeft - dragOffsetRef.current);
  };

  const onPointerEnd = (event: PointerEvent<HTMLDivElement>) => {
    if (activePointerIdRef.current !== event.pointerId) return;
    finishResize(event.currentTarget, event.pointerId);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      resetWidth();
      return;
    }
    let next: number | undefined;
    if (event.key === "ArrowLeft") next = widthRef.current - KEYBOARD_STEP;
    if (event.key === "ArrowRight") next = widthRef.current + KEYBOARD_STEP;
    if (event.key === "Home") next = MINIMUM_SIDEBAR_WIDTH;
    if (event.key === "End") next = maximumWidth(viewportWidthRef.current);
    if (next === undefined) return;
    event.preventDefault();
    persistWidth(applyWidth(next));
  };

  const style: SidebarStyle = { "--sidebar-width": `${sidebarWidth}px` };
  const maximum = maximumWidth(viewportWidth);

  return (
    <div
      ref={shellRef}
      className={`app-shell ${resizing ? "sidebar-resizing" : ""}`}
      style={style}
      data-sidebar-width={sidebarWidth}
    >
      {sidebar}
      <div
        className="sidebar-resizer"
        role="separator"
        aria-label="调整侧边栏宽度"
        aria-description="拖动或使用方向键调整；按 Enter 或空格恢复默认宽度"
        aria-keyshortcuts="Enter Space"
        aria-controls={sidebarId}
        aria-orientation="vertical"
        aria-valuemin={MINIMUM_SIDEBAR_WIDTH}
        aria-valuemax={maximum}
        aria-valuenow={sidebarWidth}
        tabIndex={0}
        title="拖动或使用方向键调整；按 Enter、空格或双击恢复默认"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerEnd}
        onPointerCancel={onPointerEnd}
        onLostPointerCapture={onPointerEnd}
        onDoubleClick={resetWidth}
        onKeyDown={onKeyDown}
      />
      {children}
    </div>
  );
}
