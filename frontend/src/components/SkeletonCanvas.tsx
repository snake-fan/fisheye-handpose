import { RotateCcw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { TraceRecord } from "../api/types";
import {
  best3dRecord,
  FHP21_EDGES,
  FINGER_COLORS,
  payloadOf,
  points3,
  validityValues,
} from "../domain/trace";

interface SkeletonCanvasProps {
  records: TraceRecord[];
  trackId: string;
}

interface ProjectedPoint {
  x: number;
  y: number;
  z: number;
}

export function SkeletonCanvas({ records, trackId }: SkeletonCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const [rotation, setRotation] = useState({ yaw: -0.68, pitch: 0.42 });
  const record = useMemo(() => best3dRecord(records, trackId), [records, trackId]);
  const payload = record ? payloadOf(record) : {};
  const landmarks = useMemo(() => points3(payload.landmarks_xyz_m), [payload.landmarks_xyz_m]);
  const validity = useMemo(
    () => validityValues(payload.validity, landmarks.length),
    [payload.validity, landmarks.length],
  );
  const validCount = landmarks.filter((point, index) => point && validity[index]).length;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    const cssWidth = canvas.clientWidth || 560;
    const cssHeight = Math.max(300, Math.round(cssWidth * 0.72));
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(cssWidth * ratio);
    canvas.height = Math.round(cssHeight * ratio);
    canvas.style.height = `${cssHeight}px`;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, cssWidth, cssHeight);

    const gradient = context.createLinearGradient?.(0, 0, 0, cssHeight);
    if (gradient) {
      gradient.addColorStop(0, "#0b1418");
      gradient.addColorStop(1, "#070b0e");
      context.fillStyle = gradient;
    } else {
      context.fillStyle = "#080e12";
    }
    context.fillRect(0, 0, cssWidth, cssHeight);

    const centerX = cssWidth / 2;
    const horizonY = cssHeight * 0.57;
    context.strokeStyle = "rgba(90, 119, 128, 0.18)";
    context.lineWidth = 1;
    for (let row = -4; row <= 5; row += 1) {
      const y = horizonY + row * 29;
      context.beginPath();
      context.moveTo(cssWidth * 0.08, y);
      context.lineTo(cssWidth * 0.92, y);
      context.stroke();
    }
    for (let column = -7; column <= 7; column += 1) {
      context.beginPath();
      context.moveTo(centerX + column * 20, horizonY - 115);
      context.lineTo(centerX + column * 43, cssHeight * 0.94);
      context.stroke();
    }

    const available = landmarks.filter((point, index): point is [number, number, number] => {
      return Boolean(point && validity[index]);
    });
    if (!available.length) {
      context.fillStyle = "#829097";
      context.font = "12px 'JetBrains Mono Variable', monospace";
      context.fillText("NO VALID 3D LANDMARKS", 24, 35);
      return;
    }
    const centre: [number, number, number] = [
      available.reduce((sum, point) => sum + point[0], 0) / available.length,
      available.reduce((sum, point) => sum + point[1], 0) / available.length,
      available.reduce((sum, point) => sum + point[2], 0) / available.length,
    ];
    const extent = Math.max(
      ...available.flatMap((point) => [
        Math.abs(point[0] - centre[0]),
        Math.abs(point[1] - centre[1]),
        Math.abs(point[2] - centre[2]),
      ]),
      0.03,
    );
    const scale = Math.min(cssWidth, cssHeight) * 0.34 / extent;
    const cosY = Math.cos(rotation.yaw);
    const sinY = Math.sin(rotation.yaw);
    const cosP = Math.cos(rotation.pitch);
    const sinP = Math.sin(rotation.pitch);
    const projected: Array<ProjectedPoint | null> = landmarks.map((point, index) => {
      if (!point || !validity[index]) return null;
      const x = point[0] - centre[0];
      const y = point[1] - centre[1];
      const z = point[2] - centre[2];
      const rx = cosY * x + sinY * z;
      const rz = -sinY * x + cosY * z;
      const ry = cosP * y - sinP * rz;
      const depth = sinP * y + cosP * rz;
      const perspective = 1 / (1 + depth * 1.5);
      return {
        x: centerX + rx * scale * perspective,
        y: cssHeight * 0.49 - ry * scale * perspective,
        z: depth,
      };
    });

    FHP21_EDGES.forEach(([from, to], edgeIndex) => {
      const start = projected[from];
      const end = projected[to];
      if (!start || !end) return;
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.lineCap = "round";
      context.lineWidth = 3.2;
      context.strokeStyle = FINGER_COLORS[Math.floor(edgeIndex / 4)];
      context.globalAlpha = 0.88;
      context.stroke();
    });
    projected
      .map((point, index) => ({ point, index }))
      .filter(({ point }) => point)
      .sort((a, b) => (a.point?.z ?? 0) - (b.point?.z ?? 0))
      .forEach(({ point, index }) => {
        if (!point) return;
        context.beginPath();
        context.arc(point.x, point.y, index === 0 ? 5.6 : 4.2, 0, Math.PI * 2);
        context.fillStyle = index === 0 ? "#ffffff" : FINGER_COLORS[Math.max(0, Math.ceil(index / 4) - 1)];
        context.globalAlpha = 1;
        context.fill();
        context.strokeStyle = "rgba(5, 10, 12, 0.9)";
        context.lineWidth = 1.5;
        context.stroke();
      });
    context.globalAlpha = 1;
  }, [landmarks, rotation, validity]);

  useEffect(() => {
    draw();
    window.addEventListener("resize", draw);
    return () => window.removeEventListener("resize", draw);
  }, [draw]);

  return (
    <section className="evidence-card skeleton-card">
      <header className="card-header">
        <div>
          <span className="section-index">04</span>
          <div><h2>FHP21 · 3D</h2><p>拖动旋转 · Rig coordinate frame</p></div>
        </div>
        <button
          type="button"
          className="canvas-reset"
          aria-label="重置三维视角"
          onClick={() => setRotation({ yaw: -0.68, pitch: 0.42 })}
        ><RotateCcw /></button>
      </header>
      <div className="canvas-wrap">
        <canvas
          ref={canvasRef}
          role="img"
          aria-label="FHP21 三维骨架"
          onPointerDown={(event) => {
            dragRef.current = { x: event.clientX, y: event.clientY };
            event.currentTarget.setPointerCapture?.(event.pointerId);
          }}
          onPointerMove={(event) => {
            if (!dragRef.current) return;
            const dx = event.clientX - dragRef.current.x;
            const dy = event.clientY - dragRef.current.y;
            dragRef.current = { x: event.clientX, y: event.clientY };
            setRotation((value) => ({
              yaw: value.yaw + dx * 0.009,
              pitch: Math.max(-1.25, Math.min(1.25, value.pitch + dy * 0.009)),
            }));
          }}
          onPointerUp={() => { dragRef.current = null; }}
          onPointerCancel={() => { dragRef.current = null; }}
        />
        <div className="axis-widget" aria-hidden="true"><span>X</span><span>Y</span><span>Z</span></div>
        {!record && <div className="canvas-empty">此帧没有 21 点三维记录</div>}
      </div>
      <footer className="skeleton-stats">
        <div><span>STAGE</span><strong>{record?.stage ?? "—"}</strong></div>
        <div>
          <span>VALID</span>
          <strong>{validCount} / 21</strong>
          {validCount < 21 && <small className="landmark-gap">{21 - validCount} missing</small>}
        </div>
        <div><span>FRAME</span><strong>{String(payload.coordinate_frame ?? "—").toUpperCase()}</strong></div>
        <div><span>REPROJ.</span><strong>{typeof payload.mean_reprojection_error_px === "number" ? `${payload.mean_reprojection_error_px.toFixed(2)} px` : "—"}</strong></div>
      </footer>
    </section>
  );
}
