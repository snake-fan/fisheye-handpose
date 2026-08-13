"use strict";

const state = {
  offset: 0,
  limit: 80,
  total: 0,
  frames: [],
  selectedFrameId: null,
  stage: "",
  trackId: "",
};

const artifactRoles = new Set([
  "source_left",
  "source_right",
  "rectified_left",
  "rectified_right",
  "crop",
  "overlay",
]);

const skeletonEdges = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
];

const fingerColors = ["#54e3c2", "#65a8ff", "#f7df67", "#ff9e64", "#df82ff"];

function byId(id) {
  return document.getElementById(id);
}

async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload?.error?.message || `HTTP ${response.status}`);
  }
  return payload;
}

function stringify(value) {
  return JSON.stringify(value, null, 2);
}

function text(value, fallback = "—") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function payloadOf(record) {
  return record && typeof record.payload === "object" && record.payload !== null
    ? record.payload
    : record;
}

function recordId(record) {
  return record.record_id || record.id || "unknown-record";
}

function artifactRefs(record) {
  const unique = new Map();
  const candidates = [];
  function collect(value, inheritedRole = null, seen = new Set()) {
    if (!value || typeof value !== "object" || seen.has(value)) return;
    seen.add(value);
    const explicitRole = typeof value.role === "string" ? value.role : inheritedRole;
    if (
      typeof value.sha256 === "string"
      && (typeof value.path === "string" || typeof value.relative_path === "string")
    ) {
      candidates.push(explicitRole ? { ...value, role: explicitRole } : value);
    }
    for (const [key, child] of Object.entries(value)) {
      collect(child, artifactRoles.has(key) ? key : explicitRole, seen);
    }
  }
  collect(record);
  for (const reference of candidates) {
    if (!/^[a-f0-9]{64}$/i.test(reference.sha256)) continue;
    const key = `${reference.sha256}:${reference.role || "artifact"}`;
    unique.set(key, reference);
  }
  return [...unique.values()];
}

function metric(label, value) {
  const dl = document.createElement("dl");
  dl.className = "metric";
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = label;
  dd.textContent = text(value);
  dd.title = dd.textContent;
  dl.append(dt, dd);
  return dl;
}

function renderRun(run) {
  const manifest = run.manifest || {};
  const validation = run.validation || {};
  const summary = run.summary || {};
  const businessSummary = summary.summary || {};
  const summaryElement = byId("run-summary");
  summaryElement.replaceChildren(
    metric("Run ID", manifest.run_id),
    metric("Schema", manifest.schema_version),
    metric("Created", manifest.created_at || manifest.created_at_utc),
    metric("Records", validation.record_count ?? summary.record_count),
    metric("Frames", businessSummary.frame_count ?? businessSummary.frames),
    metric("Calibration", manifest.calibration_id),
  );

  const badge = byId("validation-badge");
  const errors = validation.errors || [];
  const warnings = validation.warnings || [];
  const ok = validation.ok === true || validation.valid === true;
  const runStatus = validation.status || "UNKNOWN";
  badge.className = `status ${ok ? "ok" : "error"}`;
  badge.textContent = ok
    ? `${runStatus} · VALID${warnings.length ? ` · ${warnings.length} warnings` : ""}`
    : `${runStatus} · INVALID · ${errors.length} errors`;

  const stages = businessSummary.stages || businessSummary.stage_counts || {};
  if ((Array.isArray(stages) && stages.length) || Object.keys(stages).length) {
    renderStageSummary(stages);
  }
}

function renderStageSummary(stages) {
  const container = byId("stage-summary");
  const datalist = byId("stage-options");
  container.replaceChildren();
  datalist.replaceChildren();
  const entries = Array.isArray(stages)
    ? stages.map((stage) => [stage.stage || stage.name, stage.count])
    : Object.entries(stages);
  if (!entries.length) {
    container.className = "stage-summary muted";
    container.textContent = "阶段会在加载帧后列出";
    return;
  }
  container.className = "stage-summary";
  for (const [name, count] of entries) {
    const row = document.createElement("div");
    row.className = "stage-row";
    const label = document.createElement("span");
    const value = document.createElement("span");
    label.textContent = text(name);
    value.textContent = text(count, "");
    row.append(label, value);
    container.append(row);
    const option = document.createElement("option");
    option.value = name;
    datalist.append(option);
  }
}

function frameQuery() {
  const params = new URLSearchParams({
    offset: String(state.offset),
    limit: String(state.limit),
  });
  if (state.stage) params.set("stage", state.stage);
  if (state.trackId) params.set("track_id", state.trackId);
  return params;
}

async function loadFrames() {
  const payload = await fetchJson(`/api/frames?${frameQuery()}`);
  state.frames = payload.items || [];
  state.total = payload.total || 0;
  renderFrames();
  renderGlobalRecords(payload.global_record_ids || []);
  const names = new Set(state.frames.flatMap((frame) => frame.stages || []));
  if (names.size) renderStageSummary(Object.fromEntries([...names].sort().map((name) => [name, ""])));
}

function renderFrames() {
  const list = byId("frame-list");
  list.replaceChildren();
  for (const frame of state.frames) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `frame-button${frame.frame_id === state.selectedFrameId ? " active" : ""}`;
    const id = document.createElement("strong");
    id.textContent = frame.frame_id;
    const time = document.createElement("span");
    time.textContent = frame.timestamp_ns === null || frame.timestamp_ns === undefined
      ? "timestamp —"
      : `${frame.timestamp_ns} ns`;
    const stages = document.createElement("span");
    stages.textContent = (frame.stages || []).join(" · ") || "no stage";
    button.append(id, time, stages);
    button.addEventListener("click", () => selectFrame(frame.frame_id));
    list.append(button);
  }
  if (!state.frames.length) {
    const empty = document.createElement("div");
    empty.className = "muted";
    empty.textContent = "过滤条件下没有帧记录";
    list.append(empty);
  }
  const first = state.total ? state.offset + 1 : 0;
  const last = Math.min(state.offset + state.frames.length, state.total);
  byId("page-status").textContent = `${first}–${last} / ${state.total}`;
  byId("previous-page").disabled = state.offset === 0;
  byId("next-page").disabled = state.offset + state.limit >= state.total;
}

async function renderGlobalRecords(recordIds) {
  const list = byId("global-record-list");
  byId("global-count").textContent = String(recordIds.length);
  list.replaceChildren();
  if (!recordIds.length) {
    list.className = "global-records muted";
    list.textContent = "暂无无帧记录";
    return;
  }
  list.className = "global-records";
  for (const id of recordIds) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "global-record-button";
    button.textContent = id;
    button.title = id;
    button.addEventListener("click", async () => {
      try {
        const record = await fetchJson(`/api/records/${encodeURIComponent(id)}`);
        renderRecords([record], `全局记录 · ${id}`);
      } catch (error) {
        renderError(error);
      }
    });
    list.append(button);
  }
}

async function selectFrame(frameId) {
  state.selectedFrameId = frameId;
  renderFrames();
  try {
    const payload = await fetchJson(`/api/frames/${encodeURIComponent(frameId)}`);
    const records = payload.records || [];
    renderRecords(filterFrameRecords(records), frameId, records);
  } catch (error) {
    renderError(error);
  }
}

function filterFrameRecords(records) {
  return records.filter((record) => {
    const payload = payloadOf(record) || {};
    if (state.stage && record.stage !== state.stage) return false;
    if (state.trackId && payload.track_id !== state.trackId) return false;
    return true;
  });
}

function renderRecords(records, title, contextRecords = records) {
  byId("selected-frame-title").textContent = title;
  byId("raw-json").textContent = stringify(records);
  const list = byId("record-list");
  list.className = "record-list";
  list.replaceChildren();
  const contextualArtifacts = new Map();
  for (const record of contextRecords) {
    const payload = payloadOf(record) || {};
    const references = artifactRefs(record);
    if (!references.length || typeof payload.view_id !== "string") continue;
    contextualArtifacts.set(
      payload.view_id,
      references.map((reference) => {
        if (reference.role) return reference;
        const role = record.stage === "DECODE" ? `source_${payload.view_id}` : "artifact";
        return { ...reference, role };
      }),
    );
  }
  for (const record of records) {
    const payload = payloadOf(record) || {};
    list.append(recordCard(record, contextualArtifacts.get(payload.view_id) || []));
  }
  if (!records.length) {
    list.className = "record-list muted";
    list.textContent = "此帧没有记录";
  }
  const with3d = [...records].reverse().find((record) => {
    const landmarks = payloadOf(record)?.landmarks_xyz_m;
    return Array.isArray(landmarks) && landmarks.length === 21;
  });
  drawSkeleton(with3d ? payloadOf(with3d) : null);
}

function recordCard(record, contextualArtifacts) {
  const card = byId("record-template").content.firstElementChild.cloneNode(true);
  const payload = payloadOf(record) || {};
  card.querySelector(".stage-name").textContent = text(record.stage || record.event, "record");
  card.querySelector(".record-status").textContent = text(record.status, "");
  const meta = card.querySelector(".record-meta");
  for (const [label, value] of [
    ["id", recordId(record)],
    ["track", payload.track_id],
    ["timestamp", payload.timestamp_ns],
  ]) {
    if (value === undefined || value === null) continue;
    const wrapper = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = label;
    dd.textContent = value;
    wrapper.append(dt, dd);
    meta.append(wrapper);
  }
  card.querySelector(".copy-record").addEventListener("click", async (event) => {
    try {
      await navigator.clipboard.writeText(stringify(record));
      event.currentTarget.textContent = "已复制";
    } catch {
      event.currentTarget.textContent = "复制失败";
    }
  });
  renderArtifacts(card.querySelector(".artifact-grid"), record, payload, contextualArtifacts);
  card.querySelector(".payload-summary").textContent = evidenceSummary(payload);
  return card;
}

function evidenceSummary(payload) {
  const parts = [];
  if (Array.isArray(payload.detections)) {
    const scores = payload.detections
      .map((detection) => detection.score)
      .filter((score) => Number.isFinite(score));
    parts.push(`${payload.detections.length} detections${scores.length ? ` · max ${Math.max(...scores).toFixed(3)}` : ""}`);
  }
  if (Array.isArray(payload.keypoints_uv)) {
    const scores = Array.isArray(payload.keypoint_scores) ? payload.keypoint_scores : [];
    const visible = scores.filter((score) => Number.isFinite(score) && score > 0).length;
    parts.push(`${payload.keypoints_uv.length} keypoints${scores.length ? ` · ${visible} scored` : ""}`);
  }
  if (Array.isArray(payload.landmarks_xyz_m)) {
    const validity = Array.isArray(payload.validity) ? payload.validity : [];
    const valid = validity.filter((value) => value === true || value === "VALID").length;
    parts.push(`${payload.landmarks_xyz_m.length} 3D landmarks${validity.length ? ` · ${valid} valid` : ""}`);
  }
  return parts.join("  |  ") || "无标准 evidence payload";
}

function renderArtifacts(container, record, payload, contextualArtifacts = []) {
  container.replaceChildren();
  let references = artifactRefs(record);
  const hasOverlayEvidence = Array.isArray(payload.detections) || Array.isArray(payload.keypoints_uv);
  if (!references.length && hasOverlayEvidence) references = contextualArtifacts;
  for (const reference of references) {
    const wrapper = document.createElement("div");
    wrapper.className = "artifact";
    const label = document.createElement("span");
    label.className = "artifact-label";
    label.textContent = reference.role || "artifact";
    wrapper.append(label);
    const mediaType = reference.media_type || reference.mime_type || "";
    const url = `/artifacts/${reference.sha256}`;
    const artifactPath = reference.path || reference.relative_path || "";
    if (mediaType.startsWith("image/") || /\.(png|jpe?g|webp|svg)$/i.test(artifactPath)) {
      const image = document.createElement("img");
      image.src = url;
      image.alt = `${reference.role || "trace"} artifact`;
      image.addEventListener("load", () => drawTwoDimensionalOverlay(wrapper, image, payload));
      wrapper.append(image);
    } else if (mediaType.startsWith("video/") || /\.(mp4|webm)$/i.test(artifactPath)) {
      const video = document.createElement("video");
      video.src = url;
      video.controls = true;
      video.preload = "metadata";
      wrapper.append(video);
    } else {
      const link = document.createElement("a");
      link.className = "artifact-link";
      link.href = url;
      link.textContent = "打开工件";
      wrapper.append(link);
    }
    container.append(wrapper);
  }
  container.hidden = references.length === 0;
}

function drawTwoDimensionalOverlay(wrapper, image, payload) {
  const detections = Array.isArray(payload.detections) ? payload.detections : [];
  const keypoints = Array.isArray(payload.keypoints_uv) ? payload.keypoints_uv : [];
  if (!detections.length && !keypoints.length) return;
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  wrapper.append(canvas);
  const context = canvas.getContext("2d");
  const unit = Math.max(1, image.naturalWidth / 720);
  context.lineWidth = 2 * unit;
  context.font = `${12 * unit}px ui-monospace`;
  for (const detection of detections) {
    const box = detection.bbox_xyxy;
    if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite)) continue;
    context.strokeStyle = "#54e3c2";
    context.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
    context.fillStyle = "#54e3c2";
    context.fillText(`${text(detection.label, "hand")} ${Number(detection.score || 0).toFixed(2)}`, box[0], Math.max(12 * unit, box[1] - 4 * unit));
  }
  const scores = Array.isArray(payload.keypoint_scores) ? payload.keypoint_scores : [];
  keypoints.forEach((point, index) => {
    if (!Array.isArray(point) || point.length < 2 || !point.slice(0, 2).every(Number.isFinite)) return;
    if (Number.isFinite(scores[index]) && scores[index] <= 0) return;
    context.beginPath();
    context.fillStyle = fingerColors[Math.min(4, Math.max(0, Math.ceil(index / 4) - 1))];
    context.arc(point[0], point[1], 3.2 * unit, 0, Math.PI * 2);
    context.fill();
  });
}

function drawSkeleton(payload) {
  const canvas = byId("skeleton-canvas");
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#070a0e";
  context.fillRect(0, 0, canvas.width, canvas.height);
  const points = payload?.landmarks_xyz_m;
  if (!Array.isArray(points) || points.length !== 21) {
    context.fillStyle = "#647080";
    context.font = "15px ui-sans-serif";
    context.textAlign = "center";
    context.fillText("选择包含 landmarks_xyz_m 的记录", canvas.width / 2, canvas.height / 2);
    byId("skeleton-meta").textContent = "无 3D 数据";
    return;
  }
  const validity = Array.isArray(payload.validity) ? payload.validity : [];
  const valid = points.map((point, index) =>
    Array.isArray(point)
      && point.length === 3
      && point.every(Number.isFinite)
      && validity[index] !== false
      && validity[index] !== "INVALID",
  );
  const finitePoints = points.filter((_, index) => valid[index]);
  if (!finitePoints.length) {
    drawSkeleton(null);
    return;
  }
  const center = [0, 1, 2].map((axis) =>
    finitePoints.reduce((sum, point) => sum + point[axis], 0) / finitePoints.length,
  );
  const yaw = -0.65;
  const pitch = 0.45;
  const projected = points.map((point) => {
    if (!Array.isArray(point) || point.length !== 3) return [0, 0, 0];
    const x = point[0] - center[0];
    const y = point[1] - center[1];
    const z = point[2] - center[2];
    const x1 = x * Math.cos(yaw) + z * Math.sin(yaw);
    const z1 = -x * Math.sin(yaw) + z * Math.cos(yaw);
    return [x1, y * Math.cos(pitch) - z1 * Math.sin(pitch), y * Math.sin(pitch) + z1 * Math.cos(pitch)];
  });
  const spread = Math.max(
    ...projected.filter((_, index) => valid[index]).flatMap((point) => [Math.abs(point[0]), Math.abs(point[1])]),
    0.001,
  );
  const scale = Math.min(canvas.width, canvas.height) * 0.4 / spread;
  const screen = projected.map((point) => [
    canvas.width / 2 + point[0] * scale,
    canvas.height / 2 - point[1] * scale,
    point[2],
  ]);
  skeletonEdges.forEach(([a, b], edgeIndex) => {
    if (!valid[a] || !valid[b]) return;
    context.beginPath();
    context.moveTo(screen[a][0], screen[a][1]);
    context.lineTo(screen[b][0], screen[b][1]);
    context.lineWidth = 5;
    context.lineCap = "round";
    context.strokeStyle = fingerColors[Math.floor(edgeIndex / 4)];
    context.stroke();
  });
  [...screen.keys()]
    .filter((index) => valid[index])
    .sort((a, b) => screen[a][2] - screen[b][2])
    .forEach((index) => {
      context.beginPath();
      context.arc(screen[index][0], screen[index][1], index === 0 ? 7 : 5, 0, Math.PI * 2);
      context.fillStyle = index === 0 ? "#ffffff" : fingerColors[Math.min(4, Math.max(0, Math.ceil(index / 4) - 1))];
      context.fill();
    });
  byId("skeleton-meta").textContent = `${valid.filter(Boolean).length}/21 valid · metres`;
}

function renderError(error) {
  const list = byId("record-list");
  list.replaceChildren();
  const message = document.createElement("div");
  message.className = "error-box";
  message.textContent = error instanceof Error ? error.message : String(error);
  list.append(message);
}

function bindControls() {
  byId("apply-filters").addEventListener("click", async () => {
    state.stage = byId("stage-filter").value.trim();
    state.trackId = byId("track-filter").value.trim();
    state.offset = 0;
    state.selectedFrameId = null;
    renderRecords([], "选择一帧");
    try {
      await loadFrames();
    } catch (error) {
      renderError(error);
    }
  });
  byId("previous-page").addEventListener("click", async () => {
    state.offset = Math.max(0, state.offset - state.limit);
    await loadFrames();
  });
  byId("next-page").addEventListener("click", async () => {
    state.offset += state.limit;
    await loadFrames();
  });
}

async function main() {
  bindControls();
  drawSkeleton(null);
  try {
    const [run] = await Promise.all([fetchJson("/api/run"), loadFrames()]);
    renderRun(run);
  } catch (error) {
    byId("validation-badge").className = "status error";
    byId("validation-badge").textContent = "无法读取 trace";
    renderError(error);
  }
}

main();
