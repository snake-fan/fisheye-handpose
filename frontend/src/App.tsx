import { useEffect, useState } from "react";

import { traceApi } from "./api/client";
import type { FrameDetail, FrameSummary, RunDetail, RunSummary } from "./api/types";
import { FrameInspector } from "./components/FrameInspector";
import { FrameTimeline } from "./components/FrameTimeline";
import { RunCatalog } from "./components/RunCatalog";
import { RunHeader } from "./components/RunHeader";
import { TraceFilters } from "./components/TraceFilters";
import { useUrlState } from "./hooks/useUrlState";

const FRAME_PAGE_SIZE = 120;
const RUN_PAGE_SIZE = 50;

function pageOffset(value: string): number {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : 0;
}

export function App() {
  const [urlState, setUrlState] = useUrlState();
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runTotal, setRunTotal] = useState(0);
  const [loadedRunOffset, setLoadedRunOffset] = useState(0);
  const [runLimit, setRunLimit] = useState(RUN_PAGE_SIZE);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");
  const [frames, setFrames] = useState<FrameSummary[]>([]);
  const [frameTotal, setFrameTotal] = useState(0);
  const [loadedFrameOffset, setLoadedFrameOffset] = useState(0);
  const [frameLimit, setFrameLimit] = useState(FRAME_PAGE_SIZE);
  const [framesLoading, setFramesLoading] = useState(false);
  const [framesError, setFramesError] = useState("");
  const [frameDetail, setFrameDetail] = useState<FrameDetail | null>(null);
  const [frameLoading, setFrameLoading] = useState(false);
  const [frameError, setFrameError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    traceApi
      .listRuns({
        offset: pageOffset(urlState.run_offset),
        limit: RUN_PAGE_SIZE,
        q: urlState.q,
      }, controller.signal)
      .then((page) => {
        setRuns(page.items);
        setRunTotal(page.total);
        setLoadedRunOffset(page.offset);
        setRunLimit(page.limit);
        setError("");
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        setError(reason instanceof Error ? reason.message : "无法读取运行目录");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [urlState.q, urlState.run_offset]);

  useEffect(() => {
    if (!urlState.run) {
      setDetail(null);
      setDetailError("");
      return;
    }
    const controller = new AbortController();
    setDetailLoading(true);
    setDetailError("");
    traceApi
      .getRun(urlState.run, controller.signal)
      .then(setDetail)
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        setDetail(null);
        setDetailError(reason instanceof Error ? reason.message : "无法读取运行详情");
      })
      .finally(() => {
        if (!controller.signal.aborted) setDetailLoading(false);
      });
    return () => controller.abort();
  }, [urlState.run]);

  useEffect(() => {
    if (!urlState.run) {
      setFrames([]);
      setFrameTotal(0);
      setLoadedFrameOffset(0);
      return;
    }
    const controller = new AbortController();
    setFramesLoading(true);
    setFramesError("");
    traceApi
      .listFrames(
        urlState.run,
        {
          offset: pageOffset(urlState.offset),
          limit: FRAME_PAGE_SIZE,
          stage: urlState.stage,
          track_id: urlState.track,
          status: urlState.status,
        },
        controller.signal,
      )
      .then((page) => {
        setFrames(page.items);
        setFrameTotal(page.total);
        setLoadedFrameOffset(page.offset);
        setFrameLimit(page.limit);
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        setFrames([]);
        setFrameTotal(0);
        setFramesError(reason instanceof Error ? reason.message : "无法读取帧索引");
      })
      .finally(() => {
        if (!controller.signal.aborted) setFramesLoading(false);
      });
    return () => controller.abort();
  }, [urlState.offset, urlState.run, urlState.stage, urlState.status, urlState.track]);

  useEffect(() => {
    if (!urlState.run || !urlState.frame) {
      setFrameDetail(null);
      setFrameError("");
      return;
    }
    const controller = new AbortController();
    setFrameLoading(true);
    setFrameError("");
    traceApi
      .getFrame(urlState.run, urlState.frame, controller.signal)
      .then(setFrameDetail)
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        setFrameDetail(null);
        setFrameError(reason instanceof Error ? reason.message : "无法读取帧证据");
      })
      .finally(() => {
        if (!controller.signal.aborted) setFrameLoading(false);
      });
    return () => controller.abort();
  }, [urlState.frame, urlState.run]);

  return (
    <div className="app-shell">
      <RunCatalog
        runs={runs}
        total={runTotal}
        offset={loadedRunOffset}
        limit={runLimit}
        selectedRunKey={urlState.run}
        query={urlState.q}
        loading={loading}
        error={error}
        onSelect={(run) => setUrlState({
          run,
          frame: "",
          offset: "",
          stage: "",
          track: "",
          status: "",
        })}
        onSearch={(q) => setUrlState({
          q,
          run_offset: "",
          run: "",
          frame: "",
          offset: "",
          stage: "",
          track: "",
          status: "",
        })}
        onPage={(offset) => setUrlState({
          run_offset: offset > 0 ? String(offset) : "",
        })}
      />
      <main className="studio-main">
        {!urlState.run && (
          <div className="empty-studio">
            <span className="eyebrow">PIPELINE EVIDENCE</span>
            <h2>选择一个数据项</h2>
            <p>逐帧检查双目证据、FHP21 骨架与完整来源链。</p>
          </div>
        )}
        {urlState.run && detailLoading && !detail && (
          <div className="empty-studio"><span className="eyebrow">LOADING TRACE</span><h2>正在打开运行…</h2></div>
        )}
        {urlState.run && detailError && (
          <div className="empty-studio error" role="alert"><span className="eyebrow">API ERROR</span><h2>无法打开运行</h2><p>{detailError}</p></div>
        )}
        {detail && (
          <div className="studio-content">
            <RunHeader detail={detail} />
            <TraceFilters
              stages={detail.stages}
              tracks={detail.track_ids}
              stage={urlState.stage}
              track={urlState.track}
              status={urlState.status}
              onChange={(patch) => setUrlState({ ...patch, frame: "", offset: "" })}
            />
            <FrameTimeline
              frames={frames}
              total={frameTotal}
              offset={loadedFrameOffset}
              limit={frameLimit}
              selectedFrameKey={urlState.frame}
              loading={framesLoading}
              error={framesError}
              onSelect={(frameKey) => setUrlState({ frame: frameKey })}
              onPage={(offset) => setUrlState({ offset: offset > 0 ? String(offset) : "", frame: "" })}
            />
            <FrameInspector
              runKey={detail.run.run_key}
              runDetail={detail}
              frameDetail={frameDetail}
              selectedTrack={urlState.track}
              loading={frameLoading}
              error={frameError}
            />
          </div>
        )}
      </main>
    </div>
  );
}
