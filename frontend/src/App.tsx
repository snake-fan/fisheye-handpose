import { useEffect, useRef, useState } from "react";

import { TraceApiError, traceApi } from "./api/client";
import type { FrameDetail, FrameSummary, RunDetail, RunSummary } from "./api/types";
import { FrameInspector } from "./components/FrameInspector";
import { FrameTimeline } from "./components/FrameTimeline";
import { OverlayVideoPlayer } from "./components/OverlayVideoPlayer";
import { RunCatalog } from "./components/RunCatalog";
import { RunHeader } from "./components/RunHeader";
import { TraceFilters } from "./components/TraceFilters";
import { useUrlState } from "./hooks/useUrlState";

const FRAME_PAGE_SIZE = 40;
const RUN_PAGE_SIZE = 50;

type ConnectionStatus = "checking" | "connected" | "disconnected";

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
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("checking");
  const [retryNonce, setRetryNonce] = useState(0);
  const connectionStatusRef = useRef<ConnectionStatus>("checking");

  const recordRequestSuccess = () => {
    connectionStatusRef.current = "connected";
    setConnectionStatus("connected");
  };

  const recordRequestFailure = (reason: unknown) => {
    if (reason instanceof TraceApiError && reason.code === "CONNECTION_FAILED") {
      connectionStatusRef.current = "disconnected";
      setConnectionStatus("disconnected");
    }
  };

  const retryConnection = () => {
    connectionStatusRef.current = "checking";
    setConnectionStatus("checking");
    setRetryNonce((value) => value + 1);
  };

  useEffect(() => {
    if (connectionStatus !== "disconnected") return;
    const timeout = window.setTimeout(retryConnection, 2_000);
    return () => window.clearTimeout(timeout);
  }, [connectionStatus]);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setRuns([]);
    setError("");
    traceApi
      .listRuns({
        offset: pageOffset(urlState.run_offset),
        limit: RUN_PAGE_SIZE,
        q: urlState.q,
      }, controller.signal)
      .then((page) => {
        if (controller.signal.aborted) return;
        setRuns(page.items);
        setRunTotal(page.total);
        setLoadedRunOffset(page.offset);
        setRunLimit(page.limit);
        setError("");
        recordRequestSuccess();
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        recordRequestFailure(reason);
        setError(reason instanceof Error ? reason.message : "无法读取运行目录");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [retryNonce, urlState.q, urlState.run_offset]);

  useEffect(() => {
    if (!urlState.run) {
      setDetail(null);
      setDetailError("");
      setDetailLoading(false);
      return;
    }
    const controller = new AbortController();
    setDetail(null);
    setDetailLoading(true);
    setDetailError("");
    traceApi
      .getRun(urlState.run, controller.signal)
      .then((value) => {
        if (controller.signal.aborted) return;
        setDetail(value);
        recordRequestSuccess();
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        recordRequestFailure(reason);
        setDetail(null);
        setDetailError(reason instanceof Error ? reason.message : "无法读取运行详情");
      })
      .finally(() => {
        if (!controller.signal.aborted) setDetailLoading(false);
      });
    return () => controller.abort();
  }, [retryNonce, urlState.run]);

  useEffect(() => {
    if (!urlState.run) {
      setFrames([]);
      setFrameTotal(0);
      setLoadedFrameOffset(0);
      setFramesError("");
      setFramesLoading(false);
      return;
    }
    const controller = new AbortController();
    setFrames([]);
    setFrameTotal(0);
    setLoadedFrameOffset(0);
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
        if (controller.signal.aborted) return;
        setFrames(page.items);
        setFrameTotal(page.total);
        setLoadedFrameOffset(page.offset);
        setFrameLimit(page.limit);
        recordRequestSuccess();
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        recordRequestFailure(reason);
        setFrames([]);
        setFrameTotal(0);
        setFramesError(reason instanceof Error ? reason.message : "无法读取帧索引");
      })
      .finally(() => {
        if (!controller.signal.aborted) setFramesLoading(false);
      });
    return () => controller.abort();
  }, [retryNonce, urlState.offset, urlState.run, urlState.stage, urlState.status, urlState.track]);

  useEffect(() => {
    if (!urlState.run || !urlState.frame) {
      setFrameDetail(null);
      setFrameError("");
      setFrameLoading(false);
      return;
    }
    const controller = new AbortController();
    setFrameDetail(null);
    setFrameLoading(true);
    setFrameError("");
    traceApi
      .getFrame(urlState.run, urlState.frame, controller.signal)
      .then((value) => {
        if (controller.signal.aborted) return;
        setFrameDetail(value);
        recordRequestSuccess();
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        recordRequestFailure(reason);
        setFrameDetail(null);
        setFrameError(reason instanceof Error ? reason.message : "无法读取帧证据");
      })
      .finally(() => {
        if (!controller.signal.aborted) setFrameLoading(false);
      });
    return () => controller.abort();
  }, [retryNonce, urlState.frame, urlState.run]);

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
        connectionStatus={connectionStatus}
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
            <OverlayVideoPlayer runKey={detail.run.run_key} detail={detail} />
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
      {connectionStatus === "disconnected" && (
        <div className="connection-alert" role="alert">
          <span>追踪 API 连接已断开</span>
          <button type="button" className="retry-button" onClick={retryConnection}>重试连接</button>
        </div>
      )}
    </div>
  );
}
