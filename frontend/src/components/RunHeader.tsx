import { AlertTriangle, CheckCircle2, CircleDashed, ShieldCheck, XCircle } from "lucide-react";

import type { RunDetail } from "../api/types";

interface RunHeaderProps {
  detail: RunDetail;
}

function formatDate(value: string | null): string {
  if (!value) return "仍在运行";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

export function RunHeader({ detail }: RunHeaderProps) {
  const { run, validation } = detail;
  const StatusIcon = run.status === "COMPLETED"
    ? CheckCircle2
    : run.status === "FAILED"
      ? XCircle
      : CircleDashed;

  return (
    <header className="run-header">
      <div className="run-heading">
        <div className="breadcrumb">
          <span>RUNS</span><span>/</span><span>INSPECTION</span>
        </div>
        <div className="run-name-row">
          <div className="run-identity">
            <span>{run.item_id}</span>
            <h1>{run.run_id}</h1>
          </div>
          <span className={`large-status status-${run.status.toLowerCase()}`}>
            <StatusIcon aria-hidden="true" /> {run.status}
          </span>
        </div>
        <p>
          创建于 {formatDate(run.created_at_utc)} · Pipeline {run.pipeline_version ?? "unknown"}
        </p>
      </div>

      <div className="run-kpis" aria-label="运行统计">
        <div>
          <span>FRAMES</span>
          <strong>{run.frame_count.toLocaleString()}</strong>
          <small>{run.frame_count.toLocaleString()} 帧</small>
        </div>
        <div>
          <span>RECORDS</span>
          <strong>{run.record_count.toLocaleString()}</strong>
          <small>可追溯记录</small>
        </div>
        <div className={run.warning_count ? "kpi-warn" : ""}>
          <span>WARNINGS</span>
          <strong>{run.warning_count}</strong>
          <small>{run.warning_count ? <><AlertTriangle aria-hidden="true" /> 需关注</> : "无警告"}</small>
        </div>
        <div className={run.failure_count ? "kpi-fail" : ""}>
          <span>FAILURES</span>
          <strong>{run.failure_count}</strong>
          <small>{run.failure_count ? "阶段失败" : "无失败"}</small>
        </div>
        <div className={validation.ok ? "kpi-valid" : "kpi-fail"}>
          <span>INTEGRITY</span>
          <strong>{validation.ok ? "OK" : "ERR"}</strong>
          <small><ShieldCheck aria-hidden="true" /> hash chain</small>
        </div>
      </div>
    </header>
  );
}
