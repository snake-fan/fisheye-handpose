import { AlertTriangle, Check, ChevronLeft, ChevronRight, LoaderCircle, Search, X } from "lucide-react";
import { FormEvent, useEffect, useState } from "react";

import type { RunSummary } from "../api/types";

interface RunCatalogProps {
  runs: RunSummary[];
  total: number;
  offset: number;
  limit: number;
  selectedRunKey: string;
  query: string;
  loading: boolean;
  error: string;
  onSelect: (runKey: string) => void;
  onSearch: (query: string) => void;
  onPage: (offset: number) => void;
}

function compact(value: number): string {
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function StatusIcon({ status }: { status: string }) {
  if (status === "COMPLETED") return <Check aria-hidden="true" />;
  if (status === "FAILED") return <X aria-hidden="true" />;
  return <LoaderCircle aria-hidden="true" />;
}

export function RunCatalog({
  runs,
  total,
  offset,
  limit,
  selectedRunKey,
  query,
  loading,
  error,
  onSelect,
  onSearch,
  onPage,
}: RunCatalogProps) {
  const [draftQuery, setDraftQuery] = useState(query);
  useEffect(() => setDraftQuery(query), [query]);
  const submitSearch = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    onSearch(draftQuery.trim());
  };
  const pageEnd = offset + runs.length;
  const hasPrevious = offset > 0;
  const hasNext = runs.length > 0 && pageEnd < total;

  return (
    <aside className="run-catalog" aria-label="运行目录">
      <div className="brand-lockup">
        <div className="brand-mark" aria-hidden="true">
          <span />
          <span />
        </div>
        <div>
          <span className="eyebrow">FISHEYE · FHP21</span>
          <strong>Trace Studio</strong>
        </div>
      </div>

      <div className="catalog-title">
        <div>
          <span className="section-index">01</span>
          <h1>运行记录</h1>
        </div>
        <span className="run-total">{total}</span>
      </div>

      <form className="catalog-search" role="search" onSubmit={submitSearch}>
        <Search aria-hidden="true" />
        <input
          type="search"
          placeholder="搜索数据项或 Run ID…"
          aria-label="搜索运行"
          value={draftQuery}
          onChange={(event) => setDraftQuery(event.target.value)}
        />
        {draftQuery && (
          <button
            type="button"
            aria-label="清除运行搜索"
            onClick={() => {
              setDraftQuery("");
              onSearch("");
            }}
          ><X aria-hidden="true" /></button>
        )}
      </form>

      <div className="run-list">
        {loading && (
          <div className="catalog-message">
            <LoaderCircle className="spin" aria-hidden="true" /> 正在读取运行目录
          </div>
        )}
        {error && <div className="catalog-message error" role="alert">{error}</div>}
        {!loading && !error && runs.length === 0 && (
          <div className="catalog-message">还没有可检查的运行</div>
        )}
        {runs.map((run) => (
          <button
            key={run.run_key}
            type="button"
            className={`run-item ${selectedRunKey === run.run_key ? "selected" : ""}`}
            onClick={() => onSelect(run.run_key)}
            aria-pressed={selectedRunKey === run.run_key}
          >
            <span className={`run-status status-${run.status.toLowerCase()}`}>
              <StatusIcon status={run.status} />
            </span>
            <span className="run-item-main">
              <strong>{run.item_id ?? run.data_item_id ?? run.run_id}</strong>
              <span>
                {run.item_id || run.data_item_id ? `${run.run_id} · ` : ""}{run.record_count.toLocaleString()} records
              </span>
            </span>
            <span className="run-item-side">
              <span>{compact(run.frame_count)}f</span>
              {run.warning_count > 0 && (
                <span className="warning-count">
                  <AlertTriangle aria-hidden="true" /> {run.warning_count} warnings
                </span>
              )}
            </span>
          </button>
        ))}
      </div>

      <nav className="catalog-pagination" aria-label="运行目录分页">
        <button
          type="button"
          aria-label="上一页运行"
          disabled={!hasPrevious || loading}
          onClick={() => onPage(Math.max(0, offset - limit))}
        >
          <ChevronLeft aria-hidden="true" />
        </button>
        <span>{total > 0 ? `${offset + 1}–${Math.min(pageEnd, total)} / ${total}` : "0 / 0"}</span>
        <button
          type="button"
          aria-label="下一页运行"
          disabled={!hasNext || loading}
          onClick={() => onPage(pageEnd)}
        >
          <ChevronRight aria-hidden="true" />
        </button>
      </nav>

      <footer className="catalog-footer">
        <span className="connection-dot" /> API v1 · READ ONLY
      </footer>
    </aside>
  );
}
