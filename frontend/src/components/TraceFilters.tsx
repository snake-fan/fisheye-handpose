import { Filter, Layers3, ScanSearch } from "lucide-react";

interface TraceFiltersProps {
  stages: string[];
  tracks: string[];
  stage: string;
  track: string;
  status: string;
  onChange: (patch: { stage?: string; track?: string; status?: string }) => void;
}

export function TraceFilters({
  stages,
  tracks,
  stage,
  track,
  status,
  onChange,
}: TraceFiltersProps) {
  return (
    <section className="trace-toolbar" aria-label="帧筛选">
      <div className="toolbar-label">
        <Filter aria-hidden="true" />
        <span>FILTER EVIDENCE</span>
      </div>
      <label className="select-field">
        <Layers3 aria-hidden="true" />
        <span>阶段</span>
        <select
          aria-label="阶段"
          value={stage}
          onChange={(event) => onChange({ stage: event.target.value })}
        >
          <option value="">全部阶段</option>
          {stages.map((value) => <option key={value} value={value}>{value}</option>)}
        </select>
      </label>
      <label className="select-field">
        <ScanSearch aria-hidden="true" />
        <span>Track</span>
        <select
          aria-label="Track"
          value={track}
          onChange={(event) => onChange({ track: event.target.value })}
        >
          <option value="">全部 Track</option>
          {tracks.map((value) => <option key={value} value={value}>{value}</option>)}
        </select>
      </label>
      <label className="select-field status-field">
        <span>状态</span>
        <select
          aria-label="状态"
          value={status}
          onChange={(event) => onChange({ status: event.target.value })}
        >
          <option value="">全部状态</option>
          <option value="SUCCEEDED">SUCCEEDED</option>
          <option value="WARNING">WARNING</option>
          <option value="FAILED">FAILED</option>
          <option value="SKIPPED">SKIPPED</option>
        </select>
      </label>
      {(stage || track || status) && (
        <button
          type="button"
          className="clear-filters"
          onClick={() => onChange({ stage: "", track: "", status: "" })}
        >
          清除筛选
        </button>
      )}
    </section>
  );
}
