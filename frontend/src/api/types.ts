export type RunStatus = "ACTIVE" | "COMPLETED" | "FAILED" | string;

export interface RunSummary {
  run_key: string;
  item_id: string;
  run_id: string;
  data_item_id?: string | null;
  status: RunStatus;
  created_at_utc: string | null;
  finalized_at_utc: string | null;
  pipeline_version: string | null;
  record_count: number;
  frame_count: number;
  stage_counts: Record<string, number>;
  warning_count: number;
  failure_count: number;
}

export interface Page<T> {
  items: T[];
  offset: number;
  limit: number;
  total: number;
}

export interface ValidationSummary {
  ok: boolean;
  errors: unknown[];
  warnings: unknown[];
  [key: string]: unknown;
}

export interface RunDetail {
  run: RunSummary;
  manifest: Record<string, unknown>;
  summary: Record<string, unknown> | null;
  validation: ValidationSummary;
  provenance?: RunProvenance;
  stages: string[];
  track_ids: string[];
  view_ids: string[];
  global_records?: TraceRecord[];
}

export interface ProvenanceFacet {
  record_id?: string;
  payload?: Record<string, unknown>;
  worker_provenance?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface RunProvenance {
  manifest?: ProvenanceFacet;
  worker_inputs?: ProvenanceFacet;
  calibration?: ProvenanceFacet;
  mano?: ProvenanceFacet;
  mano_configuration?: ProvenanceFacet;
  [key: string]: ProvenanceFacet | undefined;
}

export interface FrameSummary {
  frame_key: string;
  frame_id: string;
  frame_index: number | null;
  timestamp_ns: number | null;
  record_ids: string[];
  stages: string[];
  statuses: string[];
  track_ids: string[];
  view_ids: string[];
}

export interface ArtifactRef {
  sha256?: string;
  bytes?: number;
  role?: string;
  media_type?: string;
  relative_path?: string;
  path?: string;
  [key: string]: unknown;
}

export interface TraceRecord {
  record_id?: string;
  ordinal?: number;
  timestamp_utc?: string;
  stage?: string;
  status?: string;
  event?: string;
  parent_ids?: string[];
  blobs?: ArtifactRef[];
  artifacts?: ArtifactRef[];
  payload?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface FrameDetail {
  run_key?: string;
  run_id: string;
  frame: FrameSummary;
  records: TraceRecord[];
}

export interface RunQuery {
  offset?: number;
  limit?: number;
  status?: string;
  q?: string;
}

export interface FrameQuery {
  offset?: number;
  limit?: number;
  stage?: string;
  track_id?: string;
  status?: string;
}
