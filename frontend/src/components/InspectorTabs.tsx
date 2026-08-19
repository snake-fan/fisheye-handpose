import {
  Braces,
  CheckCircle2,
  CircleMinus,
  Download,
  FileJson2,
  GitBranch,
  ListTree,
  TriangleAlert,
  XCircle,
} from "lucide-react";
import { useState } from "react";

import { traceApi } from "../api/client";
import type { RunDetail, TraceRecord } from "../api/types";
import { artifactPathFor } from "../domain/frameEvidence";
import { bytesDisplay, provenanceFacts, runArtifacts } from "../domain/provenance";
import { asDisplay, payloadOf, recordLabel } from "../domain/trace";

interface InspectorTabsProps {
  detail: RunDetail;
  records: TraceRecord[];
}

type Tab = "stages" | "qa" | "provenance" | "json";

const tabs: Array<{ id: Tab; label: string; icon: typeof ListTree }> = [
  { id: "stages", label: "阶段", icon: ListTree },
  { id: "qa", label: "QA", icon: CheckCircle2 },
  { id: "provenance", label: "来源", icon: GitBranch },
  { id: "json", label: "JSON", icon: Braces },
];

function StatusMark({ status }: { status?: string }) {
  if (status === "FAILED") return <XCircle className="record-failed" aria-hidden="true" />;
  if (status === "WARNING") return <TriangleAlert className="record-warning" aria-hidden="true" />;
  if (status === "SKIPPED") return <CircleMinus className="record-skipped" aria-hidden="true" />;
  return <CheckCircle2 className="record-success" aria-hidden="true" />;
}

function StageRecordCard({
  record,
  continued,
}: {
  record: TraceRecord;
  continued: boolean;
}) {
  const payload = payloadOf(record);
  return (
    <article className="stage-record">
      <div className="stage-node"><StatusMark status={record.status} />{continued && <i />}</div>
      <div>
        <div><span>{record.stage ?? "RECORD"}</span><small>{record.status ?? "UNKNOWN"}</small></div>
        <strong>{record.event ?? recordLabel(record)}</strong>
        {payload.output_status !== undefined && (
          <span className="stage-output-status">{asDisplay(payload.output_status)}</span>
        )}
        {payload.reason !== undefined && <p className="stage-reason">{asDisplay(payload.reason)}</p>}
        <p className="stage-record-id">{recordLabel(record)}</p>
      </div>
    </article>
  );
}

function StagePanel({ records }: { records: TraceRecord[] }) {
  return (
    <div className="stage-records">
      {records.map((record, index) => (
        <StageRecordCard
          key={recordLabel(record)}
          record={record}
          continued={index < records.length - 1}
        />
      ))}
      {!records.length && <div className="tab-empty">此帧没有阶段记录</div>}
    </div>
  );
}

function QaPanel({ detail, records }: InspectorTabsProps) {
  const qaRecords = records.filter((record) => record.stage === "QA" || record.status === "WARNING" || record.status === "FAILED");
  return (
    <div className="qa-grid">
      <div className={`integrity-banner ${detail.validation.ok ? "ok" : "failed"}`}>
        <StatusMark status={detail.validation.ok ? "SUCCEEDED" : "FAILED"} />
        <div><strong>{detail.validation.ok ? "Trace integrity verified" : "Trace integrity failed"}</strong><span>{detail.validation.errors.length} errors · {detail.validation.warnings.length} warnings</span></div>
      </div>
      {qaRecords.map((record) => {
        const payload = payloadOf(record);
        return (
          <article key={recordLabel(record)} className={`qa-item qa-${(record.status ?? "unknown").toLowerCase()}`}>
            <header><StatusMark status={record.status} /><strong>{asDisplay(payload.metric, record.event ?? recordLabel(record))}</strong><span>{record.status}</span></header>
            <div className="qa-values">
              <div><span>VALUE</span><strong>{asDisplay(payload.value)}</strong></div>
              <div><span>THRESHOLD</span><strong>{asDisplay(payload.threshold)}</strong></div>
            </div>
            {payload.message !== undefined && <p>{asDisplay(payload.message)}</p>}
          </article>
        );
      })}
      {qaRecords.length === 0 && <div className="tab-empty">此帧没有 QA 警告或失败</div>}
    </div>
  );
}

function ProvenancePanel({ detail, records }: InspectorTabsProps) {
  const facts = provenanceFacts(detail, records);
  const artifacts = runArtifacts(detail, records);
  return (
    <div className="provenance-layout">
      <div className="provenance-sidebar">
        <section className="manifest-facts">
          <h3>H20 PROVENANCE</h3>
          <dl>
            <div><dt>Model manifest</dt><dd title={facts.modelManifestSha256}>{asDisplay(facts.modelManifestSha256)}</dd></div>
            <div><dt>MMPose commit</dt><dd title={facts.mmposeCommit}>{asDisplay(facts.mmposeCommit)}</dd></div>
            <div><dt>Request hash</dt><dd title={facts.requestSha256}>{asDisplay(facts.requestSha256)}</dd></div>
            <div><dt>Calibration ID</dt><dd title={facts.calibrationId}>{asDisplay(facts.calibrationId)}</dd></div>
            <div><dt>Calibration hash</dt><dd title={facts.calibrationSha256}>{asDisplay(facts.calibrationSha256)}</dd></div>
            <div><dt>MANO manifest</dt><dd title={facts.manoManifestSha256}>{asDisplay(facts.manoManifestSha256)}</dd></div>
            <div><dt>MANO mapping</dt><dd title={facts.manoMappingId}>{asDisplay(facts.manoMappingId)}</dd></div>
            <div><dt>Pipeline</dt><dd>{detail.run.pipeline_version ?? "unknown"}</dd></div>
          </dl>
        </section>
        <section className="run-artifacts">
          <div className="artifact-section-title">
            <h3>CONTENT-ADDRESSED ARTIFACTS</h3>
            <span>{artifacts.length}</span>
          </div>
          <div className="artifact-downloads">
            {artifacts.map((artifact) => {
              const path = artifactPathFor(artifact);
              const role = String(artifact.role ?? "artifact");
              const finalOutput = role === "worker_fhp21_output";
              return (
                <a
                  key={`${path}:${String(artifact.sha256)}:${role}`}
                  className={finalOutput ? "artifact-download final-output" : "artifact-download"}
                  href={traceApi.artifactUrl(detail.run.run_key, path)}
                  aria-label={`下载 ${role}`}
                  download={path.split("/").at(-1)}
                >
                  <span className="artifact-file-icon">{finalOutput ? <FileJson2 aria-hidden="true" /> : <Download aria-hidden="true" />}</span>
                  <span className="artifact-download-copy">
                    <strong>{role}</strong>
                    <small>{bytesDisplay(artifact.bytes)} · sha256:{String(artifact.sha256).slice(0, 12)}</small>
                  </span>
                  {finalOutput && <i>FINAL FHP21</i>}
                </a>
              );
            })}
            {artifacts.length === 0 && <div className="artifact-empty">没有可下载的内容寻址工件</div>}
          </div>
        </section>
      </div>
      <section className="dag-list">
        <h3>FRAME PROVENANCE DAG</h3>
        {records.map((record) => (
          <article key={recordLabel(record)}>
            <span>{record.stage}</span>
            <strong>{recordLabel(record)}</strong>
            <small>{record.parent_ids?.length ? record.parent_ids.join(" + ") : "root evidence"}</small>
          </article>
        ))}
      </section>
    </div>
  );
}

export function InspectorTabs({ detail, records }: InspectorTabsProps) {
  const [active, setActive] = useState<Tab>("stages");
  return (
    <section className="inspector-tabs">
      <div className="tab-list" role="tablist" aria-label="帧详情">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            role="tab"
            id={`tab-${id}`}
            aria-selected={active === id}
            aria-controls={`panel-${id}`}
            onClick={() => setActive(id)}
          ><Icon aria-hidden="true" /> {label}</button>
        ))}
      </div>
      <div className="tab-panel" role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {active === "stages" && <StagePanel records={records} />}
        {active === "qa" && <QaPanel detail={detail} records={records} />}
        {active === "provenance" && <ProvenancePanel detail={detail} records={records} />}
        {active === "json" && <pre className="raw-json">{JSON.stringify(records, null, 2)}</pre>}
      </div>
    </section>
  );
}
