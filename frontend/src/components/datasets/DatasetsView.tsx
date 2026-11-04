"use client";

import React, { useState, useEffect } from "react";
import {
  DatasetMetadata,
  DataQualityReport,
  ScientificDatasetStatus,
  ScientificDataMode,
} from "@/types/dataset";
import {
  Database,
  ShieldCheck,
  AlertTriangle,
  FileCheck,
  Search,
  ExternalLink,
  Key,
  Download,
  Info,
  Layers,
  Cpu,
  RefreshCw,
  X,
  CheckCircle2,
  XCircle,
  Hash,
} from "lucide-react";
import { KaggleINSAT3DSection } from "./KaggleINSAT3DSection";

export function DatasetsView() {
  const [datasets, setDatasets] = useState<DatasetMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Modal States
  const [inspectDataset, setInspectDataset] = useState<DatasetMetadata | null>(null);
  const [qcReport, setQcReport] = useState<DataQualityReport | null>(null);
  const [qcLoading, setQcLoading] = useState(false);
  const [provenanceData, setProvenanceData] = useState<any | null>(null);

  const fetchDatasets = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/datasets");
      if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch datasets`);
      const data = await res.json();
      setDatasets(data);
    } catch (err: any) {
      console.error("Dataset fetch error:", err);
      setError(err.message || "Failed to load datasets from backend gateway");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  const handleInspect = (ds: DatasetMetadata) => {
    setInspectDataset(ds);
  };

  const handleValidate = async (datasetId: string) => {
    setQcLoading(true);
    setQcReport(null);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/datasets/${datasetId}/quality`);
      if (!res.ok) throw new Error(`HTTP ${res.status}: Quality check failed`);
      const data = await res.json();
      setQcReport(data);
    } catch (err: any) {
      console.error("Quality check error:", err);
    } finally {
      setQcLoading(false);
    }
  };

  const handleViewProvenance = async (datasetId: string) => {
    setProvenanceData(null);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/datasets/${datasetId}/provenance`);
      if (!res.ok) throw new Error(`HTTP ${res.status}: Provenance query failed`);
      const data = await res.json();
      setProvenanceData(data);
    } catch (err: any) {
      console.error("Provenance error:", err);
    }
  };

  const getStatusBadge = (status: ScientificDatasetStatus) => {
    switch (status) {
      case "AVAILABLE":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-50 text-emerald-700 border border-emerald-200 flex items-center gap-1">
            <CheckCircle2 className="w-3 h-3" /> AVAILABLE
          </span>
        );
      case "CREDENTIALS_REQUIRED":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-rose-50 text-rose-700 border border-rose-200 flex items-center gap-1">
            <Key className="w-3 h-3" /> CREDENTIALS REQUIRED
          </span>
        );
      case "NOT_DOWNLOADED":
      case "DOWNLOAD_REQUIRED":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-50 text-amber-700 border border-amber-200 flex items-center gap-1">
            <Download className="w-3 h-3" /> DOWNLOAD REQUIRED
          </span>
        );
      case "ACCESS_RESTRICTED":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-purple-50 text-purple-700 border border-purple-200 flex items-center gap-1">
            <ShieldCheck className="w-3 h-3" /> ACCESS RESTRICTED
          </span>
        );
      default:
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-slate-100 text-slate-600 border border-slate-200">
            {status}
          </span>
        );
    }
  };

  const getDataModeBadge = (mode: ScientificDataMode) => {
    switch (mode) {
      case "REAL_RESEARCH":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-blue-50 text-blue-700 border border-blue-200">
            REAL RESEARCH DATA
          </span>
        );
      case "DEMO / SCENARIO":
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-50 text-amber-700 border border-amber-200">
            DEMO / SCENARIO
          </span>
        );
      default:
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-slate-100 text-slate-600 border border-slate-200">
            {mode}
          </span>
        );
    }
  };

  const availableCount = datasets.filter((d) => d.status === "AVAILABLE").length;
  const credsCount = datasets.filter((d) => d.status === "CREDENTIALS_REQUIRED").length;

  return (
    <div className="space-y-6 select-none p-2 sm:p-4">
      {/* Top Banner */}
      <div className="rounded-xl border border-blue-200 bg-white p-5 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 text-[10px] font-mono font-bold uppercase bg-blue-50 text-blue-700 border border-blue-200 rounded">
                Phase 7 Architecture
              </span>
              <span className="text-[11px] font-mono text-slate-500">
                MoES / IMD Problem Statement 26070
              </span>
            </div>
            <h1 className="text-xl sm:text-2xl font-black tracking-wider text-slate-900 flex items-center gap-2">
              <Database className="w-6 h-6 text-blue-600" />
              CYCLONE-X SCIENTIFIC DATA HUB &amp; REPOSITORY
            </h1>
            <p className="text-xs text-slate-600 max-w-4xl leading-relaxed">
              Reproducible, provenance-aware scientific data pipeline supporting NOAA IBTrACS, IMD Best Tracks, MOSDAC INSAT-3D/3DR, NASA GPM IMERG, ECMWF ERA5, and Exposure GIS layers. Unconnected sources truthfully report credential requirements; zero observations or probabilities are simulated.
            </p>
          </div>
          <button
            onClick={fetchDatasets}
            disabled={loading}
            className="self-start sm:self-center px-3 py-2 rounded-lg bg-blue-50 hover:bg-blue-100 border border-blue-200 text-blue-700 text-xs font-mono font-bold flex items-center gap-2 transition-colors shrink-0 shadow-xs"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            Refresh Catalogs
          </button>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-xs">
          <div className="text-[10px] font-mono uppercase text-slate-500">Registered Datasets</div>
          <div className="text-2xl font-black font-mono text-slate-900 mt-1">8</div>
          <div className="text-[10px] font-mono text-slate-500 mt-1">Tracks, Satellite, Hydrology, ERA5</div>
        </div>
        <div className="rounded-xl border border-emerald-200 bg-emerald-50/60 p-3 shadow-xs">
          <div className="text-[10px] font-mono uppercase text-emerald-800 font-semibold">Available On Disk</div>
          <div className="text-2xl font-black font-mono text-emerald-700 mt-1">{availableCount}</div>
          <div className="text-[10px] font-mono text-emerald-600 mt-1">Verified physical raw files</div>
        </div>
        <div className="rounded-xl border border-rose-200 bg-rose-50/60 p-3 shadow-xs">
          <div className="text-[10px] font-mono uppercase text-rose-800 font-semibold">Credentials Required</div>
          <div className="text-2xl font-black font-mono text-rose-700 mt-1">{credsCount}</div>
          <div className="text-[10px] font-mono text-rose-600 mt-1">NASA Earthdata / ECMWF CDS API</div>
        </div>
        <div className="rounded-xl border border-blue-200 bg-blue-50/60 p-3 shadow-xs">
          <div className="text-[10px] font-mono uppercase text-blue-800 font-semibold">Data Leakage Guard</div>
          <div className="text-2xl font-black font-mono text-blue-700 mt-1">ACTIVE</div>
          <div className="text-[10px] font-mono text-blue-600 mt-1">Temporal storm-level partitioning</div>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="rounded-xl border border-red-200 bg-rose-50 p-4 text-rose-800 text-xs font-mono flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 shrink-0 text-rose-600" />
          <div>{error}</div>
        </div>
      )}

      {/* Featured Kaggle INSAT-3D Cyclone Dataset & Model V2 Section */}
      <KaggleINSAT3DSection />

      {/* Dataset Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {datasets.map((ds) => (
          <div
            key={ds.dataset_id}
            className="rounded-xl border border-slate-200 bg-white p-4 flex flex-col justify-between hover:border-blue-300 transition-all shadow-sm"
          >
            <div className="space-y-3">
              {/* Header tags */}
              <div className="flex items-start justify-between gap-2">
                <div className="space-y-1">
                  <span className="text-[10px] font-mono text-slate-500 block">{ds.source}</span>
                  <h3 className="text-sm font-bold text-slate-900 line-clamp-2 leading-snug">{ds.name}</h3>
                </div>
              </div>

              {/* Status and Mode badges */}
              <div className="flex flex-wrap items-center gap-1.5 pt-1">
                {getStatusBadge(ds.status)}
                {getDataModeBadge(ds.data_mode)}
              </div>

              <p className="text-xs text-slate-600 line-clamp-2 leading-relaxed">
                {ds.description}
              </p>

              {/* Technical specs */}
              <div className="rounded-lg bg-slate-50 border border-slate-200 p-2.5 space-y-1.5 text-[11px] font-mono text-slate-700">
                <div className="flex justify-between">
                  <span className="text-slate-500">Format:</span>
                  <span className="text-slate-900 font-semibold">{ds.format}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Resolution:</span>
                  <span className="text-slate-800">{ds.spatial_resolution || "N/A"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Coverage:</span>
                  <span className="text-slate-800 truncate max-w-[140px]" title={ds.coverage}>
                    {ds.coverage}
                  </span>
                </div>
                {ds.checksum && (
                  <div className="flex justify-between items-center pt-1 border-t border-slate-200">
                    <span className="text-slate-500 flex items-center gap-1">
                      <Hash className="w-3 h-3 text-blue-600" /> SHA-256:
                    </span>
                    <span className="text-[10px] font-mono text-blue-700 truncate max-w-[120px]">
                      {ds.checksum.substring(0, 12)}...
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="pt-4 border-t border-slate-800/80 grid grid-cols-3 gap-1.5 text-[10px] font-mono font-bold mt-4">
              <button
                onClick={() => handleInspect(ds)}
                className="px-2 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-center transition-colors flex items-center justify-center gap-1"
              >
                <Info className="w-3 h-3 text-cyan-400" /> Inspect
              </button>
              <button
                onClick={() => handleValidate(ds.dataset_id)}
                className="px-2 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-center transition-colors flex items-center justify-center gap-1"
              >
                <FileCheck className="w-3 h-3 text-emerald-400" /> Validate
              </button>
              <button
                onClick={() => handleViewProvenance(ds.dataset_id)}
                className="px-2 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-center transition-colors flex items-center justify-center gap-1"
              >
                <ShieldCheck className="w-3 h-3 text-purple-400" /> Provenance
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* MODAL 1: Inspect Dataset */}
      {inspectDataset && (
        <div className="fixed inset-0 z-50 bg-slate-900/50 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white border border-slate-200 rounded-xl w-full max-w-2xl max-h-[85vh] overflow-y-auto shadow-2xl p-5 space-y-4 text-slate-800">
            <div className="flex items-center justify-between border-b border-slate-200 pb-3">
              <div className="space-y-0.5">
                <span className="text-[10px] font-mono text-blue-700 font-semibold uppercase tracking-wider">
                  Dataset Technical Specification
                </span>
                <h2 className="text-base font-bold text-slate-900">{inspectDataset.name}</h2>
              </div>
              <button
                onClick={() => setInspectDataset(null)}
                className="text-slate-400 hover:text-slate-700 p-1 rounded-lg hover:bg-slate-100"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <p className="text-slate-600 leading-relaxed">{inspectDataset.description}</p>

              <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
                <div className="bg-slate-50 p-2.5 rounded border border-slate-200 space-y-1">
                  <div className="text-slate-500 font-medium">Publishing Provider:</div>
                  <div className="font-bold text-slate-900">{inspectDataset.provider}</div>
                </div>
                <div className="bg-slate-50 p-2.5 rounded border border-slate-200 space-y-1">
                  <div className="text-slate-500 font-medium">License / Policy:</div>
                  <div className="font-bold text-slate-900">{inspectDataset.license}</div>
                </div>
                <div className="bg-slate-50 p-2.5 rounded border border-slate-200 space-y-1">
                  <div className="text-slate-500 font-medium">Coverage:</div>
                  <div className="font-bold text-slate-900">{inspectDataset.coverage}</div>
                </div>
                <div className="bg-slate-50 p-2.5 rounded border border-slate-200 space-y-1">
                  <div className="text-slate-500 font-medium">Temporal Range:</div>
                  <div className="font-bold text-slate-900">{inspectDataset.time_range || "Continuous"}</div>
                </div>
              </div>

              {/* Variables & Units Table */}
              <div className="space-y-1">
                <div className="text-[11px] font-mono font-bold uppercase text-slate-600">
                  Contained Physical Variables &amp; Units
                </div>
                <div className="bg-white rounded border border-slate-200 overflow-hidden shadow-2xs">
                  <table className="w-full text-left text-[11px] font-mono">
                    <thead className="bg-slate-100 text-slate-700 border-b border-slate-200">
                      <tr>
                        <th className="p-2">Variable</th>
                        <th className="p-2">Unit</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 text-slate-700">
                      {inspectDataset.variables.map((v) => (
                        <tr key={v}>
                          <td className="p-2 font-semibold text-blue-700">{v}</td>
                          <td className="p-2 text-slate-500">{inspectDataset.units[v] || "standard"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Source URL & Checksum */}
              <div className="bg-slate-50 p-2.5 rounded border border-slate-200 space-y-1.5 font-mono text-[11px]">
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Catalog URL:</span>
                  <a
                    href={inspectDataset.source_url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-blue-600 hover:underline flex items-center gap-1 font-semibold"
                  >
                    Open Source <ExternalLink className="w-3 h-3" />
                  </a>
                </div>
                {inspectDataset.checksum && (
                  <div className="break-all pt-1 border-t border-slate-200">
                    <span className="text-slate-500 block">SHA-256 Checksum:</span>
                    <span className="text-emerald-700 font-bold">{inspectDataset.checksum}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* MODAL 2: Quality Control Report */}
      {qcReport && (
        <div className="fixed inset-0 z-50 bg-slate-900/50 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white border border-slate-200 rounded-xl w-full max-w-xl max-h-[85vh] overflow-y-auto shadow-2xl p-5 space-y-4 text-slate-800">
            <div className="flex items-center justify-between border-b border-slate-200 pb-3">
              <div className="space-y-0.5">
                <span className="text-[10px] font-mono text-emerald-700 uppercase tracking-wider font-semibold">
                  Automated Scientific Quality Control Audit
                </span>
                <h2 className="text-base font-bold text-slate-900">{qcReport.dataset_name}</h2>
              </div>
              <button
                onClick={() => setQcReport(null)}
                className="text-slate-400 hover:text-slate-700 p-1 rounded-lg hover:bg-slate-100"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3 text-xs font-mono">
              <div className="flex items-center justify-between bg-slate-50 p-3 rounded-lg border border-slate-200">
                <span className="text-slate-600 font-medium">QC Verdict:</span>
                <span
                  className={`px-2 py-0.5 rounded text-xs font-bold ${
                    qcReport.quality_status === "PASS"
                      ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                      : "bg-amber-50 text-amber-700 border border-amber-200"
                  }`}
                >
                  {qcReport.quality_status}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2">
                <div className="bg-slate-50 p-2.5 rounded border border-slate-200 text-center">
                  <div className="text-[10px] text-slate-500">Total Records</div>
                  <div className="text-lg font-bold text-slate-900">{qcReport.total_records}</div>
                </div>
                <div className="bg-slate-50 p-2.5 rounded border border-slate-200 text-center">
                  <div className="text-[10px] text-emerald-700">Valid Records</div>
                  <div className="text-lg font-bold text-emerald-700">{qcReport.valid_records}</div>
                </div>
                <div className="bg-slate-50 p-2.5 rounded border border-slate-200 text-center">
                  <div className="text-[10px] text-rose-700">Invalid Records</div>
                  <div className="text-lg font-bold text-rose-700">{qcReport.invalid_records}</div>
                </div>
              </div>

              {qcReport.time_range && (
                <div className="bg-slate-50 p-2 rounded border border-slate-200">
                  <span className="text-slate-500">Time Range: </span>
                  <span className="text-slate-800 font-medium">{qcReport.time_range}</span>
                </div>
              )}

              {qcReport.spatial_coverage && (
                <div className="bg-slate-50 p-2 rounded border border-slate-200">
                  <span className="text-slate-500">Spatial Bounds: </span>
                  <span className="text-slate-800 font-medium">{qcReport.spatial_coverage}</span>
                </div>
              )}

              {qcReport.issues.length > 0 && (
                <div className="bg-amber-50 border border-amber-200 p-3 rounded-lg space-y-1">
                  <div className="text-amber-800 font-bold">Audit Notes &amp; Observations:</div>
                  <ul className="list-disc pl-4 text-amber-900 text-[11px] space-y-0.5">
                    {qcReport.issues.map((iss, idx) => (
                      <li key={idx}>{iss}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* MODAL 3: Cryptographic Provenance */}
      {provenanceData && (
        <div className="fixed inset-0 z-50 bg-slate-900/50 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white border border-slate-200 rounded-xl w-full max-w-xl max-h-[85vh] overflow-y-auto shadow-2xl p-5 space-y-4 text-slate-800">
            <div className="flex items-center justify-between border-b border-slate-200 pb-3">
              <div className="space-y-0.5">
                <span className="text-[10px] font-mono text-blue-700 uppercase tracking-wider font-semibold">
                  Cryptographic Provenance Audit Ledger
                </span>
                <h2 className="text-base font-bold text-slate-900">{provenanceData.dataset_id}</h2>
              </div>
              <button
                onClick={() => setProvenanceData(null)}
                className="text-slate-400 hover:text-slate-700 p-1 rounded-lg hover:bg-slate-100"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-2 text-xs font-mono">
              <div className="bg-slate-50 p-2.5 rounded border border-slate-200">
                <span className="text-slate-500">Provider Agency: </span>
                <span className="text-blue-700 font-bold">{provenanceData.provider}</span>
              </div>
              {provenanceData.checksum_sha256 && (
                <div className="bg-slate-50 p-2.5 rounded border border-slate-200 break-all">
                  <span className="text-slate-500 block mb-0.5">Artifact SHA-256:</span>
                  <span className="text-emerald-700 font-bold">{provenanceData.checksum_sha256}</span>
                </div>
              )}
              <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 space-y-2">
                <span className="text-slate-600 block font-bold">Provenance Metadata Payload:</span>
                <pre className="text-[11px] text-slate-800 overflow-x-auto p-2.5 bg-slate-100 border border-slate-200 rounded">
                  {JSON.stringify(provenanceData.provenance_chain, null, 2)}
                </pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

