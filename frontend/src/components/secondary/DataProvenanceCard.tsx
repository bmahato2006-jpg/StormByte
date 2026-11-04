"use client";

import React from "react";
import { Database, ShieldCheck, Clock, Layers, FileCode } from "lucide-react";
import { DataProvenance } from "@/types/provenance";

interface DataProvenanceCardProps {
  provenance: DataProvenance | null;
}

export const DataProvenanceCard: React.FC<DataProvenanceCardProps> = ({ provenance }) => {
  if (!provenance) {
    return null;
  }

  return (
    <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-xs flex flex-col justify-between h-full">
      <div className="flex items-center justify-between border-b border-slate-200 pb-2 mb-3">
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4 text-blue-700" />
          <h3 className="text-xs font-mono font-bold text-slate-900 uppercase tracking-wider">
            DATA PROVENANCE & ATTRIBUTION AUDIT
          </h3>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-50 border border-emerald-200 text-emerald-800 font-bold">
          VERIFIED AUDIT LOG
        </span>
      </div>

      <div className="space-y-2 text-xs font-mono">
        <div className="p-2.5 rounded bg-slate-50 border border-slate-200">
          <div className="text-slate-500 text-[10px] uppercase font-semibold">Data Classification & Tag:</div>
          <div className="text-sm font-bold text-amber-700 mt-0.5 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-amber-500 animate-ping" />
            {provenance.dataType}
          </div>
        </div>

        <div className="p-2.5 rounded bg-slate-50 border border-slate-200">
          <div className="text-slate-500 text-[10px] uppercase font-semibold">Primary Observation Origin:</div>
          <div className="text-xs font-semibold text-slate-800 mt-0.5">{provenance.source}</div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div className="p-2 rounded bg-slate-50 border border-slate-200">
            <div className="text-slate-500 text-[10px] uppercase font-semibold">Processing Engine:</div>
            <div className="text-xs font-bold text-blue-700 mt-0.5">{provenance.processingVersion}</div>
          </div>
          <div className="p-2 rounded bg-slate-50 border border-slate-200">
            <div className="text-slate-500 text-[10px] uppercase font-semibold">Model State:</div>
            <div className="text-xs font-semibold text-slate-800 mt-0.5 truncate">{provenance.modelVersion}</div>
          </div>
        </div>

        <div className="p-2 rounded bg-slate-50 border border-slate-200">
          <div className="text-slate-500 text-[10px] uppercase font-semibold">Ingestion UTC Timestamp:</div>
          <div className="text-xs font-mono text-slate-900 mt-0.5 font-semibold">{new Date(provenance.timestamp).toUTCString()}</div>
        </div>
      </div>

      <div className="mt-2 text-[10px] font-mono text-slate-500 pt-2 border-t border-slate-200">
        Every payload is cryptographically tagged with provenance metadata to maintain scientific reproducibility.
      </div>
    </div>
  );
};

