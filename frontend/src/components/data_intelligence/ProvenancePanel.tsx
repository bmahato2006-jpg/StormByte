"use client";

import React, { useState, useEffect } from "react";
import { DataProvenance, DataMode } from "@/types/data_intelligence";
import { fetchProvenance } from "@/lib/api";

export function ProvenancePanel() {
  const [records, setRecords] = useState<DataProvenance[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [modeFilter, setModeFilter] = useState<string>("ALL");

  const loadData = async () => {
    try {
      setLoading(true);
      const res = await fetchProvenance(100);
      setRecords(res);
    } catch (err) {
      console.error("Failed to load provenance records:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const filtered = records.filter((r) => {
    const matchesSearch =
      r.source.toLowerCase().includes(search.toLowerCase()) ||
      r.provider.toLowerCase().includes(search.toLowerCase()) ||
      r.product.toLowerCase().includes(search.toLowerCase()) ||
      (r.source_reference && r.source_reference.toLowerCase().includes(search.toLowerCase()));

    const matchesMode = modeFilter === "ALL" || r.data_mode === modeFilter;
    return matchesSearch && matchesMode;
  });

  const getModeBadge = (mode: DataMode) => {
    switch (mode) {
      case "DEMO":
        return "bg-blue-50 text-blue-700 border-blue-200";
      case "HISTORICAL":
        return "bg-purple-50 text-purple-700 border-purple-200";
      case "USER_UPLOAD":
        return "bg-emerald-50 text-emerald-700 border-emerald-200";
      case "LIVE_NRT":
        return "bg-cyan-50 text-cyan-800 border-cyan-200";
      case "OFFICIAL_INFO":
        return "bg-amber-50 text-amber-800 border-amber-200";
      default:
        return "bg-slate-100 text-slate-700 border-slate-200";
    }
  };

  return (
    <div className="rounded-xl bg-white border border-slate-200 p-5 flex flex-col gap-4 shadow-sm">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-200 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-500 shadow-xs"></span>
            <h3 className="text-sm font-semibold tracking-wider text-slate-900 uppercase font-mono">
              Scientific Data Provenance & Audit Trail
            </h3>
          </div>
          <p className="text-xs text-slate-600 mt-1 font-mono">
            Immutable tracking of observational origin, physical sensors, acquisition UTC, and scientific licensing.
          </p>
        </div>

        {/* Filter Controls */}
        <div className="flex items-center gap-2">
          <input
            type="text"
            placeholder="Search provider, product..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="px-3 py-1.5 rounded-lg bg-white border border-slate-300 text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-blue-500 font-mono"
          />

          <select
            value={modeFilter}
            onChange={(e) => setModeFilter(e.target.value)}
            className="px-3 py-1.5 rounded-lg bg-white border border-slate-300 text-xs text-slate-800 focus:outline-none focus:border-blue-500 font-mono"
          >
            <option value="ALL">ALL MODES</option>
            <option value="DEMO">DEMO</option>
            <option value="HISTORICAL">HISTORICAL</option>
            <option value="USER_UPLOAD">USER UPLOAD</option>
            <option value="LIVE_NRT">LIVE NRT</option>
          </select>

          <button
            onClick={loadData}
            title="Refresh provenance log"
            className="px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 border border-slate-300 text-xs text-slate-700 font-mono transition-colors"
          >
            REFRESH
          </button>
        </div>
      </div>

      {/* Provenance Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-slate-200 text-slate-600 font-mono text-[11px] uppercase tracking-wider bg-slate-50">
              <th className="py-2.5 px-3">DATA SOURCE</th>
              <th className="py-2.5 px-3">PROVIDER</th>
              <th className="py-2.5 px-3">PRODUCT</th>
              <th className="py-2.5 px-3">DATA MODE</th>
              <th className="py-2.5 px-3">ACQUISITION TIME</th>
              <th className="py-2.5 px-3">PROCESSING TIME</th>
              <th className="py-2.5 px-3">VERSION</th>
              <th className="py-2.5 px-3">STATUS</th>
              <th className="py-2.5 px-3">SOURCE REFERENCE</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 font-mono">
            {loading ? (
              <tr>
                <td colSpan={9} className="py-8 text-center text-slate-500">
                  <div className="flex items-center justify-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-blue-500 animate-ping"></span>
                    Loading scientific provenance ledger...
                  </div>
                </td>
              </tr>
            ) : filtered.length === 0 ? (
              <tr>
                <td colSpan={9} className="py-6 text-center text-slate-500">
                  No provenance records found matching criteria.
                </td>
              </tr>
            ) : (
              filtered.map((item) => (
                <tr
                  key={item.id}
                  className="hover:bg-slate-50 transition-colors text-slate-700"
                >
                  <td className="py-2.5 px-3 font-semibold text-blue-700 whitespace-nowrap">
                    {item.source}
                  </td>
                  <td className="py-2.5 px-3 text-slate-800 whitespace-nowrap">
                    {item.provider}
                  </td>
                  <td className="py-2.5 px-3 max-w-[200px] truncate text-slate-700" title={item.product}>
                    {item.product}
                  </td>
                  <td className="py-2.5 px-3 whitespace-nowrap">
                    <span
                      className={`inline-block px-2 py-0.5 rounded text-[10px] border font-bold ${getModeBadge(
                        item.data_mode
                      )}`}
                    >
                      {item.data_mode}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-slate-600 whitespace-nowrap">
                    {item.acquisition_time
                      ? new Date(item.acquisition_time).toISOString().replace("T", " ").replace("Z", " UTC")
                      : "—"}
                  </td>
                  <td className="py-2.5 px-3 text-slate-600 whitespace-nowrap">
                    {item.processing_time
                      ? new Date(item.processing_time).toISOString().replace("T", " ").replace("Z", " UTC")
                      : "—"}
                  </td>
                  <td className="py-2.5 px-3 text-slate-600">{item.dataset_version}</td>
                  <td className="py-2.5 px-3 whitespace-nowrap">
                    <span className="px-1.5 py-0.5 rounded text-[10px] bg-slate-100 text-slate-700 border border-slate-200">
                      {item.status}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-slate-500 max-w-[220px] truncate" title={item.source_reference || ""}>
                    {item.source_reference || "Official Archive"}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

