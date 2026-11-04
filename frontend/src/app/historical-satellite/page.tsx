"use client";

import React, { useState } from "react";
import Link from "next/link";
import { ArrowLeft, History, ShieldCheck, ShieldAlert } from "lucide-react";
import { HistoricalSatelliteView } from "@/components/satellite/HistoricalSatelliteView";
import { TopStatusBar } from "@/components/navigation/TopStatusBar";
import { BestTrackStorm } from "@/lib/api";

export default function HistoricalSatellitePage() {
  const [trackMode, setTrackMode] = useState<"REAL" | "DEMO" | "NONE">("REAL");
  const [selectedStorm, setSelectedStorm] = useState<BestTrackStorm | null>(null);

  const isRealHistorical = trackMode === "REAL";

  return (
    <div className="min-h-screen font-sans flex flex-col text-slate-800 bg-[linear-gradient(90deg,#93c5fd_0%,#bfdbfe_14%,#dbeafe_26%,#eff6ff_38%,#ffffff_48%,#ffffff_52%,#eff6ff_62%,#dbeafe_74%,#bfdbfe_86%,#93c5fd_100%)]">
      {/* 1. Dynamic Top Status Bar */}
      <TopStatusBar
        isBackendHealthy={true}
        dataMode="REAL"
        provenanceLabel="VERIFIED HISTORICAL DATA"
        sourceAgency={selectedStorm?.source || "IMD_RSMC_NEW_DELHI"}
      />

      <div className="flex-1 p-4 md:p-6 space-y-6">
        {/* Navigation & Context Sub-header */}
        <header className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-200">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-amber-50 border border-amber-200 flex items-center justify-center shadow-xs">
              <History className="w-5 h-5 text-amber-700 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-black text-lg tracking-wider text-slate-900">CYCLONE</span>
                <span className="font-black text-lg tracking-wider text-blue-700">-X</span>
                <span
                  className={`text-xs px-2 py-0.5 rounded font-mono font-bold ml-2 border ${
                    isRealHistorical
                      ? "bg-emerald-50 text-emerald-800 border-emerald-200"
                      : "bg-amber-50 text-amber-800 border-amber-200"
                  }`}
                >
                  {isRealHistorical ? "VERIFIED HISTORICAL MODE" : "HISTORICAL DEMO MODE"}
                </span>
              </div>
              <span className="text-xs text-slate-500 font-mono block font-medium">
                Historical INSAT-3D Satellite Intelligence • Authentic Archive (2013–2021)
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-white border border-slate-300 text-xs font-mono text-slate-700 hover:text-slate-900 hover:border-blue-500 transition-all shadow-2xs font-semibold"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>OPERATIONS DASHBOARD</span>
            </Link>
          </div>
        </header>

        {/* 4-Step End-to-End Analysis Workflow Header */}
        <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-xs">
          <div className="text-[10px] font-mono text-slate-500 font-bold uppercase tracking-wider pb-2 mb-3 border-b border-slate-100 flex items-center justify-between">
            <span>Historical Evaluation Workflow</span>
            <span className="text-emerald-700 font-semibold">CNN V3 &amp; TRANSFORMER V1 ACTIVE</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs font-mono">
            <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 flex items-center gap-2.5">
              <span className="w-5 h-5 rounded-full bg-blue-100 border border-blue-300 text-blue-800 flex items-center justify-center text-[10px] font-bold">1</span>
              <div>
                <div className="text-slate-800 font-bold">Select Satellite Tile</div>
                <div className="text-[10px] text-slate-500">INSAT-3D IR / PDF Archive</div>
              </div>
            </div>

            <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 flex items-center gap-2.5">
              <span className="w-5 h-5 rounded-full bg-cyan-100 border border-cyan-300 text-cyan-800 flex items-center justify-center text-[10px] font-bold">2</span>
              <div>
                <div className="text-slate-800 font-bold">Run Detection</div>
                <div className="text-[10px] text-cyan-700 font-medium">CNN V3 Inference</div>
              </div>
            </div>

            <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 flex items-center gap-2.5">
              <span className="w-5 h-5 rounded-full bg-emerald-100 border border-emerald-300 text-emerald-800 flex items-center justify-center text-[10px] font-bold">3</span>
              <div>
                <div className="text-slate-800 font-bold">Select Track Fixes</div>
                <div className="text-[10px] text-emerald-700 font-medium">IMD / IBTrACS Sequence</div>
              </div>
            </div>

            <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 flex items-center gap-2.5">
              <span className="w-5 h-5 rounded-full bg-purple-100 border border-purple-300 text-purple-800 flex items-center justify-center text-[10px] font-bold">4</span>
              <div>
                <div className="text-slate-800 font-bold">Predict Trajectory</div>
                <div className="text-[10px] text-purple-700 font-medium">TRANSFORMER V1 + Risk Engine</div>
              </div>
            </div>
          </div>
        </div>

        {/* Main Historical Satellite View */}
        <HistoricalSatelliteView
          onTrackProvenanceChange={(mode, storm) => {
            setTrackMode(mode);
            setSelectedStorm(storm);
          }}
        />
      </div>

      {/* Persistent Bottom Status Bar */}
      <footer className="border-t border-slate-200 bg-white px-4 py-2.5 text-[11px] font-mono text-slate-600 flex flex-wrap items-center justify-between gap-2 select-none shadow-2xs">
        <div className="flex items-center gap-3">
          <span className="text-slate-900 font-bold">CYCLONE-X HISTORICAL SATELLITE STUDIO</span>
          <span>•</span>
          <span className="text-blue-700 font-medium">AUTHENTIC INSAT-3D ARCHIVE (2013–2021)</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-emerald-700 font-bold">CNN V3, RESNET-18 &amp; TRANSFORMER V1 READY</span>
          <span>•</span>
          <span className="text-emerald-700 font-bold flex items-center gap-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
            VERIFIED HISTORICAL DATA
          </span>
        </div>
      </footer>
    </div>
  );
}

