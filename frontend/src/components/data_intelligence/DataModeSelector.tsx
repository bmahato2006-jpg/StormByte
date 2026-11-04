"use client";

import React from "react";
import { DataMode } from "@/types/data_intelligence";

interface DataModeSelectorProps {
  currentMode: DataMode;
  onModeSelect?: (mode: DataMode) => void;
  className?: string;
}

export function DataModeSelector({
  currentMode,
  onModeSelect,
  className = "",
}: DataModeSelectorProps) {
  const modes: {
    id: DataMode;
    label: string;
    subLabel: string;
    badgeColor: string;
    isConfigured: boolean;
  }[] = [
    {
      id: "LIVE_NRT",
      label: "LIVE / NRT",
      subLabel: "IMD INSAT-3D & NASA GIBS",
      badgeColor: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
      isConfigured: true,
    },
    {
      id: "HISTORICAL",
      label: "HISTORICAL",
      subLabel: "IMD / IBTrACS Archives",
      badgeColor: "bg-purple-500/20 text-purple-400 border-purple-500/30",
      isConfigured: true,
    },
    {
      id: "USER_UPLOAD",
      label: "USER UPLOAD",
      subLabel: "Imported Files",
      badgeColor: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
      isConfigured: true,
    },
    {
      id: "DEMO",
      label: "DEMO BENCHMARK",
      subLabel: "Offline Benchmarks",
      badgeColor: "bg-blue-500/20 text-blue-400 border-blue-500/30",
      isConfigured: true,
    },
  ];

  return (
    <div
      className={`flex items-center gap-1.5 p-1 rounded-lg bg-white border border-slate-200 shadow-2xs ${className}`}
    >
      <div className="px-2 py-0.5 text-[10px] font-mono uppercase tracking-wider text-slate-500 border-r border-slate-200 flex items-center gap-1 font-semibold">
        <span className="w-1.5 h-1.5 rounded-full bg-blue-600 animate-pulse"></span>
        DATA MODE
      </div>

      <div className="flex items-center gap-1">
        {modes.map((m) => {
          const isActive = currentMode === m.id;
          return (
            <button
              key={m.id}
              onClick={() => {
                if (m.isConfigured && onModeSelect) {
                  onModeSelect(m.id);
                }
              }}
              title={
                !m.isConfigured
                  ? "Live satellite feeds require authentic MOSDAC / IMD API credentials. Zero-Fabrication policy active."
                  : `Switch active viewing mode to ${m.label}`
              }
              className={`relative px-2.5 py-1 rounded text-xs font-mono transition-all flex items-center gap-1.5 border ${
                isActive
                  ? `${m.badgeColor} font-bold shadow-2xs`
                  : "bg-slate-50 text-slate-600 border-slate-200 hover:text-slate-900 hover:bg-slate-100"
              } ${!m.isConfigured ? "opacity-75 cursor-not-allowed" : ""}`}
            >
              <span>{m.label}</span>
              {!m.isConfigured && (
                <span className="text-[9px] px-1 py-0.2 rounded bg-amber-50 text-amber-800 border border-amber-200 font-semibold">
                  NOT CONFIGURED
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

