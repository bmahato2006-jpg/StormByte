"use client";

import React from "react";
import { ShieldCheck, Satellite, AlertCircle, Compass } from "lucide-react";
import { CycloneSummary } from "@/types/cyclone";

interface TopStatusBarProps {
  cyclones?: CycloneSummary[];
  selectedId?: string;
  onSelectCyclone?: (id: string) => void;
  isBackendHealthy: boolean;
  dataMode?: "REAL" | "DEMO" | "NONE" | "LIVE";
  provenanceLabel?: string;
  sourceAgency?: string;
}

export const TopStatusBar: React.FC<TopStatusBarProps> = ({
  cyclones = [],
  selectedId = "",
  onSelectCyclone,
  isBackendHealthy,
  dataMode = "LIVE",
  provenanceLabel,
  sourceAgency,
}) => {
  const isLiveSatellite =
    dataMode === "LIVE" ||
    selectedId === "LIVE-NIO-SATELLITE" ||
    provenanceLabel?.includes("LIVE") ||
    !selectedId;

  const isRealHistorical =
    !isLiveSatellite &&
    (dataMode === "REAL" ||
      (dataMode !== "DEMO" &&
        (provenanceLabel?.includes("VERIFIED") ||
          provenanceLabel?.includes("AUTHENTIC") ||
          selectedId?.startsWith("IMD-") ||
          selectedId?.startsWith("IBTRACS-"))));

  const liveStorms = cyclones.filter(
    (c) =>
      c.id === "LIVE-NIO-SATELLITE" ||
      c.dataType === "LIVE SATELLITE STREAM" ||
      c.name.includes("LIVE")
  );

  const authenticStorms = cyclones.filter(
    (c) =>
      c.id !== "LIVE-NIO-SATELLITE" &&
      c.dataType !== "LIVE SATELLITE STREAM" &&
      !c.name.includes("LIVE") &&
      (c.id.startsWith("IMD-") ||
        c.id.startsWith("IBTRACS-") ||
        c.dataType === "VERIFIED HISTORICAL DATA" ||
        c.name.includes("AUTHENTIC"))
  );


  return (
    <div className="bg-white border-b border-slate-200 px-4 py-1.5 flex flex-wrap items-center justify-between gap-2.5 text-xs font-sans select-none shadow-2xs">
      {/* Left: Storm Selection Control */}
      <div className="flex items-center gap-2.5">
        {cyclones.length > 0 && onSelectCyclone && (
          <div className="flex items-center gap-2 bg-slate-50 border border-slate-300 rounded-lg px-2.5 py-1 shadow-2xs">
            <Satellite className="w-3.5 h-3.5 text-blue-600 shrink-0" />
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider font-mono">
              TARGET:
            </span>
            <select
              value={selectedId}
              onChange={(e) => onSelectCyclone(e.target.value)}
              className="bg-transparent border-0 text-slate-800 text-xs font-semibold focus:outline-none cursor-pointer max-w-[240px] sm:max-w-xs md:max-w-sm truncate font-sans"
            >
              {liveStorms.length > 0 && (
                <optgroup label="🛰 REAL-TIME LIVE SATELLITE FEEDS">
                  {liveStorms.map((c) => (
                    <option key={c.id} value={c.id} className="bg-white text-blue-800 font-bold">
                      {c.name}
                    </option>
                  ))}
                </optgroup>
              )}

              {authenticStorms.length > 0 && (
                <optgroup label="HISTORICAL BEST TRACK ARCHIVE (IMD / IBTrACS)">
                  {authenticStorms.map((c) => (
                    <option key={c.id} value={c.id} className="bg-white text-emerald-800 font-semibold">
                      {c.name} — {c.intensity}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
          </div>
        )}

        <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-50 border border-slate-200 text-[11px] text-slate-600 font-medium">
          <Compass className="w-3.5 h-3.5 text-slate-500" />
          <span>Basin:</span>
          <span className="text-slate-800 font-bold">North Indian Ocean (0-30°N)</span>
        </div>
      </div>

      {/* Right: Operational Status Badges */}
      <div className="flex items-center gap-2">
        {/* Stream Status Badge */}
        {isLiveSatellite ? (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800 text-[11px] font-bold shadow-2xs">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 animate-pulse" />
            <span>NRT SATELLITE (INSAT-3D &amp; NASA)</span>
          </div>
        ) : (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-blue-50 border border-blue-200 text-blue-800 text-[11px] font-bold shadow-2xs">
            <ShieldCheck className="w-3.5 h-3.5 text-blue-600" />
            <span>VERIFIED BEST TRACK (IMD)</span>
          </div>
        )}
      </div>
    </div>
  );
};
