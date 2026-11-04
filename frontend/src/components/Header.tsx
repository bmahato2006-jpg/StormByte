"use client";

import React, { useEffect, useState } from "react";
import { Activity, AlertTriangle, ShieldCheck, Radio, Satellite } from "lucide-react";
import { SCIENTIFIC_DISCLAIMER } from "@/lib/constants";
import { CycloneSummary } from "@/types/cyclone";

interface HeaderProps {
  cyclones: CycloneSummary[];
  selectedId: string;
  onSelectCyclone: (id: string) => void;
  isBackendHealthy: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  cyclones,
  selectedId,
  onSelectCyclone,
  isBackendHealthy,
}) => {
  const [currentTime, setCurrentTime] = useState<string>("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setCurrentTime(now.toUTCString().replace("GMT", "UTC"));
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="border-b border-slate-200 bg-white/95 backdrop-blur-md shadow-xs sticky top-0 z-40">
      {/* Top Advisory Disclaimer Banner */}
      <div className="bg-amber-50 border-b border-amber-200 px-4 py-1.5 flex items-center justify-between text-xs text-amber-900 font-mono">
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-3.5 h-3.5 text-amber-700 shrink-0" />
          <span className="tracking-tight font-semibold uppercase">
            {SCIENTIFIC_DISCLAIMER}
          </span>
        </div>
        <div className="hidden md:flex items-center gap-2 font-mono text-[11px] text-emerald-700 font-bold">
          <span>OPERATIONAL LEVEL: ACTIVE SATELLITE MONITORING</span>
        </div>
      </div>

      {/* Main Command Bar */}
      <div className="max-w-[1920px] mx-auto px-4 py-2.5 flex flex-wrap items-center justify-between gap-4">
        {/* Brand & Mission */}
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-lg bg-blue-50 border border-blue-200 shadow-xs">
            <Radio className="w-5 h-5 text-blue-600 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-black tracking-widest text-slate-900 uppercase flex items-center">
                CYCLONE<span className="text-blue-600">-X</span>
              </h1>
              <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold bg-slate-100 border border-slate-200 text-slate-600">
                v0.1.0-FOUNDATION
              </span>
            </div>
            <p className="text-xs text-slate-600 font-medium">
              AI-Powered Tropical Cyclone Intelligence & Disaster Decision Support
            </p>
          </div>
        </div>

        {/* Center: Cyclone Selector */}
        <div className="flex items-center gap-3 bg-white border border-slate-200 rounded-lg px-3 py-1.5 shadow-xs">
          <Satellite className="w-4 h-4 text-blue-600" />
          <span className="text-xs font-mono text-slate-600 uppercase font-semibold">Target Storm:</span>
          <select
            value={selectedId}
            onChange={(e) => onSelectCyclone(e.target.value)}
            className="bg-slate-50 border border-slate-300 text-slate-900 text-xs font-mono font-semibold rounded px-2.5 py-1 focus:outline-none focus:border-blue-500 transition-colors cursor-pointer"
          >
            {cyclones.map((c) => (
              <option key={c.id} value={c.id} className="bg-white text-slate-900">
                {c.name} ({c.id}) — {c.intensity}
              </option>
            ))}
          </select>
        </div>

        {/* Right: Telemetry Time, Health, and Demo Mode Badge */}
        <div className="flex items-center gap-3">
          {/* UTC Clock */}
          <div className="hidden lg:flex flex-col items-end px-3 py-1 bg-white border border-slate-200 rounded shadow-xs">
            <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">System UTC Clock</span>
            <span className="text-xs font-mono font-bold text-blue-700">{currentTime || "SYNCING..."}</span>
          </div>

          {/* Backend Status */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-200 bg-white shadow-xs">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                isBackendHealthy ? "bg-emerald-500 animate-pulse shadow-xs" : "bg-red-500 shadow-xs"
              }`}
            />
            <span className="text-xs font-mono font-medium text-slate-700">
              API {isBackendHealthy ? "ONLINE" : "DISCONNECTED"}
            </span>
          </div>

          {/* Operational / Real Mode Badge */}
          {process.env.NEXT_PUBLIC_DEMO_MODE === "true" ? (
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-50 border border-amber-300 shadow-xs">
              <ShieldCheck className="w-4 h-4 text-amber-700" />
              <div className="flex flex-col">
                <span className="text-[11px] font-mono font-black text-amber-900 tracking-wider">DEMO MODE</span>
                <span className="text-[9px] font-mono text-amber-700 font-semibold -mt-0.5">HISTORICAL BENCHMARK</span>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-50 border border-emerald-300 shadow-xs">
              <ShieldCheck className="w-4 h-4 text-emerald-700" />
              <div className="flex flex-col">
                <span className="text-[11px] font-mono font-black text-emerald-900 tracking-wider">REAL OPERATIONAL</span>
                <span className="text-[9px] font-mono text-emerald-700 font-semibold -mt-0.5">NRT LIVE TELEMETRY</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

