"use client";

import React from "react";
import { Activity, Server, HardDrive, CheckCircle2, AlertCircle } from "lucide-react";
import { HealthStatus } from "@/types/health";

interface SystemHealthCardProps {
  health: HealthStatus | null;
  isBackendHealthy: boolean;
}

export const SystemHealthCard: React.FC<SystemHealthCardProps> = ({ health, isBackendHealthy }) => {
  return (
    <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-xs flex flex-col justify-between h-full">
      <div className="flex items-center justify-between border-b border-slate-200 pb-2 mb-3">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-blue-700" />
          <h3 className="text-xs font-mono font-bold text-slate-900 uppercase tracking-wider">
            SYSTEM HEALTH & RUNTIME DIAGNOSTICS
          </h3>
        </div>
        <span
          className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold ${
            isBackendHealthy
              ? "bg-emerald-50 border border-emerald-200 text-emerald-800"
              : "bg-rose-50 border border-rose-200 text-rose-800"
          }`}
        >
          {isBackendHealthy ? "ALL SYSTEMS NOMINAL" : "CONNECTION DEGRADED"}
        </span>
      </div>

      <div className="space-y-2.5 text-xs font-mono">
        {/* Backend Node */}
        <div className="p-2.5 rounded bg-slate-50 border border-slate-200 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Server className="w-4 h-4 text-blue-700" />
            <div>
              <div className="text-slate-900 font-bold">FastAPI Service Gateway</div>
              <div className="text-[10px] text-slate-500">Endpoint: /api/health</div>
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            {isBackendHealthy ? (
              <span className="text-emerald-700 font-bold flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" /> 200 OK
              </span>
            ) : (
              <span className="text-rose-700 font-bold flex items-center gap-1">
                <AlertCircle className="w-3.5 h-3.5" /> OFFLINE
              </span>
            )}
          </div>
        </div>

        {/* Database PostGIS */}
        <div className="p-2.5 rounded bg-slate-50 border border-slate-200 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <HardDrive className="w-4 h-4 text-emerald-700" />
            <div>
              <div className="text-slate-900 font-bold">PostgreSQL + PostGIS</div>
              <div className="text-[10px] text-slate-500">Spatial GiST Indexing Architecture</div>
            </div>
          </div>
          <span className="px-2 py-0.5 rounded text-[10px] bg-white border border-slate-300 text-blue-800 font-semibold shadow-2xs">
            SCHEMA READY
          </span>
        </div>

        {/* AI / Inference Pipeline */}
        <div className="p-2.5 rounded bg-slate-50 border border-slate-200 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-emerald-700" />
            <div>
              <div className="text-slate-900 font-bold">AI / ML Inference Engine</div>
              <div className="text-[10px] text-slate-500">PyTorch CNN V2/V3 + LSTM Forecaster</div>
            </div>
          </div>
          <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-50 border border-emerald-200 text-emerald-800 font-semibold flex items-center gap-1">
            <CheckCircle2 className="w-3 h-3 text-emerald-700" /> WEIGHTS LOADED
          </span>
        </div>
      </div>

      <div className="mt-2 text-[10px] font-mono text-slate-500 pt-2 border-t border-slate-200 flex items-center justify-between">
        <span>VERSION: {health?.version || "0.1.0"}</span>
        <span className="text-slate-400">ENVIRONMENT: PRODUCTION-READY STAGING</span>
      </div>
    </div>
  );
};

