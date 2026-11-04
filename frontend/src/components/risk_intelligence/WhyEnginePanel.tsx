"use client";

import React from "react";
import { BrainCircuit, CheckCircle2, AlertTriangle, Info } from "lucide-react";
import { WhyExplanation } from "@/types/impact";

interface WhyEnginePanelProps {
  whyExplanation: WhyExplanation | null;
}

export const WhyEnginePanel: React.FC<WhyEnginePanelProps> = ({ whyExplanation }) => {
  if (!whyExplanation) return null;

  return (
    <div className="p-5 rounded-xl border border-slate-200 bg-white shadow-xs font-mono space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-200">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-blue-50 border border-blue-200 flex items-center justify-center">
            <BrainCircuit className="w-4 h-4 text-blue-700 animate-pulse" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-900 tracking-wide uppercase">
              AI RISK FACTOR ATTRIBUTION ENGINE
            </h3>
            <span className="text-[10px] text-slate-500 font-semibold">Deterministic Multi-Hazard Causality Analysis</span>
          </div>
        </div>

        <span className="text-[10px] px-2.5 py-1 rounded-full border border-blue-200 bg-blue-50 text-blue-800 font-bold">
          {whyExplanation.engineType}
        </span>
      </div>

      {/* Primary Headline */}
      <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-800 leading-relaxed">
        <span className="font-bold text-blue-700 mr-1">EXPLANATION SYNTHESIS:</span>
        {whyExplanation.headline}
      </div>

      {/* Factor Breakdown Grid */}
      <div className="space-y-2">
        <div className="text-[11px] font-bold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
          <span>Contributing Hazard Factors</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
          {whyExplanation.factorBreakdown.map((factor, idx) => (
            <div
              key={idx}
              className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-700 flex items-start gap-2"
            >
              <span className="text-blue-700 font-bold mt-0.5">•</span>
              <span>{factor}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Primary Driving Hazards */}
      {whyExplanation.primaryDrivers && whyExplanation.primaryDrivers.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 pt-2">
          <span className="text-[11px] text-slate-600 font-bold uppercase">PRIMARY DRIVERS:</span>
          {whyExplanation.primaryDrivers.map((driver, idx) => (
            <span
              key={idx}
              className="px-2 py-0.5 rounded text-[10px] bg-red-50 text-red-700 border border-red-200 font-bold"
            >
              {driver}
            </span>
          ))}
        </div>
      )}

      {/* Limitations Notice */}
      <div className="p-3 rounded-lg bg-amber-50 border border-amber-200 text-[10px] text-amber-900 flex items-start gap-2">
        <Info className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
        <div className="space-y-0.5">
          <div className="font-bold text-amber-800">SCIENTIFIC BOUNDARIES &amp; LIMITATIONS</div>
          <ul className="list-disc list-inside space-y-0.5 text-amber-800/90">
            {whyExplanation.limitations.map((lim, idx) => (
              <li key={idx}>{lim}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

