"use client";

import React from "react";
import { Cpu, AlertTriangle, Calendar, Layers, ShieldCheck, ArrowRight } from "lucide-react";

interface ModulePlaceholderProps {
  title: string;
  moduleNumber: string;
  targetPhase: string;
  description: string;
  plannedFeatures: string[];
  scientificIntegrityNote: string;
}

export const ModulePlaceholder: React.FC<ModulePlaceholderProps> = ({
  title,
  moduleNumber,
  targetPhase,
  description,
  plannedFeatures,
  scientificIntegrityNote,
}) => {
  return (
    <div className="p-8 rounded-xl border border-slate-200 bg-white shadow-md max-w-4xl mx-auto space-y-6 font-mono">
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-slate-200 pb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-blue-50 border border-blue-200 text-blue-700 uppercase">
              MODULE {moduleNumber}
            </span>
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-50 border border-amber-200 text-amber-700">
              SCHEDULED FOR {targetPhase}
            </span>
          </div>
          <h2 className="text-xl font-black text-slate-900 uppercase tracking-wider">{title}</h2>
          <p className="text-xs text-slate-500 mt-1">{description}</p>
        </div>

        {/* Model State Pill */}
        <div className="p-2.5 rounded-lg bg-amber-50 border border-amber-200 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-amber-600 animate-pulse" />
          <div className="text-right">
            <div className="text-[10px] text-amber-700 font-semibold uppercase">AI ENGINE STATE</div>
            <div className="text-xs font-bold text-amber-800">MODEL NOT TRAINED</div>
          </div>
        </div>
      </div>

      {/* Planned Technical Architecture */}
      <div className="space-y-3">
        <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2">
          <Layers className="w-4 h-4 text-blue-600" />
          <span>PLANNED ENGINE CAPABILITIES ({targetPhase})</span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
          {plannedFeatures.map((feat, idx) => (
            <div
              key={idx}
              className="p-3 rounded-lg bg-slate-50 border border-slate-200 flex items-center gap-2.5 text-xs text-slate-700"
            >
              <ArrowRight className="w-3.5 h-3.5 text-blue-600 shrink-0" />
              <span>{feat}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Scientific Integrity Callout */}
      <div className="p-4 rounded-lg bg-amber-50 border border-amber-200 flex items-start gap-3 text-xs text-amber-900">
        <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
        <div>
          <span className="font-bold text-amber-800 uppercase">Scientific Integrity &amp; Ethics Notice:</span>
          <div className="mt-0.5 leading-relaxed text-amber-800">{scientificIntegrityNote}</div>
        </div>
      </div>

      {/* Footer Meta */}
      <div className="pt-2 border-t border-slate-200 flex items-center justify-between text-[11px] text-slate-500">
        <span>STATUS: ARCHITECTURE PREPARED • AWAITING PHASE IMPLEMENTATION</span>
        <span className="text-slate-400">CYCLONE-X PLATFORM FOUNDATION</span>
      </div>
    </div>
  );
};

