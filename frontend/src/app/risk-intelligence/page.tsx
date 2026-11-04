"use client";

import React from "react";
import Link from "next/link";
import { ArrowLeft, Radio } from "lucide-react";
import { RiskIntelligenceView } from "@/components/risk_intelligence/RiskIntelligenceView";

export default function RiskIntelligencePage() {
  return (
    <div className="min-h-screen font-mono p-4 md:p-6 space-y-5 select-none text-slate-800 bg-[linear-gradient(90deg,#93c5fd_0%,#bfdbfe_14%,#dbeafe_26%,#eff6ff_38%,#ffffff_48%,#ffffff_52%,#eff6ff_62%,#dbeafe_74%,#bfdbfe_86%,#93c5fd_100%)]">
      {/* Top Header & Navigation */}
      <header className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-50 border border-blue-200 flex items-center justify-center shadow-xs">
            <Radio className="w-5 h-5 text-blue-700 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-black text-lg tracking-wider text-slate-900">CYCLONE</span>
              <span className="font-black text-lg tracking-wider text-blue-700">-X</span>
              <span className="text-xs px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200 font-bold ml-2">
                OPERATIONAL INTELLIGENCE
              </span>
            </div>
            <span className="text-xs text-slate-500 block font-medium">
              Multi-Hazard Impact &amp; Disaster Decision-Support Platform
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-white border border-slate-300 text-xs text-slate-700 hover:text-slate-900 hover:border-blue-500 transition-all shadow-2xs font-semibold"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>OPERATIONS DASHBOARD</span>
          </Link>
        </div>
      </header>

      {/* Master Risk Intelligence Container */}
      <main className="max-w-[1720px] mx-auto">
        <RiskIntelligenceView initialStormId="LIVE-NIO-SATELLITE" />
      </main>
    </div>
  );
}

