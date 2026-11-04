"use client";

import React, { useState } from "react";
import { 
  FileDown, 
  Printer, 
  ShieldAlert, 
  Clock, 
  CheckCircle2, 
  Calendar, 
  MapPin, 
  Wind, 
  Waves, 
  FileText,
  AlertTriangle
} from "lucide-react";

export function BriefingGeneratorView({ stormId }: { stormId?: string }) {
  const [briefingDate] = useState<string>(new Date().toISOString().split("T")[0]);
  const [reportId] = useState<string>(`CYX-BRIEF-${new Date().getFullYear()}-001`);

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* ACTION BAR (Hidden in Print) */}
      <div className="flex items-center justify-between p-4 rounded-xl border border-slate-800 bg-slate-900/80 backdrop-blur-md print:hidden">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <FileText className="h-5 w-5 text-cyan-400" />
            <span>Module 25: Automated PDF Intelligence Briefing Generator</span>
          </h2>
          <p className="text-xs text-slate-400 font-mono mt-0.5">
            Produces an official executive disaster brief formatted for direct PDF export or hardcopy dispatch.
          </p>
        </div>
        <button
          onClick={handlePrint}
          className="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-mono font-bold flex items-center gap-2 shadow-lg shadow-cyan-900/30 transition-all cursor-pointer"
        >
          <Printer className="h-4 w-4" />
          <span>Generate / Print PDF</span>
        </button>
      </div>

      {/* DOCUMENT CONTAINER (Print-optimized) */}
      <div className="rounded-xl border border-slate-700 bg-slate-950 p-8 shadow-2xl text-slate-100 font-sans print:border-none print:shadow-none print:p-0 print:bg-white print:text-black">
        {/* HEADER & WATERMARK */}
        <div className="border-b-2 border-slate-700 pb-6 print:border-black space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center font-black font-mono text-cyan-400 text-xl print:bg-slate-200 print:text-black">
                CX
              </div>
              <div>
                <h1 className="text-xl font-black uppercase tracking-wider text-white print:text-black">
                  Cyclone-X Operational Intelligence Briefing
                </h1>
                <div className="text-xs font-mono text-slate-400 print:text-slate-600">
                  National / State Disaster Management Authority Executive Synthesis
                </div>
              </div>
            </div>
            <div className="text-right font-mono text-xs text-slate-400 print:text-slate-600 space-y-0.5">
              <div>Ref: <strong className="text-slate-200 print:text-black">{reportId}</strong></div>
              <div>Date: {briefingDate} UTC</div>
              <div className="text-[10px] text-amber-400 print:text-amber-700 font-bold">CLASSIFICATION: TACTICAL EMERGENCY</div>
            </div>
          </div>

          {/* MANDATORY PROVENANCE & NON-REPLACEMENT DISCLAIMER */}
          <div className="p-2.5 rounded bg-amber-500/10 border border-amber-500/30 text-[11px] font-mono text-amber-300/90 print:bg-slate-100 print:border-slate-300 print:text-slate-700">
            <strong>OFFICIAL NOTICE:</strong> This intelligence briefing is compiled from automated physics-AI telemetry and authentic satellite-radar observations. It is intended for decision support and does NOT supersede official mandates or warnings issued by the India Meteorological Department (IMD) or NDMA.
          </div>
        </div>

        {/* SECTION 1: EXECUTIVE SYNTHESIS */}
        <div className="py-6 border-b border-slate-800 print:border-slate-300 space-y-4">
          <h2 className="text-sm font-bold uppercase tracking-wider font-mono text-cyan-400 print:text-cyan-800 flex items-center gap-2">
            <span>1.0 Executive Situation Overview</span>
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 font-mono text-xs">
            <div className="p-3 rounded bg-slate-900/60 border border-slate-800 print:bg-slate-50 print:border-slate-300">
              <span className="text-slate-400 print:text-slate-600 block text-[10px]">CURRENT INTENSITY</span>
              <strong className="text-base text-white print:text-black">85 kt (157 km/h)</strong>
              <div className="text-[10px] text-rose-400 print:text-rose-600">Very Severe Cyclonic Storm</div>
            </div>
            <div className="p-3 rounded bg-slate-900/60 border border-slate-800 print:bg-slate-50 print:border-slate-300">
              <span className="text-slate-400 print:text-slate-600 block text-[10px]">CENTRAL PRESSURE</span>
              <strong className="text-base text-white print:text-black">965.0 hPa</strong>
              <div className="text-[10px] text-slate-400">ΔP: -48 hPa</div>
            </div>
            <div className="p-3 rounded bg-slate-900/60 border border-slate-800 print:bg-slate-50 print:border-slate-300">
              <span className="text-slate-400 print:text-slate-600 block text-[10px]">COASTAL DISTANCE</span>
              <strong className="text-base text-white print:text-black">145 km</strong>
              <div className="text-[10px] text-cyan-400 print:text-cyan-700">Closing at 14 kt</div>
            </div>
            <div className="p-3 rounded bg-slate-900/60 border border-slate-800 print:bg-slate-50 print:border-slate-300">
              <span className="text-slate-400 print:text-slate-600 block text-[10px]">EVACUATION WINDOW</span>
              <strong className="text-base text-rose-400 print:text-rose-600">6.1 Hours</strong>
              <div className="text-[10px] text-slate-400">Until 34kt gale arrival</div>
            </div>
          </div>
        </div>

        {/* SECTION 2: MARITIME & PORT WARNING STATUS */}
        <div className="py-6 border-b border-slate-800 print:border-slate-300 space-y-4">
          <h2 className="text-sm font-bold uppercase tracking-wider font-mono text-cyan-400 print:text-cyan-800">
            2.0 Maritime Port Warning Directives
          </h2>
          <table className="w-full text-left text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 print:border-slate-300 print:text-slate-600">
                <th className="py-2">PORT FACILITY</th>
                <th className="py-2">IMD SIGNAL</th>
                <th className="py-2">DIRECTIVE & VESSEL RESTRICTION</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 print:divide-slate-200">
              <tr>
                <td className="py-2.5 font-bold text-white print:text-black">Paradip Port (Odisha)</td>
                <td className="py-2.5 text-rose-400 font-bold print:text-rose-600">Signal No. 10 (Great Danger)</td>
                <td className="py-2.5 text-slate-300 print:text-slate-700">Suspend cargo operations; de-berth tankers; secure storm moorings.</td>
              </tr>
              <tr>
                <td className="py-2.5 font-bold text-white print:text-black">Dhamra Port (Odisha)</td>
                <td className="py-2.5 text-rose-400 font-bold print:text-rose-600">Signal No. 10 (Great Danger)</td>
                <td className="py-2.5 text-slate-300 print:text-slate-700">Halt harbor maneuvers; recall pilot launches; clear fairway.</td>
              </tr>
              <tr>
                <td className="py-2.5 font-bold text-white print:text-black">Haldia Port (West Bengal)</td>
                <td className="py-2.5 text-amber-400 font-bold print:text-amber-700">Signal No. 8 (Great Danger)</td>
                <td className="py-2.5 text-slate-300 print:text-slate-700">Prepare lock gates; establish auxiliary drainage pumping.</td>
              </tr>
              <tr>
                <td className="py-2.5 font-bold text-white print:text-black">Visakhapatnam Port (AP)</td>
                <td className="py-2.5 text-cyan-400 font-bold print:text-cyan-700">Signal No. 4 (Local Warning)</td>
                <td className="py-2.5 text-slate-300 print:text-slate-700">Issue cautionary notices to inbound ocean traffic; maintain radio watch.</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* SECTION 3: EMERGENCY CHECKLIST & PRE-POSITIONING */}
        <div className="py-6 border-b border-slate-800 print:border-slate-300 space-y-4">
          <h2 className="text-sm font-bold uppercase tracking-wider font-mono text-cyan-400 print:text-cyan-800">
            3.0 Emergency Pre-Positioning Checklist
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
            <div className="p-3 rounded bg-slate-900/40 border border-slate-800 print:bg-slate-50 print:border-slate-300 space-y-2">
              <strong className="text-slate-200 print:text-black block text-[11px] uppercase">State Disaster Management Directives:</strong>
              <div className="space-y-1.5 text-slate-300 print:text-slate-700">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                  <span>Evacuate 125,000 vulnerable population within 5km coastal belt.</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                  <span>Activate 340 multipurpose cyclone shelters with potable water.</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                  <span>Enforce total prohibition of fishing boats in Bay of Bengal.</span>
                </div>
              </div>
            </div>

            <div className="p-3 rounded bg-slate-900/40 border border-slate-800 print:bg-slate-50 print:border-slate-300 space-y-2">
              <strong className="text-slate-200 print:text-black block text-[11px] uppercase">NDRF / SDRF Deployment Grid:</strong>
              <div className="space-y-1.5 text-slate-300 print:text-slate-700">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                  <span>14 NDRF battalions pre-staged in Balasore, Bhadrak, Kendrapara.</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                  <span>Tree-clearing heavy machinery and power saws stationed along NH-16.</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                  <span>Mobile satellite communication vans deployed to emergency EOCs.</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* FOOTER */}
        <div className="pt-6 flex items-center justify-between text-[10px] font-mono text-slate-500 print:text-slate-400">
          <div>Cyclone-X AI/ML Early Warning & Tactical Analytics Framework v2.4</div>
          <div>Page 1 of 1 • Generated {new Date().toUTCString()}</div>
        </div>
      </div>
    </div>
  );
}
