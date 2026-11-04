"use client";

import React, { useEffect, useState } from "react";
import { Cpu, Compass, Waves, AlertTriangle, ShieldCheck, Wind } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { DigitalTwinResponse } from "@/types/digital_twin";
import { fetchDigitalTwin } from "@/lib/api";

interface DigitalTwinPanelProps {
  stormId: string;
}

export const DigitalTwinPanel: React.FC<DigitalTwinPanelProps> = ({ stormId }) => {
  const [digitalTwin, setDigitalTwin] = useState<DigitalTwinResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    if (!stormId) return;
    setLoading(true);
    fetchDigitalTwin(stormId)
      .then((data) => setDigitalTwin(data))
      .catch((err) => console.warn("Digital twin fetch fallback:", err))
      .finally(() => setLoading(false));
  }, [stormId]);

  if (loading || !digitalTwin) {
    return (
      <div className="p-6 rounded-xl border border-command-cardBorder bg-command-card text-center font-mono text-xs text-slate-400">
        INITIALIZING DIGITAL TWIN VORTEX SOLVER...
      </div>
    );
  }

  const quads = digitalTwin.quadrantRadii;

  return (
    <div className="p-5 rounded-xl border border-slate-200 bg-white shadow-xs space-y-5">
      {/* Panel Header */}
      <div className="flex items-center justify-between border-b border-slate-200 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-blue-50 border border-blue-200">
            <Cpu className="w-5 h-5 text-blue-700" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-mono font-black text-slate-900 uppercase tracking-wider">
                CYCLONE DIGITAL TWIN — VORTEX PHYSICS &amp; DYNAMICS
              </h3>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-blue-50 border border-blue-200 text-blue-800 font-bold">
                HOLLAND (1980) VORTEX
              </span>
            </div>
            <p className="text-xs text-slate-500 font-mono mt-0.5">
              Continuum state modeling of tangential wind velocity fields and coastal storm surge stresses
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-2.5 py-1 rounded text-xs font-mono font-bold bg-blue-50 border border-blue-200 text-blue-800">
            PHYSICS VORTEX MODEL
          </span>
        </div>
      </div>

      {/* Grid of Core Parameters & Holland Radial Curve */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left: Vortex Parameters & Quadrant Compass Table */}
        <div className="lg:col-span-6 space-y-4 font-mono text-xs">
          {/* Key Vortex Telemetry Cards */}
          <div className="grid grid-cols-3 gap-3">
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <div className="text-[10px] text-slate-500 uppercase font-semibold">Radius of Max Wind</div>
              <div className="text-xl font-black text-blue-700 mt-1">{digitalTwin.radiusMaxWindKm} km</div>
              <div className="text-[10px] text-slate-500">Eyewall locus (Rmax)</div>
            </div>

            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <div className="text-[10px] text-slate-500 uppercase font-semibold">Holland Shape (B)</div>
              <div className="text-xl font-black text-amber-700 mt-1">{digitalTwin.hollandB}</div>
              <div className="text-[10px] text-slate-500">Pressure gradient scale</div>
            </div>

            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <div className="text-[10px] text-slate-500 uppercase font-semibold">Coriolis (f)</div>
              <div className="text-xl font-black text-sky-700 mt-1">
                {(digitalTwin.coriolisF * 100000).toFixed(2)}e-5
              </div>
              <div className="text-[10px] text-slate-500">rad/s at latitude</div>
            </div>
          </div>

          {/* Quadrant Wind Radii Table */}
          <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
            <div className="flex items-center justify-between border-b border-slate-200 pb-1.5">
              <div className="flex items-center gap-1.5 font-bold text-slate-800">
                <Compass className="w-4 h-4 text-blue-700" />
                <span>QUADRANT WIND RADII (R34 / R50 / R64)</span>
              </div>
              <span className="text-[10px] text-slate-500 font-semibold">RADII IN KM</span>
            </div>

            <table className="w-full text-left text-[11px] font-mono">
              <thead>
                <tr className="text-slate-500 border-b border-slate-200">
                  <th className="py-1">THRESHOLD</th>
                  <th className="py-1 text-right text-rose-700 font-bold">NE (R-FRONT)</th>
                  <th className="py-1 text-right text-slate-700 font-medium">SE</th>
                  <th className="py-1 text-right text-slate-700 font-medium">SW</th>
                  <th className="py-1 text-right text-slate-700 font-medium">NW</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                <tr>
                  <td className="py-1.5 font-bold text-blue-700">R34 (34 kt / Gale)</td>
                  <td className="py-1.5 text-right font-bold text-rose-700">{quads.R34.NE} km</td>
                  <td className="py-1.5 text-right text-slate-700">{quads.R34.SE} km</td>
                  <td className="py-1.5 text-right text-slate-700">{quads.R34.SW} km</td>
                  <td className="py-1.5 text-right text-slate-700">{quads.R34.NW} km</td>
                </tr>
                <tr>
                  <td className="py-1.5 font-bold text-amber-700">R50 (50 kt / Storm)</td>
                  <td className="py-1.5 text-right font-bold text-rose-700">{quads.R50.NE} km</td>
                  <td className="py-1.5 text-right text-slate-700">{quads.R50.SE} km</td>
                  <td className="py-1.5 text-right text-slate-700">{quads.R50.SW} km</td>
                  <td className="py-1.5 text-right text-slate-700">{quads.R50.NW} km</td>
                </tr>
                <tr>
                  <td className="py-1.5 font-bold text-red-700">R64 (64 kt / Hurricane)</td>
                  <td className="py-1.5 text-right font-bold text-rose-700">{quads.R64.NE} km</td>
                  <td className="py-1.5 text-right text-slate-700">{quads.R64.SE} km</td>
                  <td className="py-1.5 text-right text-slate-700">{quads.R64.SW} km</td>
                  <td className="py-1.5 text-right text-slate-700">{quads.R64.NW} km</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Sector Hazard Banner */}
          <div className="p-3 rounded-lg bg-red-50 border border-red-200 flex items-start gap-2.5 text-red-900">
            <AlertTriangle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
            <div>
              <div className="font-bold text-xs uppercase tracking-wide">
                MAXIMUM ASYMMETRIC SECTOR RISK:
              </div>
              <div className="text-[11px] text-red-800 mt-0.5">{digitalTwin.rightFrontQuadrantRisk}</div>
            </div>
          </div>
        </div>

        {/* Right: Radial Wind Profile Chart V(r) */}
        <div className="lg:col-span-6 flex flex-col justify-between font-mono">
          <div className="p-4 rounded-lg bg-white border border-slate-200 shadow-2xs">
            <div className="flex items-center justify-between border-b border-slate-200 pb-2 mb-2">
              <span className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                <Wind className="w-3.5 h-3.5 text-blue-700" />
                RADIAL WIND VELOCITY PROFILE: V(r) vs RADIUS
              </span>
              <span className="text-[10px] text-slate-500 font-semibold">HOLLAND EQUATION</span>
            </div>

            <div className="w-full h-52">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={digitalTwin.radialProfile}
                  margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis
                    dataKey="radiusKm"
                    stroke="#94a3b8"
                    tick={{ fontSize: 10, fontFamily: "monospace", fill: "#475569" }}
                    unit="km"
                  />
                  <YAxis
                    stroke="#0284c7"
                    tick={{ fontSize: 10, fontFamily: "monospace", fill: "#0284c7" }}
                    unit="kt"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#ffffff",
                      borderColor: "#cbd5e1",
                      fontFamily: "monospace",
                      fontSize: "11px",
                      color: "#0f172a",
                      boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
                    }}
                    formatter={(val: any) => [`${val} knots`, "Wind Velocity"]}
                    labelFormatter={(label: any) => `Distance: ${label} km from eye`}
                  />
                  <Line
                    type="monotone"
                    dataKey="windSpeedKnots"
                    stroke="#0284c7"
                    strokeWidth={2.5}
                    dot={{ r: 2, fill: "#0284c7" }}
                    name="Tangential Velocity V(r)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="flex items-center justify-between text-[10px] text-slate-500 pt-2 border-t border-slate-200 mt-1">
              <span className="text-blue-700 font-bold">PEAK AT RMAX: {digitalTwin.radiusMaxWindKm} km</span>
              <span className="text-slate-500">ASYMPTOTIC RETURN TO AMBIENT FLOW</span>
            </div>
          </div>

          <div className="mt-3 p-3 rounded-lg bg-slate-50 border border-slate-200 text-[11px] text-slate-600">
            <span className="text-slate-900 font-bold">Digital Twin Continuity:</span> Enables high-resolution hydrodynamic
            surge modeling and district-level damage projections between 6-hour observation fixes.
          </div>
        </div>
      </div>
    </div>
  );
};

