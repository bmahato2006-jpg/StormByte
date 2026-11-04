"use client";

import React, { useState, useEffect } from "react";
import { Cpu, Compass, Wind, Layers, Activity, Info, RefreshCw, Sliders, AlertTriangle } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { HollandVortexProfile } from "@/types/prediction";
import { fetchCycloneVortexProfile } from "@/lib/api";

interface HollandVortexViewerProps {
  stormId: string;
}

export function HollandVortexViewer({ stormId }: HollandVortexViewerProps) {
  const [profile, setProfile] = useState<HollandVortexProfile | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Theoretical Sensitivity Modifiers
  const [simPressureDeficit, setSimPressureDeficit] = useState<number | null>(null);

  useEffect(() => {
    if (!stormId) return;
    setLoading(true);
    setErrorMsg(null);
    fetchCycloneVortexProfile(stormId)
      .then((data) => {
        setProfile(data);
        setSimPressureDeficit(data.central_pressure_deficit_hpa);
      })
      .catch((err) => {
        setErrorMsg(err.message || "Failed to load vortex profile");
      })
      .finally(() => setLoading(false));
  }, [stormId]);

  if (loading) {
    return (
      <div className="p-8 rounded-2xl bg-white border border-slate-200 text-center font-mono text-xs text-blue-700 shadow-sm flex items-center justify-center gap-3">
        <RefreshCw className="w-4 h-4 animate-spin text-blue-600" />
        <span>SOLVING HOLLAND (1980) NON-LINEAR VORTEX EQUATIONS...</span>
      </div>
    );
  }

  if (errorMsg || !profile) {
    return (
      <div className="p-6 rounded-2xl bg-rose-50 border border-rose-200 text-rose-800 text-xs font-mono shadow-xs">
        Error loading physical vortex model: {errorMsg}
      </div>
    );
  }

  const quads = profile.quadrant_radii;

  return (
    <div className="flex flex-col gap-5 w-full max-w-7xl mx-auto p-2">
      {/* Top Header Card */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 bg-white p-5 rounded-2xl border border-slate-200 shadow-xs">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-blue-600 animate-pulse"></span>
            <h2 className="text-lg font-bold tracking-wider text-slate-900 uppercase font-mono">
              Holland (1980) Vortex State Engine
            </h2>
          </div>
          <p className="text-xs text-slate-500 mt-1 font-mono">
            Analytical gradient wind profile $V(r)$ and multi-quadrant gale boundary estimation with translation asymmetry
          </p>
        </div>

        <div className="flex items-center gap-2 bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-mono">
          <span className="text-slate-500">Solver:</span>
          <span className="text-blue-700 font-bold">holland_vortex_v1</span>
          <span className="px-1.5 py-0.5 rounded text-[10px] bg-emerald-50 text-emerald-800 border border-emerald-200 font-bold">
            ANALYTIC BASELINE v0.5.0
          </span>
        </div>
      </div>

      {/* Card-Level Scientific Advisory */}
      <div className="p-3.5 rounded-xl bg-amber-50 border border-amber-200 font-mono text-xs text-amber-900 flex items-start gap-3 shadow-2xs">
        <AlertTriangle className="w-5 h-5 text-amber-700 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <div className="font-bold text-amber-800 tracking-wide uppercase flex items-center gap-2">
            <span>MATHEMATICAL VORTEX PROFILE — NOT AN OFFICIAL IMD ADVISORY</span>
          </div>
          <p className="text-[11px] text-amber-800/90 leading-relaxed">
            The wind speed curve $V(r)$, core eyewall boundary (RMW), and quadrant gale radii (R34, R50, R64) displayed here are model-derived analytic estimates computed via the Holland (1980) vortex state equations. They do not constitute official warning or advisory wind radii issued by the India Meteorological Department (IMD) or RSMC New Delhi.
          </p>
        </div>
      </div>

      {/* Physics Parameter Highlights */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-white border border-slate-200 font-mono shadow-xs">
          <span className="text-[10px] text-slate-500 block uppercase font-semibold">Radius of Max Wind (RMW)</span>
          <span className="text-xl font-bold text-blue-700 mt-1 block">
            {profile.radius_max_wind_km} <span className="text-xs text-slate-500">km</span>
          </span>
          <span className="text-[10px] text-slate-500">Core Eyewall Boundary</span>
        </div>

        <div className="p-4 rounded-xl bg-white border border-slate-200 font-mono shadow-xs">
          <span className="text-[10px] text-slate-500 block uppercase font-semibold">Holland Parameter (B)</span>
          <span className="text-xl font-bold text-amber-700 mt-1 block">
            {profile.holland_b}
          </span>
          <span className="text-[10px] text-slate-500">Vortex Peakedness [1.0, 2.5]</span>
        </div>

        <div className="p-4 rounded-xl bg-white border border-slate-200 font-mono shadow-xs">
          <span className="text-[10px] text-slate-500 block uppercase font-semibold">Pressure Deficit (ΔP)</span>
          <span className="text-xl font-bold text-rose-700 mt-1 block">
            {profile.central_pressure_deficit_hpa} <span className="text-xs text-slate-500">hPa</span>
          </span>
          <span className="text-[10px] text-slate-500">
            {profile.central_pressure_hpa} hPa vs 1012 hPa
          </span>
        </div>

        <div className="p-4 rounded-xl bg-white border border-slate-200 font-mono shadow-xs">
          <span className="text-[10px] text-slate-500 block uppercase font-semibold">Peak Gradient Wind</span>
          <span className="text-xl font-bold text-emerald-700 mt-1 block">
            {profile.peak_gradient_wind_knots} <span className="text-xs text-slate-500">kt</span>
          </span>
          <span className="text-[10px] text-slate-500">
            ~{(profile.peak_gradient_wind_knots * 1.852).toFixed(0)} km/h
          </span>
        </div>
      </div>

      {/* Main Curve Chart & Quadrant Table */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Continuous Holland Radial Wind Profile Curve (7 cols) */}
        <div className="lg:col-span-7 rounded-2xl bg-white border border-slate-200 p-5 flex flex-col gap-4 shadow-xs">
          <div className="flex items-center justify-between border-b border-slate-200 pb-2">
            <span className="text-xs font-mono font-bold text-slate-900 uppercase flex items-center gap-2">
              <Activity className="w-4 h-4 text-blue-700" />
              Continuous Radial Wind Speed Profile V(r)
            </span>
            <span className="text-[10px] font-mono text-blue-700 font-semibold">
              RMW = {profile.radius_max_wind_km} km
            </span>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={profile.radial_curve} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="radius_km"
                  stroke="#94a3b8"
                  fontSize={11}
                  tickFormatter={(val) => `${val}km`}
                />
                <YAxis
                  stroke="#0284c7"
                  fontSize={11}
                  tickFormatter={(val) => `${val}kt`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#ffffff",
                    borderColor: "#cbd5e1",
                    fontSize: "12px",
                    fontFamily: "monospace",
                    borderRadius: "8px",
                    color: "#0f172a",
                    boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
                  }}
                  formatter={(value: any) => [`${value} knots`, "Wind Speed"]}
                  labelFormatter={(label) => `Radius: ${label} km`}
                />
                <ReferenceLine
                  x={profile.radius_max_wind_km}
                  stroke="#d97706"
                  strokeDasharray="4 4"
                  label={{ value: "RMW", fill: "#d97706", fontSize: 10, position: "top" }}
                />
                <ReferenceLine
                  y={64}
                  stroke="#ef4444"
                  strokeDasharray="2 2"
                  label={{ value: "64kt (Hurricane)", fill: "#ef4444", fontSize: 9, position: "right" }}
                />
                <ReferenceLine
                  y={34}
                  stroke="#0284c7"
                  strokeDasharray="2 2"
                  label={{ value: "34kt (Gale)", fill: "#0284c7", fontSize: 9, position: "right" }}
                />
                <Line
                  type="monotone"
                  dataKey="wind_speed_knots"
                  stroke="#0284c7"
                  strokeWidth={2.5}
                  dot={false}
                  activeDot={{ r: 5, fill: "#0284c7" }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="text-[10px] font-mono text-slate-500 pt-1 border-t border-slate-200 flex items-center justify-between">
            <span>Holland (1980): Gradient balance between pressure gradient, centrifugal and Coriolis forces</span>
            <span className="text-amber-700 font-semibold">Golden line = RMW ({profile.radius_max_wind_km} km)</span>
          </div>
        </div>

        {/* Right: Quadrant Gale Radii Compass Breakdown (5 cols) */}
        <div className="lg:col-span-5 rounded-2xl bg-white border border-slate-200 p-5 flex flex-col justify-between gap-4 shadow-xs font-mono text-xs">
          <div>
            <div className="flex items-center justify-between border-b border-slate-200 pb-2 mb-3">
              <span className="font-bold text-slate-900 uppercase flex items-center gap-2">
                <Compass className="w-4 h-4 text-blue-700" />
                Quadrant Gale Radii (R34 / R50 / R64)
              </span>
              <span className="text-[10px] text-amber-800 bg-amber-50 px-2 py-0.5 rounded border border-amber-200 font-bold">
                ASYMMETRIC
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500 text-[10px] uppercase bg-slate-50">
                    <th className="py-2 px-2">Quadrant</th>
                    <th className="py-2 px-2 text-blue-700">R34 (Gale)</th>
                    <th className="py-2 px-2 text-amber-700">R50 (Storm)</th>
                    <th className="py-2 px-2 text-rose-700">R64 (Hurricane)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-slate-700">
                  {["NE", "SE", "SW", "NW"].map((quad) => {
                    const qData = quads[quad] || { r34_km: 0, r50_km: 0, r64_km: 0 };
                    const isForwardRight = quad === "NE";
                    return (
                      <tr key={quad} className={isForwardRight ? "bg-amber-50/60 font-bold" : ""}>
                        <td className="py-2 px-2 flex items-center gap-1.5">
                          <span className={isForwardRight ? "text-amber-800" : "text-slate-800"}>
                            {quad}
                          </span>
                          {isForwardRight && (
                            <span className="text-[9px] px-1 py-0.2 rounded bg-amber-100 text-amber-800 border border-amber-300">
                              DANGEROUS
                            </span>
                          )}
                        </td>
                        <td className="py-2 px-2 text-blue-700 font-bold">{qData.r34_km} km</td>
                        <td className="py-2 px-2 text-amber-700 font-bold">{qData.r50_km} km</td>
                        <td className="py-2 px-2 text-rose-700 font-bold">{qData.r64_km} km</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="mt-4 p-3 rounded-xl bg-slate-50 border border-slate-200 text-[11px] text-slate-700 space-y-1.5">
              <div className="font-bold text-amber-800 flex items-center gap-1.5">
                <Info className="w-3.5 h-3.5 text-amber-600" />
                <span>Model-Derived Wind Radii &amp; Translation Asymmetry Notice</span>
              </div>
              <p className="text-slate-600 text-[10px] leading-relaxed">
                R34, R50, and R64 values are model-derived analytic estimates computed via the Holland gradient wind formulation, not official warning or advisory wind radii issued by national meteorological authorities. In the Northern Hemisphere, forward translation velocity constructively adds to the cyclonic rotational wind field in the forward-right (Northeast) quadrant, significantly increasing coastal inundation hazard and destructive gale footprint.
              </p>
            </div>
          </div>

          <div className="text-[10px] text-slate-500 pt-2 border-t border-slate-200">
            {profile.scientific_note}
          </div>
        </div>
      </div>
    </div>
  );
}

