"use client";

import React from "react";
import { Thermometer, Wind, CloudRain, ShieldCheck, Droplet, ArrowUpRight } from "lucide-react";
import { EnvironmentalConditions } from "@/types/environment";

interface EnvironmentalPanelProps {
  environment: EnvironmentalConditions | null;
}

export const EnvironmentalPanel: React.FC<EnvironmentalPanelProps> = ({ environment }) => {
  if (!environment) {
    return (
      <div className="p-4 rounded-xl border border-command-cardBorder bg-command-card text-center">
        <span className="font-mono text-xs text-slate-400">LOADING ENVIRONMENTAL DATA...</span>
      </div>
    );
  }

  const isSstFavorable = (environment.seaSurfaceTemperature ?? 0) >= 26.5;
  const isShearFavorable = environment.verticalWindShear !== null && environment.verticalWindShear !== undefined
    ? environment.verticalWindShear <= 15.0
    : null;

  return (
    <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-xs flex flex-col justify-between h-full">
      <div className="flex items-center justify-between border-b border-slate-200 pb-2 mb-3">
        <div className="flex items-center gap-2">
          <Thermometer className="w-4 h-4 text-blue-700" />
          <h3 className="text-xs font-mono font-bold text-slate-900 uppercase tracking-wider">
            ENVIRONMENTAL ATMOSPHERIC CONDITIONS
          </h3>
        </div>
        <span className="text-[10px] font-mono text-slate-500 font-semibold">Open-Meteo NRT</span>
      </div>

      <div className="grid grid-cols-2 gap-3 text-xs font-mono">
        {/* Sea Surface Temperature */}
        <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
          <div className="flex items-center justify-between text-slate-600 mb-1">
            <span className="text-[11px] font-semibold">SST Oceanic Core</span>
            {environment.seaSurfaceTemperature !== null && environment.seaSurfaceTemperature !== undefined ? (
              <span className={`text-[10px] font-bold px-1.5 py-0.2 rounded ${isSstFavorable ? "bg-emerald-50 text-emerald-800 border border-emerald-200" : "bg-amber-50 text-amber-800 border border-amber-200"}`}>
                {isSstFavorable ? "GENESIS READY" : "MARGINAL"}
              </span>
            ) : (
              <span className="text-[10px] font-bold px-1.5 py-0.2 rounded bg-slate-100 text-slate-500 border border-slate-200">N/A</span>
            )}
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-2xl font-black text-slate-900">
              {environment.seaSurfaceTemperature !== null && environment.seaSurfaceTemperature !== undefined
                ? environment.seaSurfaceTemperature
                : "—"}
            </span>
            <span className="text-xs font-bold text-slate-600">°C</span>
          </div>
          <div className="text-[10px] text-slate-500 mt-1">
            {environment.seaSurfaceTemperature !== null && environment.seaSurfaceTemperature !== undefined
              ? `Threshold: > 26.5°C (+${(environment.seaSurfaceTemperature - 26.5).toFixed(1)}°C surplus)`
              : "Source: Open-Meteo Marine API (NRT)"}
          </div>
        </div>

        {/* Vertical Wind Shear */}
        <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
          <div className="flex items-center justify-between text-slate-600 mb-1">
            <span className="text-[11px] font-semibold">850-200hPa Shear</span>
            {isShearFavorable !== null ? (
              <span className={`text-[10px] font-bold px-1.5 py-0.2 rounded ${isShearFavorable ? "bg-emerald-50 text-emerald-800 border border-emerald-200" : "bg-rose-50 text-rose-800 border border-rose-200"}`}>
                {isShearFavorable ? "FAVORABLE" : "HOSTILE"}
              </span>
            ) : (
              <span className="text-[10px] font-bold px-1.5 py-0.2 rounded bg-slate-100 text-slate-500 border border-slate-200">N/A</span>
            )}
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-2xl font-black text-blue-700">
              {environment.verticalWindShear !== null && environment.verticalWindShear !== undefined
                ? environment.verticalWindShear
                : "—"}
            </span>
            <span className="text-xs font-bold text-slate-600">kt</span>
          </div>
          <div className="text-[10px] text-slate-500 mt-1">
            Low shear &lt; 15 kt enables rapid intensification
          </div>
        </div>

        {/* Ambient MSLP */}
        <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
          <div className="text-slate-600 text-[11px] mb-1 font-semibold">Peripheral MSLP</div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-xl font-bold text-amber-700">
              {environment.meanSeaLevelPressure !== null && environment.meanSeaLevelPressure !== undefined
                ? environment.meanSeaLevelPressure
                : "—"}
            </span>
            <span className="text-xs text-slate-600">hPa</span>
          </div>
          <div className="text-[10px] text-slate-500 mt-1">Synoptic background pressure</div>
        </div>

        {/* Relative Humidity */}
        <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
          <div className="text-slate-600 text-[11px] mb-1 font-semibold">Mid-Tropospheric RH</div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-xl font-bold text-blue-700">
              {environment.relativeHumidityMidTroposphere !== null && environment.relativeHumidityMidTroposphere !== undefined
                ? environment.relativeHumidityMidTroposphere
                : "—"}
            </span>
            <span className="text-xs text-slate-600">%</span>
          </div>
          <div className="text-[10px] text-slate-500 mt-1">700-500 hPa moist conveyor</div>
        </div>
      </div>

      <div className="mt-3 p-2.5 rounded-lg bg-slate-50 border border-slate-200 flex items-center justify-between text-[11px] font-mono">
        <span className="text-slate-600 font-medium">Tropical Cyclone Heat Potential (TCHP):</span>
        <span className="font-bold text-blue-800">
          {environment.oceanHeatContent !== null && environment.oceanHeatContent !== undefined
            ? `${environment.oceanHeatContent} kJ/cm²`
            : "N/A"}
        </span>
      </div>

      {/* NASA GPM / PPS IMERG Precipitation Observation Panel */}
      <NASAIMERGPanel />
    </div>
  );
};

const NASAIMERGPanel: React.FC = () => {
  const [lat, setLat] = React.useState<string>("22.57");
  const [lon, setLon] = React.useState<string>("88.36");
  const [loading, setLoading] = React.useState<boolean>(false);
  const [result, setResult] = React.useState<any | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  const queryIMERG = async (targetLat?: string, targetLon?: string) => {
    const qLat = targetLat !== undefined ? targetLat : lat;
    const qLon = targetLon !== undefined ? targetLon : lon;
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`/api/satellite/nrt/nasa/location?lat=${encodeURIComponent(qLat)}&lon=${encodeURIComponent(qLon)}`);
      if (!resp.ok) {
        const errJson = await resp.json().catch(() => ({}));
        throw new Error(errJson.detail || `Lookup failed with status ${resp.status}`);
      }
      const data = await resp.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to query NASA IMERG precipitation.");
    } finally {
      setLoading(false);
    }
  };

  // Perform initial query on mount for Kolkata default
  React.useEffect(() => {
    queryIMERG("22.57", "88.36");
  }, []);

  return (
    <div className="mt-4 pt-3 border-t border-slate-200 font-mono text-xs space-y-3">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
        <div className="flex items-center gap-1.5">
          <CloudRain className="w-4 h-4 text-blue-700" />
          <span className="font-bold text-slate-900 uppercase tracking-wider text-[11px]">
            NASA GPM / PPS IMERG Observation
          </span>
        </div>
        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-blue-50 border border-blue-200 text-blue-800">
          Historical / Verified NASA Observation
        </span>
      </div>

      {/* Observation Metadata Banner */}
      <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 text-[11px] grid grid-cols-2 sm:grid-cols-4 gap-2">
        <div>
          <span className="text-slate-500 block text-[10px]">Product / Ver:</span>
          <span className="text-slate-800 font-bold">3IMERGHH_GIS (V07B)</span>
        </div>
        <div>
          <span className="text-slate-500 block text-[10px]">Observation Time:</span>
          <span className="text-blue-700 font-semibold">2025-08-01 00:00–00:29 UTC</span>
        </div>
        <div>
          <span className="text-slate-500 block text-[10px]">Source / Provider:</span>
          <span className="text-slate-800">NASA PPS</span>
        </div>
        <div>
          <span className="text-slate-500 block text-[10px]">Mode:</span>
          <span className="text-emerald-700 font-semibold">Historical NASA Obs</span>
        </div>
      </div>

      {/* Quick Coordinate Presets */}
      <div className="flex flex-wrap items-center gap-1.5 text-[10px]">
        <span className="text-slate-500 font-semibold">Presets:</span>
        <button
          type="button"
          onClick={() => {
            setLat("22.57");
            setLon("88.36");
            queryIMERG("22.57", "88.36");
          }}
          className="px-2 py-0.5 rounded bg-white hover:bg-slate-50 text-blue-800 border border-slate-300 font-medium shadow-2xs transition-colors"
        >
          📍 Kolkata (22.57°N, 88.36°E)
        </button>
        <button
          type="button"
          onClick={() => {
            setLat("29.95");
            setLon("73.05");
            queryIMERG("29.95", "73.05");
          }}
          className="px-2 py-0.5 rounded bg-white hover:bg-slate-50 text-blue-800 border border-slate-300 font-medium shadow-2xs transition-colors"
        >
          🌧️ Rain Cell (29.95°N, 73.05°E)
        </button>
        <button
          type="button"
          onClick={() => {
            setLat("21.35");
            setLon("92.65");
            queryIMERG("21.35", "92.65");
          }}
          className="px-2 py-0.5 rounded bg-white hover:bg-slate-50 text-blue-800 border border-slate-300 font-medium shadow-2xs transition-colors"
        >
          ⚡ Bay of Bengal (21.35°N, 92.65°E)
        </button>
      </div>

      {/* Coordinate Input & Action */}
      <div className="flex items-center gap-2">
        <div className="flex-1 grid grid-cols-2 gap-2">
          <input
            type="number"
            step="0.01"
            value={lat}
            onChange={(e) => setLat(e.target.value)}
            placeholder="Lat (-90 to +90)"
            className="px-2.5 py-1 bg-white border border-slate-300 rounded text-xs text-slate-900 shadow-2xs"
          />
          <input
            type="number"
            step="0.01"
            value={lon}
            onChange={(e) => setLon(e.target.value)}
            placeholder="Lon (-180 to +180)"
            className="px-2.5 py-1 bg-white border border-slate-300 rounded text-xs text-slate-900 shadow-2xs"
          />
        </div>
        <button
          type="button"
          disabled={loading}
          onClick={() => queryIMERG()}
          className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded text-xs font-bold disabled:opacity-50 shadow-2xs transition-colors"
        >
          {loading ? "QUERYING..." : "QUERY IMERG"}
        </button>
      </div>

      {error && (
        <div className="p-2 rounded bg-rose-50 border border-rose-200 text-rose-800 text-[11px]">
          {error}
        </div>
      )}

      {/* Lookup Result */}
      {result && (
        <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
          <div className="grid grid-cols-2 gap-2 text-[11px] border-b border-slate-200 pb-1.5">
            <div>
              <span className="text-slate-500 block text-[10px]">Requested Target:</span>
              <span className="text-blue-800 font-bold">
                {result.requested_latitude?.toFixed(4)}°N, {result.requested_longitude?.toFixed(4)}°E
              </span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">Actual IMERG Grid Cell:</span>
              <span className="text-emerald-800 font-bold">
                {result.grid_latitude != null ? `${result.grid_latitude.toFixed(2)}°N, ${result.grid_longitude?.toFixed(2)}°E` : "OUTSIDE COVERAGE"}
              </span>
            </div>
          </div>
          <div className="flex items-baseline justify-between">
            <span className="text-slate-600 text-xs">Actual Precipitation:</span>
            <div className="flex items-baseline gap-1.5">
              <span className="text-xl font-black text-slate-900">
                {result.precipitation_mm_per_hr != null ? result.precipitation_mm_per_hr.toFixed(2) : "0.00"}
              </span>
              <span className="text-xs font-bold text-blue-700">mm/hr</span>
              <span className="text-[10px] text-slate-500">
                ({result.precipitation ?? 0} in {result.unit})
              </span>
            </div>
          </div>
          <div className="text-[10px] text-slate-500 grid grid-cols-2 gap-1 pt-1 border-t border-slate-200">
            <div>Provider: <b className="text-slate-700">{result.provider}</b></div>
            <div>Source: <b className="text-slate-700">{result.source}</b></div>
            <div>Product: <b className="text-slate-700">{result.product} ({result.version})</b></div>
            <div>Mode: <b className="text-emerald-700">{result.mode}</b></div>
          </div>
        </div>
      )}
    </div>
  );
};


