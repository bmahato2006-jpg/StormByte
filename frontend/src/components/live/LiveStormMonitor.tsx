"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Activity, Wifi, AlertTriangle, CheckCircle2, RefreshCw, Radio, Thermometer, Wind, MapPin, ExternalLink, ChevronRight } from "lucide-react";

// ─── Types ─────────────────────────────────────────────────────────────────

interface StormInfo {
  name: string; lat: number; lon: number; basin: string;
  alert_level: string; alert_score: number; severity_kmh: number;
  severity_text: string; from_date: string; to_date: string;
  gdacs_event_id: number; report_url: string; source: string;
}

interface BasinEnv {
  centre_lat: number; centre_lon: number;
  sst_celsius?: number; wind_speed_knots?: number;
  sst_source: string; wind_source: string;
}

interface LiveStatus {
  basin_quiet: boolean;
  active_storms: StormInfo[];
  storm_count: number;
  basin_environment: {
    bay_of_bengal: BasinEnv;
    arabian_sea: BasinEnv;
    observation_time_utc: string;
  };
  storms_provenance: string;
  monitor_timestamp_utc: string;
  data_sources: Record<string, string>;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

const ALERT_COLORS: Record<string, string> = {
  Green: "text-green-400 border-green-400/40 bg-green-400/10",
  Orange: "text-orange-400 border-orange-400/40 bg-orange-400/10",
  Red: "text-red-400 border-red-400/40 bg-red-400/10",
};

function EnvCard({ label, env }: { label: string; env?: BasinEnv }) {
  if (!env) return null;
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-2 shadow-xs">
      <div className="text-xs font-bold text-slate-700 uppercase tracking-widest">{label}</div>
      <div className="flex gap-6">
        <div className="flex items-center gap-2">
          <Thermometer className="w-4 h-4 text-blue-600" />
          <span className="text-sm font-mono font-bold text-slate-900">
            {env.sst_celsius != null ? `${env.sst_celsius}°C` : "—"}
          </span>
          <span className="text-xs text-slate-500 font-semibold">SST</span>
        </div>
        <div className="flex items-center gap-2">
          <Wind className="w-4 h-4 text-blue-600" />
          <span className="text-sm font-mono font-bold text-slate-900">
            {env.wind_speed_knots != null ? `${env.wind_speed_knots} kt` : "—"}
          </span>
          <span className="text-xs text-slate-500 font-semibold">Wind</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-600">
          <MapPin className="w-3 h-3 text-slate-500" />
          {env.centre_lat}°N, {env.centre_lon}°E
        </div>
      </div>
      <div className="text-[10px] text-slate-500 truncate">{env.sst_source}</div>
    </div>
  );
}

// ─── Main Component ─────────────────────────────────────────────────────────

interface LiveStormMonitorProps {
  onAnalyzeStorm?: (result: unknown) => void;
}

export const LiveStormMonitor: React.FC<LiveStormMonitorProps> = ({ onAnalyzeStorm }) => {
  const [status, setStatus] = useState<LiveStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [analyzingStorm, setAnalyzingStorm] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const fetchStatus = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/live/status");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: LiveStatus = await res.json();
      setStatus(data);
      setLastRefresh(new Date());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to fetch live status");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    // Auto-refresh every 15 minutes
    const interval = setInterval(fetchStatus, 15 * 60 * 1000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const analyzeStorm = async (stormName: string) => {
    setAnalyzingStorm(stormName);
    try {
      const res = await fetch(`/api/live/analyze/${encodeURIComponent(stormName)}`, { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const result = await res.json();
      onAnalyzeStorm?.(result);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setAnalyzingStorm(null);
    }
  };

  return (
    <div className="space-y-4">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Radio className="w-5 h-5 text-blue-600" />
            <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-blue-600 animate-ping" />
          </div>
          <div>
            <div className="text-sm font-bold text-slate-900 tracking-wide">LIVE NORTH INDIAN OCEAN MONITOR</div>
            <div className="text-xs text-slate-500">Bay of Bengal · Arabian Sea · Real-time</div>
          </div>
        </div>
        <button
          onClick={fetchStatus}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs text-slate-700 hover:bg-slate-50 shadow-2xs transition-colors disabled:opacity-50 font-semibold"
        >
          <RefreshCw className={`w-3 h-3 ${loading ? "animate-spin text-blue-600" : ""}`} />
          {lastRefresh ? `Updated ${lastRefresh.toLocaleTimeString()}` : "Refresh"}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="rounded-lg border border-red-300 bg-red-50 p-3 text-xs text-red-700 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && !status && (
        <div className="rounded-xl border border-slate-200 bg-white p-8 flex items-center justify-center gap-3 shadow-xs">
          <Activity className="w-5 h-5 text-blue-600 animate-pulse" />
          <span className="text-sm text-slate-600 font-medium">Fetching live data from GDACS and Open-Meteo...</span>
        </div>
      )}

      {status && (
        <>
          {/* Basin Status Banner */}
          <div className={`rounded-xl border p-4 flex items-center gap-4 ${
            status.basin_quiet
              ? "border-emerald-300 bg-white text-emerald-950 shadow-xs"
              : "border-orange-300 bg-white text-orange-950 shadow-xs"
          }`}>
            {status.basin_quiet
              ? <CheckCircle2 className="w-6 h-6 text-emerald-600 shrink-0" />
              : <AlertTriangle className="w-6 h-6 text-orange-600 shrink-0 animate-pulse" />
            }
            <div>
              <div className={`text-sm font-bold ${status.basin_quiet ? "text-emerald-800" : "text-orange-800"}`}>
                {status.basin_quiet
                  ? "NO ACTIVE CYCLONE — North Indian Ocean Basin Quiet"
                  : `${status.storm_count} ACTIVE CYCLONE${status.storm_count > 1 ? "S" : ""} DETECTED`
                }
              </div>
              <div className="text-xs text-slate-600 mt-0.5">
                {status.basin_quiet
                  ? "Environmental monitoring active. Conditions shown below are real-time observations."
                  : "Real-time data from GDACS (UNOCHA/EC-JRC). Click Analyze to run the full AI pipeline."
                }
              </div>
            </div>
          </div>

          {/* Active Storm Cards */}
          {status.active_storms.map((storm) => (
            <div key={storm.gdacs_event_id}
              className="rounded-lg border border-orange-500/30 bg-orange-500/5 p-4 space-y-3">
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-3">
                  <div className={`rounded border px-2 py-0.5 text-xs font-bold ${ALERT_COLORS[storm.alert_level] ?? ALERT_COLORS.Green}`}>
                    {storm.alert_level.toUpperCase()}
                  </div>
                  <div>
                    <div className="text-sm font-bold text-white">{storm.name}</div>
                    <div className="text-xs text-slate-400">{storm.basin === "BOB" ? "Bay of Bengal" : "Arabian Sea"}</div>
                  </div>
                </div>
                {storm.report_url && (
                  <a href={storm.report_url} target="_blank" rel="noopener noreferrer"
                    className="text-xs text-slate-500 hover:text-cyan-400 flex items-center gap-1">
                    GDACS <ExternalLink className="w-3 h-3" />
                  </a>
                )}
              </div>
              <div className="grid grid-cols-3 gap-3 text-xs">
                <div className="rounded bg-white/5 p-2">
                  <div className="text-slate-500">Position</div>
                  <div className="font-mono text-white">{storm.lat.toFixed(2)}°N, {storm.lon.toFixed(2)}°E</div>
                </div>
                <div className="rounded bg-white/5 p-2">
                  <div className="text-slate-500">Intensity</div>
                  <div className="font-mono text-orange-300">{storm.severity_text || `${storm.severity_kmh} km/h`}</div>
                </div>
                <div className="rounded bg-white/5 p-2">
                  <div className="text-slate-500">Source</div>
                  <div className="text-slate-300">GDACS (UNOCHA)</div>
                </div>
              </div>
              <button
                onClick={() => analyzeStorm(storm.name)}
                disabled={analyzingStorm === storm.name}
                className="w-full flex items-center justify-center gap-2 rounded-lg bg-orange-500/20 border border-orange-500/40 px-4 py-2 text-sm font-semibold text-orange-300 hover:bg-orange-500/30 transition-colors disabled:opacity-60"
              >
                {analyzingStorm === storm.name
                  ? <><RefreshCw className="w-4 h-4 animate-spin" /> Running Live Pipeline with Satellite Feed...</>
                  : <><Wifi className="w-4 h-4" /> Analyze Live with NASA &amp; IMD Satellite Stream</>
                }
              </button>
              <div className="text-[10px] text-emerald-400/80 font-mono">
                Live NASA Aqua / IMD INSAT-3D Stream · Track from IBTrACS ACTIVE · SST + Wind from Open-Meteo
              </div>
            </div>
          ))}

          {/* Basin Environment (always shown) */}
          <div className="space-y-2">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest flex items-center gap-2">
              <Thermometer className="w-3 h-3" /> Real-Time Basin Conditions
            </div>
            <EnvCard label="Bay of Bengal" env={status.basin_environment?.bay_of_bengal} />
            <EnvCard label="Arabian Sea" env={status.basin_environment?.arabian_sea} />
            <div className="text-[10px] text-slate-600">
              {status.basin_environment?.observation_time_utc
                ? `Observed: ${new Date(status.basin_environment.observation_time_utc).toUTCString()}`
                : ""}
            </div>
          </div>

          {/* Data Sources */}
          <div className="rounded-xl border border-slate-200 bg-white p-3 space-y-1 shadow-xs">
            <div className="text-[10px] font-semibold text-slate-700 uppercase tracking-wider mb-2">Data Sources</div>
            {Object.entries(status.data_sources).map(([k, v]) => (
              <div key={k} className="flex gap-2 text-[10px]">
                <ChevronRight className="w-3 h-3 text-blue-600 shrink-0 mt-0.5" />
                <span className="text-slate-600 capitalize font-medium">{k.replace(/_/g, " ")}:</span>
                <span className="text-slate-800">{v}</span>
              </div>
            ))}
          </div>

          {/* Disclaimer */}
          <div className="text-[10px] text-slate-500 border-t border-slate-200 pt-3">
            OPERATIONAL SATELLITE MONITORING · Auto-refreshes every 15 minutes
          </div>
        </>
      )}
    </div>
  );
};

export default LiveStormMonitor;
