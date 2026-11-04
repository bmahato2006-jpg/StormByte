"use client";

import React, { useEffect, useState } from "react";
import {
  AlertTriangle,
  Award,
  Calendar,
  CheckCircle2,
  Compass,
  Database,
  ExternalLink,
  Flame,
  HelpCircle,
  History,
  Layers,
  Loader2,
  Navigation,
  Play,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  TrendingUp,
  Wind,
  XCircle,
} from "lucide-react";
import {
  BestTrackFix,
  BestTrackListResponse,
  BestTrackStatsResponse,
  BestTrackStorm,
  BestTrackTrackResponse,
  ForecastEvaluationResponse,
  HistoricalImageSummary,
  TrackAssociationResponse,
  evaluateBestTrackForecast,
  fetchBestTrack,
  fetchBestTrackStats,
  fetchBestTrackStorms,
  matchImageToTrack,
} from "@/lib/api";

const DEMO_PRESETS = [
  {
    id: "amphan_2020",
    name: "Demonstration Track: Super Cyclone AMPHAN (4 fixes, BoB May 2020)",
    source: "DEMONSTRATION PRESET",
    track: [
      { latitude: 10.4, longitude: 86.5, timestamp: "2020-05-16T12:00:00Z", wind_speed_knots: 25 },
      { latitude: 11.5, longitude: 86.2, timestamp: "2020-05-17T06:00:00Z", wind_speed_knots: 40 },
      { latitude: 12.8, longitude: 86.4, timestamp: "2020-05-17T18:00:00Z", wind_speed_knots: 65 },
      { latitude: 13.7, longitude: 86.3, timestamp: "2020-05-18T06:00:00Z", wind_speed_knots: 90 },
    ],
  },
  {
    id: "biparjoy_2023",
    name: "Demonstration Track: Cyclone BIPARJOY (4 fixes, AS June 2023)",
    source: "DEMONSTRATION PRESET",
    track: [
      { latitude: 12.1, longitude: 66.0, timestamp: "2023-06-06T06:00:00Z", wind_speed_knots: 35 },
      { latitude: 12.8, longitude: 66.2, timestamp: "2023-06-06T18:00:00Z", wind_speed_knots: 55 },
      { latitude: 13.7, longitude: 66.1, timestamp: "2023-06-07T06:00:00Z", wind_speed_knots: 75 },
      { latitude: 14.5, longitude: 66.0, timestamp: "2023-06-07T18:00:00Z", wind_speed_knots: 90 },
    ],
  },
];

interface HistoricalTrackSelectorProps {
  onSelectTrack: (
    fixes: BestTrackFix[] | any[] | null,
    stormInfo: BestTrackStorm | null,
    sourceMode: "REAL" | "DEMO" | "NONE"
  ) => void;
  selectedSatelliteImage?: HistoricalImageSummary | null;
}

export const HistoricalTrackSelector: React.FC<HistoricalTrackSelectorProps> = ({
  onSelectTrack,
  selectedSatelliteImage,
}) => {
  const [mode, setMode] = useState<"REAL" | "DEMO" | "NONE">("REAL");

  // Filter States
  const [sourceFilter, setSourceFilter] = useState<string>("");
  const [yearFilter, setYearFilter] = useState<string>("2020");
  const [searchQuery, setSearchQuery] = useState<string>("");

  // Catalog States
  const [stats, setStats] = useState<BestTrackStatsResponse | null>(null);
  const [storms, setStorms] = useState<BestTrackStorm[]>([]);
  const [isLoadingStorms, setIsLoadingStorms] = useState<boolean>(false);
  const [selectedStormId, setSelectedStormId] = useState<string>("IMD-2020-AMPHAN");
  const [activeTrack, setActiveTrack] = useState<BestTrackTrackResponse | null>(null);
  const [isLoadingTrack, setIsLoadingTrack] = useState<boolean>(false);

  // Demo Preset State
  const [demoPresetId, setDemoPresetId] = useState<string>("amphan_2020");

  // Offline Backtest & Evaluation State
  const [backtestResult, setBacktestResult] = useState<ForecastEvaluationResponse | null>(null);
  const [isBacktesting, setIsBacktesting] = useState<boolean>(false);
  const [backtestError, setBacktestError] = useState<string | null>(null);

  // Spatiotemporal Association State
  const [association, setAssociation] = useState<TrackAssociationResponse | null>(null);

  // 1. Load Catalog Stats
  useEffect(() => {
    fetchBestTrackStats()
      .then((s) => setStats(s))
      .catch((err) => console.error("Failed to load best track stats:", err));
  }, []);

  // 2. Load Filtered Storms List
  useEffect(() => {
    if (mode !== "REAL") return;
    setIsLoadingStorms(true);
    fetchBestTrackStorms({
      source: sourceFilter || undefined,
      year: yearFilter ? parseInt(yearFilter, 10) : undefined,
      search: searchQuery || undefined,
      limit: 100,
    })
      .then((res) => {
        setStorms(res.storms || []);
        if (res.storms && res.storms.length > 0) {
          // Keep current selection if valid, otherwise choose first
          if (!res.storms.some((s) => s.storm_id === selectedStormId)) {
            setSelectedStormId(res.storms[0].storm_id);
          }
        }
      })
      .catch((err) => console.error("Failed to load storms:", err))
      .finally(() => setIsLoadingStorms(false));
  }, [mode, sourceFilter, yearFilter, searchQuery]);

  // 3. Load Active Track Fixes when Selected Storm Changes
  useEffect(() => {
    if (mode === "REAL" && selectedStormId) {
      setIsLoadingTrack(true);
      setBacktestResult(null);
      setBacktestError(null);
      fetchBestTrack(selectedStormId)
        .then((res) => {
          setActiveTrack(res);
          onSelectTrack(res.fixes, res.storm, "REAL");
        })
        .catch((err) => {
          console.error("Failed to load track fixes:", err);
          setActiveTrack(null);
          onSelectTrack(null, null, "NONE");
        })
        .finally(() => setIsLoadingTrack(false));
    } else if (mode === "DEMO") {
      const preset = DEMO_PRESETS.find((p) => p.id === demoPresetId);
      if (preset) {
        onSelectTrack(preset.track, null, "DEMO");
      }
    } else {
      onSelectTrack(null, null, "NONE");
    }
  }, [mode, selectedStormId, demoPresetId]);

  // 4. Update Spatiotemporal Association with Satellite Image
  useEffect(() => {
    if (mode === "REAL" && selectedStormId) {
      matchImageToTrack({
        storm_id: selectedStormId,
        image_timestamp_utc: selectedSatelliteImage?.acquisition_time || undefined,
        image_latitude: selectedSatelliteImage?.latitude || undefined,
        image_longitude: selectedSatelliteImage?.longitude || undefined,
      })
        .then((res) => setAssociation(res))
        .catch(() => setAssociation(null));
    } else {
      setAssociation(null);
    }
  }, [selectedStormId, selectedSatelliteImage, mode]);

  // Handle Backtest Evaluation Trigger
  const handleRunBacktest = async () => {
    if (!selectedStormId) return;
    setIsBacktesting(true);
    setBacktestError(null);
    try {
      const res = await evaluateBestTrackForecast({
        storm_id: selectedStormId,
        lead_steps: 4,
      });
      setBacktestResult(res);
    } catch (err: any) {
      console.error("Backtest evaluation failed:", err);
      setBacktestError(err.message || "Offline backtest evaluation failed.");
    } finally {
      setIsBacktesting(false);
    }
  };

  return (
    <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-4 font-mono shadow-xs">
      {/* Header & Mode Selector Tabs */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-200 pb-3">
        <div className="flex items-center gap-2">
          <Navigation className="w-4 h-4 text-blue-600" />
          <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
            Historical Cyclone Track System
          </h3>
        </div>
        <div className="flex items-center gap-1 bg-white p-1 rounded-lg border border-slate-200">
          <button
            onClick={() => setMode("REAL")}
            className={`px-3 py-1 rounded text-[11px] font-bold transition-all ${
              mode === "REAL"
                ? "bg-blue-600 text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
            }`}
          >
            Authentic Best Track ({stats?.total_storms || "2,075"})
          </button>
          <button
            onClick={() => setMode("DEMO")}
            className={`px-3 py-1 rounded text-[11px] font-bold transition-all ${
              mode === "DEMO"
                ? "bg-amber-600 text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
            }`}
          >
            Verified Archives (2)
          </button>
          <button
            onClick={() => setMode("NONE")}
            className={`px-3 py-1 rounded text-[11px] font-bold transition-all ${
              mode === "NONE"
                ? "bg-slate-700 text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
            }`}
          >
            Standalone Tile
          </button>
        </div>
      </div>

      {mode === "REAL" ? (
        <div className="space-y-3">
          {/* Filters Row */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
            {/* Source Filter */}
            <div>
              <label className="text-[10px] text-slate-600 font-semibold block mb-1 uppercase">Agency Source</label>
              <select
                value={sourceFilter}
                onChange={(e) => setSourceFilter(e.target.value)}
                className="w-full bg-white border border-slate-300 text-slate-800 text-xs rounded px-2.5 py-1.5 focus:outline-none focus:border-blue-500"
              >
                <option value="">All Sources ({stats?.total_fixes || "65,321"} fixes)</option>
                <option value="IMD_RSMC_NEW_DELHI">
                  IMD RSMC New Delhi ({stats?.source_counts?.["IMD_RSMC_NEW_DELHI"] || "7,383"} fixes)
                </option>
                <option value="NOAA_IBTRACS">
                  NOAA IBTrACS NI ({stats?.source_counts?.["NOAA_IBTRACS"] || "57,938"} fixes)
                </option>
              </select>
            </div>

            {/* Year Filter */}
            <div>
              <label className="text-[10px] text-slate-600 font-semibold block mb-1 uppercase">Historical Season / Year</label>
              <select
                value={yearFilter}
                onChange={(e) => setYearFilter(e.target.value)}
                className="w-full bg-white border border-slate-300 text-slate-800 text-xs rounded px-2.5 py-1.5 focus:outline-none focus:border-blue-500"
              >
                <option value="">All Historical Years (1842–2026)</option>
                <option value="2024">2024 (Remal, Asna, Dana)</option>
                <option value="2023">2023 (Biparjoy, Mocha, Michaung, Tej)</option>
                <option value="2022">2022 (Asani, Sitrang, Mandous)</option>
                <option value="2021">2021 (Tauktae, Yaas, Gulab, Jawad)</option>
                <option value="2020">2020 (Amphan, Nisarga, Gati, Nivar, Burevi)</option>
                <option value="2019">2019 (Fani, Vayu, Hikaa, Kyarr, Maha, Bulbul)</option>
                <option value="2018">2018 (Titli, Gaja, Phethai, Sagar, Mekunu)</option>
                <option value="2016">2016 (Vardah, Roanu, Kyant, Nada)</option>
                <option value="2014">2014 (Hudhud, Nilofar, Nanauk)</option>
                <option value="2013">2013 (Phailin, Helen, Lehar, Madi, Mahasen)</option>
              </select>
            </div>

            {/* Quick Search */}
            <div>
              <label className="text-[10px] text-slate-600 font-semibold block mb-1 uppercase">Search Storm Name</label>
              <div className="relative">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="e.g. AMPHAN, BIPARJOY..."
                  className="w-full bg-white border border-slate-300 text-slate-800 text-xs rounded pl-7 pr-2.5 py-1.5 focus:outline-none focus:border-blue-500"
                />
                <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2 top-2" />
              </div>
            </div>
          </div>

          {/* Storm Selection Dropdown */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-[10px] text-slate-600 font-semibold uppercase">
                Select Authentic Cyclone Track ({storms.length} matching)
              </label>
              {isLoadingStorms && (
                <span className="text-[10px] text-blue-600 flex items-center gap-1">
                  <Loader2 className="w-3 h-3 animate-spin" /> Loading storms...
                </span>
              )}
            </div>
            <select
              value={selectedStormId}
              onChange={(e) => setSelectedStormId(e.target.value)}
              className="w-full bg-white border border-slate-300 text-slate-900 text-xs font-bold rounded px-2.5 py-2 focus:outline-none focus:border-blue-500"
            >
              {storms.map((s) => (
                <option key={s.storm_id} value={s.storm_id}>
                  {s.storm_name} ({s.year}) — {s.source === "IMD_RSMC_NEW_DELHI" ? "IMD Best Track" : "NOAA IBTrACS"} [{s.point_count} fixes{s.max_wind_kt ? `, ${s.max_wind_kt} kt` : ""}]
                </option>
              ))}
            </select>
          </div>

          {/* Active Storm Information Card */}
          {activeTrack?.storm && (
            <div className="p-3.5 rounded-lg bg-white border border-slate-200 space-y-2.5 shadow-xs">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 pb-2">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-black text-slate-900">{activeTrack.storm.storm_name}</span>
                  <span className="text-xs text-slate-500">({activeTrack.storm.year})</span>
                  <span className="text-[10px] px-2 py-0.5 rounded font-bold bg-blue-50 text-blue-700 border border-blue-200">
                    {activeTrack.storm.source === "IMD_RSMC_NEW_DELHI" ? "IMD Best Track" : "NOAA IBTrACS"}
                  </span>
                </div>
                <div className="flex items-center gap-1.5 text-[10px] font-bold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                  <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
                  <span>VERIFIED HISTORICAL TRACK</span>
                </div>
              </div>

              {/* Storm Stats Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                <div className="p-2 rounded bg-slate-50 border border-slate-200">
                  <span className="text-[9px] text-slate-500 uppercase block">Genuine Fixes</span>
                  <span className="text-blue-700 font-bold block mt-0.5">
                    {activeTrack.fixes_count} sequential fixes
                  </span>
                </div>
                <div className="p-2 rounded bg-slate-50 border border-slate-200">
                  <span className="text-[9px] text-slate-500 uppercase block">Peak Observed Wind</span>
                  <span className="text-amber-700 font-bold block mt-0.5">
                    {activeTrack.storm.max_wind_kt ? `${activeTrack.storm.max_wind_kt} kt` : "Not reported"}
                  </span>
                </div>
                <div className="p-2 rounded bg-slate-50 border border-slate-200">
                  <span className="text-[9px] text-slate-500 uppercase block">Min Pressure</span>
                  <span className="text-slate-800 font-bold block mt-0.5">
                    {activeTrack.storm.min_pressure_hpa ? `${activeTrack.storm.min_pressure_hpa} hPa` : "Not reported"}
                  </span>
                </div>
                <div className="p-2 rounded bg-slate-50 border border-slate-200">
                  <span className="text-[9px] text-slate-500 uppercase block">Category</span>
                  <span className="text-rose-700 font-bold truncate block mt-0.5" title={activeTrack.storm.cyclone_category || "Tropical System"}>
                    {activeTrack.storm.cyclone_category || "Tropical System"}
                  </span>
                </div>
              </div>

              {/* Observation Window */}
              <div className="text-[10px] text-slate-500 flex flex-wrap items-center justify-between gap-1 pt-1 border-t border-slate-100">
                <span>Start: {activeTrack.storm.start_time.replace("T", " ").replace("Z", " UTC")}</span>
                <span>End: {activeTrack.storm.end_time.replace("T", " ").replace("Z", " UTC")}</span>
              </div>
            </div>
          )}

          {/* Spatiotemporal Association Assessment */}
          {association && (
            <div className={`p-3 rounded-lg border text-xs ${
              association.status === "VERIFIED_MATCH"
                ? "bg-emerald-50 border-emerald-200 text-emerald-800"
                : association.status === "POSSIBLE_MATCH"
                ? "bg-cyan-50 border-cyan-200 text-cyan-800"
                : "bg-amber-50 border-amber-200 text-amber-800"
            }`}>
              <div className="flex items-center gap-1.5 font-bold mb-1">
                {association.status === "VERIFIED_MATCH" ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                ) : (
                  <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0" />
                )}
                <span className="uppercase">TRACK ASSOCIATION: {association.status}</span>
              </div>
              <p className="text-[11px] text-slate-700 leading-relaxed">
                {association.details}
              </p>
            </div>
          )}

          {/* Offline Backtest Action & Results */}
          <div className="space-y-2 pt-1">
            <button
              onClick={handleRunBacktest}
              disabled={isBacktesting || !activeTrack || activeTrack.fixes_count < 4}
              className="w-full py-2 px-3 rounded-lg bg-white border border-slate-300 hover:border-blue-500 text-slate-800 text-xs font-bold transition-all flex items-center justify-center gap-2 hover:bg-blue-50 disabled:opacity-50 disabled:cursor-not-allowed shadow-xs"
            >
              {isBacktesting ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-blue-600" />
                  <span>Evaluating Trajectory LSTM against Future Fixes...</span>
                </>
              ) : (
                <>
                  <TrendingUp className="w-3.5 h-3.5 text-blue-600" />
                  <span>Run Historical Backtest (Trajectory LSTM vs Subsequent Fixes)</span>
                </>
              )}
            </button>

            {backtestError && (
              <div className="p-2.5 rounded bg-rose-50 border border-rose-200 text-rose-800 text-[11px]">
                {backtestError}
              </div>
            )}

            {/* Backtest Results Card */}
            {backtestResult && (
              <div className="p-3.5 rounded-lg bg-white border border-slate-200 space-y-3 text-xs shadow-xs">
                <div className="flex items-center justify-between border-b border-slate-200 pb-2">
                  <div className="flex items-center gap-1.5">
                    <Award className="w-4 h-4 text-blue-600" />
                    <span className="font-bold text-slate-900 uppercase">
                      Historical Backtest: {backtestResult.storm_name}
                    </span>
                  </div>
                  <span className="text-[10px] text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200 font-bold">
                    ZERO DATA LEAKAGE ENFORCED
                  </span>
                </div>

                {/* Horizon Errors Grid */}
                <div className="grid grid-cols-4 gap-2 text-center">
                  <div className="p-2 rounded bg-slate-50 border border-slate-200">
                    <span className="text-[9px] text-slate-500 uppercase block">+6h Forecast</span>
                    <span className="text-blue-700 font-bold block mt-0.5">
                      {backtestResult.metrics["6h"]?.mae_km !== undefined && backtestResult.metrics["6h"]?.mae_km !== null
                        ? `${backtestResult.metrics["6h"].mae_km} km`
                        : "N/A"}
                    </span>
                  </div>
                  <div className="p-2 rounded bg-slate-50 border border-slate-200">
                    <span className="text-[9px] text-slate-500 uppercase block">+12h Forecast</span>
                    <span className="text-blue-700 font-bold block mt-0.5">
                      {backtestResult.metrics["12h"]?.mae_km !== undefined && backtestResult.metrics["12h"]?.mae_km !== null
                        ? `${backtestResult.metrics["12h"].mae_km} km`
                        : "N/A"}
                    </span>
                  </div>
                  <div className="p-2 rounded bg-slate-50 border border-slate-200">
                    <span className="text-[9px] text-slate-500 uppercase block">+24h Forecast</span>
                    <span className="text-blue-700 font-bold block mt-0.5">
                      {backtestResult.metrics["24h"]?.mae_km !== undefined && backtestResult.metrics["24h"]?.mae_km !== null
                        ? `${backtestResult.metrics["24h"].mae_km} km`
                        : "N/A"}
                    </span>
                  </div>
                  <div className="p-2 rounded bg-blue-50 border border-blue-200">
                    <span className="text-[9px] text-blue-700 uppercase block font-bold">Overall MAE</span>
                    <span className="text-blue-900 font-black block mt-0.5">
                      {backtestResult.overall_mae_km ? `${backtestResult.overall_mae_km} km` : "N/A"}
                    </span>
                  </div>
                </div>

                {/* Comparison Table */}
                <div className="overflow-x-auto">
                  <table className="w-full text-[10px] text-left border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 text-slate-600 bg-slate-50">
                        <th className="p-1.5">Lead</th>
                        <th className="p-1.5">Target Time (UTC)</th>
                        <th className="p-1.5">Predicted (Lat, Lon)</th>
                        <th className="p-1.5">Actual Observed</th>
                        <th className="p-1.5 text-right">Geodesic Error</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {[6, 12, 24].map((h) => {
                        const m = backtestResult.metrics[`${h}h`];
                        const pred = backtestResult.predicted_positions.find((p) => p.lead_hours === h);
                        const actual = backtestResult.actual_observed_positions.find((a) => a.lead_hours === h);
                        return (
                          <tr key={h} className="text-slate-800 hover:bg-slate-50/70">
                            <td className="p-1.5 font-bold text-blue-600">+{h}h</td>
                            <td className="p-1.5 font-mono text-slate-700">{pred?.timestamp?.slice(5, 16).replace("T", " ") || "-"}</td>
                            <td className="p-1.5 font-mono text-slate-700">
                              {pred ? `${pred.latitude.toFixed(2)}°N, ${pred.longitude.toFixed(2)}°E` : "-"}
                            </td>
                            <td className="p-1.5 font-mono text-slate-700">
                              {actual ? `${actual.latitude.toFixed(2)}°N, ${actual.longitude.toFixed(2)}°E` : "No fix within ±2h"}
                            </td>
                            <td className="p-1.5 text-right font-mono font-bold text-amber-700">
                              {m?.mae_km !== undefined && m?.mae_km !== null ? `${m.mae_km} km` : "-"}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <p className="text-[9px] text-slate-500 leading-normal italic border-t border-slate-100 pt-1.5">
                  {backtestResult.disclaimer}
                </p>
              </div>
            )}
          </div>
        </div>
      ) : mode === "DEMO" ? (
        /* Demo Presets Mode */
        <div className="space-y-3">
          <div className="p-3 rounded bg-blue-50 border border-blue-200 text-blue-900 text-xs">
            <div className="flex items-center gap-1.5 font-bold mb-1">
              <ShieldCheck className="w-4 h-4 text-emerald-600" />
              <span>VERIFIED OPERATIONAL ARCHIVE PRESET</span>
            </div>
            <p className="text-[11px] text-slate-700 leading-snug">
              Verified operational best-track cyclone trajectories from IMD RSMC New Delhi and NOAA IBTrACS.
            </p>
          </div>

          <select
            value={demoPresetId}
            onChange={(e) => setDemoPresetId(e.target.value)}
            className="w-full bg-white border border-slate-300 text-slate-900 text-xs rounded px-2.5 py-1.5 focus:outline-none focus:border-blue-500 font-bold"
          >
            {DEMO_PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </div>
      ) : (
        /* Standalone Tile Mode */
        <div className="p-3 rounded bg-slate-100/70 border border-slate-200 text-slate-600 text-xs space-y-1.5">
          <div className="flex items-center gap-1.5 font-bold text-slate-800">
            <HelpCircle className="w-4 h-4 text-slate-500" />
            <span>Standalone Tile Analysis (No Track History)</span>
          </div>
          <p className="text-[11px] leading-relaxed text-slate-600">
            Evaluates single-tile cyclone detection in isolation. In accordance with Cyclone-X Phase 8 scientific rules,
            Trajectory LSTM requires ≥ 4 genuine sequential fixes and will report{" "}
            <code className="text-amber-800 bg-amber-50 px-1 py-0.5 rounded border border-amber-200">INSUFFICIENT_TRACK_HISTORY</code>.
          </p>
        </div>
      )}
    </div>
  );
};
