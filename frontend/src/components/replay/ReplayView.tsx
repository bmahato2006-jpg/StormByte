"use client";

import React, { useState, useEffect } from "react";
import { NormalizedTrackRecord } from "@/types/dataset";
import {
  History,
  Play,
  Pause,
  RotateCcw,
  Navigation,
  Wind,
  Gauge,
  Calendar,
  Layers,
  Database,
  ShieldAlert,
  Compass,
  CheckCircle2,
} from "lucide-react";

export function ReplayView() {
  const [selectedDataset, setSelectedDataset] = useState<"ibtracs" | "imd" | "demo">("ibtracs");
  const [selectedStorm, setSelectedStorm] = useState<string>("AMPHAN");
  const [tracks, setTracks] = useState<NormalizedTrackRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const fetchTrackData = async (dataset: string, storm: string) => {
    setLoading(true);
    try {
      let url = `http://127.0.0.1:8000/api/datasets/tracks?storm_name=${storm}`;
      if (dataset === "ibtracs") url += "&source=IBTRACS";
      if (dataset === "imd") url += "&source=IMD";

      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: NormalizedTrackRecord[] = await res.json();
      setTracks(data);
      setCurrentIndex(0);
      setIsPlaying(false);
    } catch (err) {
      console.error("Failed to load tracks:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrackData(selectedDataset, selectedStorm);
  }, [selectedDataset, selectedStorm]);

  // Auto playback loop
  useEffect(() => {
    let timer: any;
    if (isPlaying && tracks.length > 0) {
      timer = setInterval(() => {
        setCurrentIndex((prev) => {
          if (prev >= tracks.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 1200);
    }
    return () => clearInterval(timer);
  }, [isPlaying, tracks.length]);

  const currentPoint = tracks[currentIndex] || null;

  return (
    <div className="space-y-6 select-none p-2 sm:p-4">
      {/* Top Banner */}
      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 text-[10px] font-mono font-bold uppercase bg-blue-50 text-blue-700 border border-blue-200 rounded">
                Verified Historical Ground Truth
              </span>
              <span className="text-[11px] font-mono text-slate-500">
                Phase 7 Real Scientific Track Integration
              </span>
            </div>
            <h1 className="text-xl sm:text-2xl font-black tracking-wider text-slate-900 flex items-center gap-2">
              <History className="w-6 h-6 text-blue-600" />
              HISTORICAL CYCLONE REPLAY &amp; CHRONOLOGICAL SCRUBBER
            </h1>
            <p className="text-xs text-slate-600 max-w-4xl font-mono">
              Chronological step-by-step playback of authentic historical tropical cyclones. Select between NOAA IBTrACS v04 and IMD RSMC New Delhi best tracks. All coordinates, sustained winds, and central pressures reflect genuine observational best tracks.
            </p>
          </div>
        </div>
      </div>

      {/* Controls Bar */}
      <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-4 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3">
          {/* Dataset Selector */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-slate-600 font-semibold">Dataset Archive:</span>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value as any)}
              className="bg-white border border-slate-300 rounded-lg px-3 py-1.5 text-xs font-mono text-slate-800 focus:outline-none focus:border-blue-500"
            >
              <option value="ibtracs">NOAA IBTrACS v04 (North Indian Ocean)</option>
              <option value="imd">IMD Best-Track Bulletins (RSMC New Delhi)</option>
              <option value="demo">Curated Historical Demo Benchmark</option>
            </select>
          </div>

          {/* Storm Selector */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-slate-600 font-semibold">Cyclone:</span>
            <div className="flex gap-1.5">
              {["AMPHAN", "FANI", "BIPARJOY", "MOCHA"].map((storm) => (
                <button
                  key={storm}
                  onClick={() => setSelectedStorm(storm)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
                    selectedStorm === storm
                      ? "bg-blue-600 text-white shadow-xs"
                      : "bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-300"
                  }`}
                >
                  {storm}
                </button>
              ))}
            </div>
          </div>

          {/* Playback Buttons */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              disabled={tracks.length === 0}
              className="px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-mono font-bold flex items-center gap-1.5 transition-colors disabled:opacity-50 shadow-xs"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isPlaying ? "Pause Replay" : "Play Track"}
            </button>
            <button
              onClick={() => {
                setIsPlaying(false);
                setCurrentIndex(0);
              }}
              className="p-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-300 transition-colors"
              title="Reset to Start"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Scrubber Slider */}
        <div className="space-y-1.5 pt-2 border-t border-slate-200">
          <div className="flex justify-between text-xs font-mono text-slate-600">
            <span>
              Observation: <strong className="text-blue-700">{currentIndex + 1}</strong> of {tracks.length}
            </span>
            <span>
              UTC Timestamp:{" "}
              <strong className="text-slate-900">
                {currentPoint ? new Date(currentPoint.timestamp).toUTCString() : "Loading..."}
              </strong>
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={Math.max(0, tracks.length - 1)}
            value={currentIndex}
            onChange={(e) => {
              setIsPlaying(false);
              setCurrentIndex(parseInt(e.target.value));
            }}
            className="w-full accent-blue-600 cursor-pointer h-2 bg-slate-200 rounded-lg"
          />
        </div>
      </div>

      {/* Telemetry Metrics & Observation Card */}
      {currentPoint ? (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-1 shadow-xs">
            <div className="text-[11px] font-mono text-slate-500 flex items-center gap-1.5">
              <Compass className="w-3.5 h-3.5 text-blue-600" /> Center Coordinates
            </div>
            <div className="text-xl font-mono font-bold text-slate-900">
              {currentPoint.latitude.toFixed(2)}°N, {currentPoint.longitude.toFixed(2)}°E
            </div>
            <div className="text-[10px] font-mono text-slate-500">Geodetic WGS84</div>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-1 shadow-xs">
            <div className="text-[11px] font-mono text-slate-500 flex items-center gap-1.5">
              <Wind className="w-3.5 h-3.5 text-amber-600" /> Max Sustained Wind
            </div>
            <div className="text-xl font-mono font-bold text-amber-600">
              {currentPoint.max_wind_kt ? `${currentPoint.max_wind_kt} kt` : "UNRECORDED"}
            </div>
            <div className="text-[10px] font-mono text-slate-500">
              {currentPoint.max_wind_kt ? `${(currentPoint.max_wind_kt * 1.852).toFixed(0)} km/h` : "N/A"}
            </div>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-1 shadow-xs">
            <div className="text-[11px] font-mono text-slate-500 flex items-center gap-1.5">
              <Gauge className="w-3.5 h-3.5 text-blue-600" /> Central Pressure
            </div>
            <div className="text-xl font-mono font-bold text-blue-600">
              {currentPoint.central_pressure_hpa ? `${currentPoint.central_pressure_hpa} hPa` : "UNRECORDED"}
            </div>
            <div className="text-[10px] font-mono text-slate-500">Estimated core minimum</div>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-1 shadow-xs">
            <div className="text-[11px] font-mono text-slate-500 flex items-center gap-1.5">
              <Navigation className="w-3.5 h-3.5 text-emerald-600" /> Translation Motion
            </div>
            <div className="text-xl font-mono font-bold text-emerald-600">
              {currentPoint.movement_speed_kmh ? `${currentPoint.movement_speed_kmh} km/h` : "N/A"}
            </div>
            <div className="text-[10px] font-mono text-slate-500">
              Heading: {currentPoint.movement_direction_deg ? `${currentPoint.movement_direction_deg}°` : "N/A"}
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center py-8 text-xs font-mono text-slate-500">
          No track observations available for selected filters.
        </div>
      )}

      {/* Observed vs Model Track Guardrail Banner */}
      <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 space-y-2">
        <div className="flex items-center justify-between text-xs font-mono">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-600" />
            <span className="font-bold text-slate-900">Observed Track (Ground Truth)</span>
            <span className="text-slate-500">Source: {currentPoint?.source || "IBTrACS/IMD"}</span>
          </div>
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-blue-50 text-blue-700 border border-blue-200">
            REAL OBSERVATION ARCHIVE
          </span>
        </div>
        <div className="flex items-center justify-between text-xs font-mono border-t border-slate-200 pt-2">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
            <span className="font-bold text-slate-700">Model Track (AI / Prediction Engine)</span>
            <span className="text-slate-500">Status: MODEL NOT RUN ON HISTORICAL REPLAY</span>
          </div>
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-slate-200 text-slate-700 border border-slate-300">
            PREDICTION DECOUPLED
          </span>
        </div>
      </div>
    </div>
  );
}

