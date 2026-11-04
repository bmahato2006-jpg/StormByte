"use client";

import React, { useState } from "react";
import { Sliders, RefreshCw, AlertCircle, Sparkles, RotateCcw } from "lucide-react";
import { Cyclone } from "@/types/cyclone";
import { CompositeRiskAssessment, ScenarioRequest } from "@/types/impact";
import { runScenarioImpact } from "@/lib/api";

interface ScenarioStudioProps {
  cyclone: Cyclone | null;
  onScenarioEvaluated: (assessment: CompositeRiskAssessment) => void;
  onResetBaseline: () => void;
  isScenarioActive: boolean;
}

export const ScenarioStudio: React.FC<ScenarioStudioProps> = ({
  cyclone,
  onScenarioEvaluated,
  onResetBaseline,
  isScenarioActive,
}) => {
  const [windOverride, setWindOverride] = useState<number>(cyclone?.windSpeed ?? 25);
  const [latOffset, setLatOffset] = useState<number>(0.0);
  const [lonOffset, setLonOffset] = useState<number>(0.0);
  const [rainScenarioMm, setRainScenarioMm] = useState<number>(180);
  const [surgeScenarioM, setSurgeScenarioM] = useState<number>(3.5);
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Synchronize when cyclone target changes
  React.useEffect(() => {
    if (cyclone?.windSpeed !== undefined && cyclone?.windSpeed !== null) {
      setWindOverride(cyclone.windSpeed);
    }
  }, [cyclone?.id, cyclone?.windSpeed]);

  const handleRunScenario = async () => {
    if (!cyclone) return;
    setIsEvaluating(true);
    setErrorMsg(null);

    const isHistoricalOrDemo = Boolean(
      cyclone.id.toLowerCase().includes("demo") ||
      cyclone.name.toLowerCase().includes("amphan") ||
      cyclone.name.toLowerCase().includes("biparjoy")
    );

    const payload: ScenarioRequest = {
      storm_id: cyclone.id,
      wind_speed_override_knots: windOverride,
      track_offset_lat: latOffset !== 0 ? latOffset : undefined,
      track_offset_lon: lonOffset !== 0 ? lonOffset : undefined,
      rainfall_scenario_mm: rainScenarioMm,
      coastal_tide_surge_scenario_m: surgeScenarioM,
      include_demo_exposure: isHistoricalOrDemo,
    };

    try {
      const res = await runScenarioImpact(payload);
      onScenarioEvaluated(res);
    } catch (err: any) {
      console.error("Scenario evaluation failed:", err);
      setErrorMsg(err.message || "Failed to execute counterfactual scenario.");
    } finally {
      setIsEvaluating(false);
    }
  };

  const handleReset = () => {
    if (cyclone) {
      setWindOverride(cyclone.windSpeed);
      setLatOffset(0.0);
      setLonOffset(0.0);
      setRainScenarioMm(180);
      setSurgeScenarioM(3.5);
    }
    onResetBaseline();
  };

  return (
    <div className="p-5 rounded-xl border border-slate-200 bg-white shadow-xs font-mono space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-200">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-amber-50 border border-amber-200 flex items-center justify-center">
            <Sliders className="w-4 h-4 text-amber-700" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-900 tracking-wide uppercase">
              WHAT-IF DISASTER SCENARIO STUDIO
            </h3>
            <span className="text-[10px] text-slate-500 font-semibold">
              Counterfactual Hazard Perturbation &amp; Stress Testing
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {isScenarioActive && (
            <span className="text-[10px] px-2.5 py-1 rounded bg-amber-50 text-amber-800 border border-amber-300 font-bold animate-pulse">
              SCENARIO ACTIVE
            </span>
          )}
          <button
            onClick={handleReset}
            disabled={!isScenarioActive}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 disabled:opacity-40 shadow-2xs"
          >
            <RotateCcw className="w-3 h-3" />
            <span>RESET BASELINE</span>
          </button>
        </div>
      </div>

      {/* Perturbation Guardrail Notice */}
      <div className="p-2.5 rounded-lg bg-amber-50/60 border border-amber-200/80 text-[11px] text-slate-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="px-1.5 py-0.5 rounded text-[9px] bg-amber-100 text-amber-900 border border-amber-300 font-bold">
            USER PERTURBATION
          </span>
          <span>Modifying parameters runs a counterfactual simulation without altering underlying satellite feeds.</span>
        </div>
      </div>

      {/* Controls Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pt-1">
        {/* 1. Sustained Wind Override */}
        <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-700 font-bold">WIND INTENSITY</span>
            <span className="text-blue-700 font-bold">{windOverride} kt</span>
          </div>
          <input
            type="range"
            min="30"
            max="165"
            step="5"
            value={windOverride}
            onChange={(e) => setWindOverride(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
          <div className="flex justify-between text-[9px] text-slate-500 font-medium">
            <span>Gale (34 kt)</span>
            <span>Super Cyclone (140+ kt)</span>
          </div>
        </div>

        {/* 2. Synthetic Rainfall Scenario */}
        <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-700 font-bold">PRECIPITATION ACCUMULATION</span>
            <span className="text-blue-700 font-bold">{rainScenarioMm} mm/24h</span>
          </div>
          <input
            type="range"
            min="20"
            max="450"
            step="10"
            value={rainScenarioMm}
            onChange={(e) => setRainScenarioMm(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
          <div className="flex justify-between text-[9px] text-slate-500 font-medium">
            <span>Moderate (50 mm)</span>
            <span>Extreme Deluge (300+ mm)</span>
          </div>
        </div>

        {/* 3. Coastal Surge Scenario */}
        <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-700 font-bold">COASTAL STORM SURGE</span>
            <span className="text-purple-700 font-bold">{surgeScenarioM} m</span>
          </div>
          <input
            type="range"
            min="0.5"
            max="7.5"
            step="0.2"
            value={surgeScenarioM}
            onChange={(e) => setSurgeScenarioM(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
          />
          <div className="flex justify-between text-[9px] text-slate-500 font-medium">
            <span>Tidal Setup (1.0m)</span>
            <span>Severe Surge (&gt; 5.0m)</span>
          </div>
        </div>

        {/* 4. Track Latitude Offset */}
        <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-700 font-bold">TRACK LATITUDE OFFSET</span>
            <span className="text-amber-800 font-bold">{latOffset >= 0 ? `+${latOffset}` : latOffset}°</span>
          </div>
          <input
            type="range"
            min="-3.0"
            max="3.0"
            step="0.2"
            value={latOffset}
            onChange={(e) => setLatOffset(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-amber-600"
          />
          <div className="flex justify-between text-[9px] text-slate-500 font-medium">
            <span>Southward (-3°)</span>
            <span>Northward (+3°)</span>
          </div>
        </div>

        {/* 5. Track Longitude Offset */}
        <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-700 font-bold">TRACK LONGITUDE OFFSET</span>
            <span className="text-amber-800 font-bold">{lonOffset >= 0 ? `+${lonOffset}` : lonOffset}°</span>
          </div>
          <input
            type="range"
            min="-3.0"
            max="3.0"
            step="0.2"
            value={lonOffset}
            onChange={(e) => setLonOffset(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-amber-600"
          />
          <div className="flex justify-between text-[9px] text-slate-500 font-medium">
            <span>Westward (-3°)</span>
            <span>Eastward (+3°)</span>
          </div>
        </div>

        {/* Action Button */}
        <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 flex flex-col justify-end space-y-2">
          <button
            onClick={handleRunScenario}
            disabled={isEvaluating}
            className="w-full py-2.5 px-4 rounded-lg bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 text-white font-bold text-xs shadow-xs transition-all flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {isEvaluating ? (
              <RefreshCw className="w-4 h-4 animate-spin text-white" />
            ) : (
              <Sparkles className="w-4 h-4 text-white" />
            )}
            <span>RUN SCENARIO SIMULATION</span>
          </button>
        </div>
      </div>

      {errorMsg && (
        <div className="p-3 rounded-lg bg-red-50 border border-red-200 text-xs text-red-700 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-red-600 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}
    </div>
  );
};

