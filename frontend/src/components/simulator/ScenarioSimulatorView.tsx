"use client";

import React, { useState, useEffect } from "react";
import { 
  Sliders, 
  Flame, 
  Wind, 
  Waves, 
  CloudRain, 
  AlertTriangle, 
  Compass, 
  Activity, 
  RefreshCw, 
  CheckCircle2,
  Info
} from "lucide-react";

interface PerturbationResult {
  baseline: {
    vmax_knots: number;
    central_pressure_hpa: number;
    translation_speed_kt: number;
    surge_height_m: number;
  };
  perturbed: {
    vmax_knots: number;
    central_pressure_hpa: number;
    translation_speed_kt: number;
    surge_height_m: number;
  };
  impact_deltas: {
    delta_vmax_knots: number;
    delta_pressure_hpa: number;
    delta_surge_m: number;
    category_transition: string;
  };
  surge_projection: {
    estimated_surge_m: number;
    coastal_angle_effect: string;
    inundation_risk_level: string;
  };
  rainfall_amplification: {
    accumulation_multiplier: number;
    risk_classification: string;
    commentary: string;
  };
  scientific_rationale: {
    thermodynamic_mpi: string;
    dynamic_shear: string;
    barometric_coupling: string;
  };
  provenance: {
    source_type: string;
    disclaimer: string;
  };
}

export function ScenarioSimulatorView({ stormId }: { stormId?: string }) {
  const [baseWind, setBaseWind] = useState<number>(75);
  const [basePressure, setBasePressure] = useState<number>(970);
  const [deltaSST, setDeltaSST] = useState<number>(1.0);
  const [deltaShear, setDeltaShear] = useState<number>(-5.0);
  const [transSpeed, setTransSpeed] = useState<number>(8.0);
  const [approachAngle, setApproachAngle] = useState<number>(75);

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PerturbationResult | null>(null);

  const runSimulation = async () => {
    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/simulation/perturb", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          storm_id: stormId || "active_cyclone",
          base_wind_kt: baseWind,
          base_pressure_hpa: basePressure,
          delta_sst_c: deltaSST,
          delta_shear_kt: deltaShear,
          translation_speed_kt: transSpeed,
          approach_angle_deg: approachAngle,
        }),
      });
      if (!res.ok) throw new Error("Simulation request failed");
      const data: PerturbationResult = await res.json();
      setResult(data);
    } catch (err) {
      console.warn("Simulator backend unreachable, executing analytical fallback:", err);
      // Client-side analytical Kerry Emanuel MPI fallback
      const net_delta_v = 6.8 * deltaSST - 0.75 * deltaShear;
      const p_wind = Math.max(25, Math.min(185, baseWind + net_delta_v));
      const delta_p = -0.72 * (p_wind - baseWind);
      const p_pres = Math.max(880, Math.min(1012, basePressure + delta_p));
      const rain_factor = Math.round((12.0 / Math.max(2.0, transSpeed)) * 100) / 100;
      const angle_rad = (approachAngle * Math.PI) / 180;
      const b_surge = Math.max(0.5, 0.00045 * Math.pow(baseWind, 2));
      const p_surge = Math.max(0.5, 0.00045 * Math.pow(p_wind, 2) * Math.pow(Math.sin(angle_rad), 0.8));

      setResult({
        baseline: { vmax_knots: baseWind, central_pressure_hpa: basePressure, translation_speed_kt: 12, surge_height_m: Math.round(b_surge * 10) / 10 },
        perturbed: { vmax_knots: Math.round(p_wind * 10) / 10, central_pressure_hpa: Math.round(p_pres * 10) / 10, translation_speed_kt: transSpeed, surge_height_m: Math.round(p_surge * 10) / 10 },
        impact_deltas: {
          delta_vmax_knots: Math.round((p_wind - baseWind) * 10) / 10,
          delta_pressure_hpa: Math.round(delta_p * 10) / 10,
          delta_surge_m: Math.round((p_surge - b_surge) * 10) / 10,
          category_transition: "Severe Cyclonic Storm -> Very Severe Cyclonic Storm",
        },
        surge_projection: {
          estimated_surge_m: Math.round(p_surge * 10) / 10,
          coastal_angle_effect: `${approachAngle}° incident angle`,
          inundation_risk_level: p_surge >= 3.5 ? "EXTREME" : p_surge >= 2.0 ? "HIGH" : "MODERATE",
        },
        rainfall_amplification: {
          accumulation_multiplier: rain_factor,
          risk_classification: transSpeed <= 4 ? "CATASTROPHIC_STALL_INUNDATION" : transSpeed <= 8 ? "HIGH_ACCUMULATION" : "NORMAL_TRANSIT",
          commentary: `Translation speed of ${transSpeed} kt yields ${rain_factor}x baseline rain accumulation.`,
        },
        scientific_rationale: {
          thermodynamic_mpi: `Kerry Emanuel MPI scaling: ${deltaSST > 0 ? "+" : ""}${deltaSST.toFixed(1)}°C SST yields ${(6.8 * deltaSST).toFixed(1)} kt.`,
          dynamic_shear: `Deep shear perturbation: ${deltaShear > 0 ? "+" : ""}${deltaShear.toFixed(1)} kt yields ${(-0.75 * deltaShear).toFixed(1)} kt.`,
          barometric_coupling: `Atkinson-Holliday wind-pressure coupling yields ${delta_p.toFixed(1)} hPa shift.`,
        },
        provenance: {
          source_type: "COUNTERFACTUAL_SIMULATION",
          disclaimer: "COUNTERFACTUAL SIMULATION — EXPERIMENTAL WHAT-IF SANDBOX — NOT AN OFFICIAL FORECAST",
        },
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runSimulation();
  }, [baseWind, basePressure, deltaSST, deltaShear, transSpeed, approachAngle]);

  return (
    <div className="space-y-6">
      {/* HEADER & MANDATORY PROVENANCE BANNER */}
      <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <AlertTriangle className="h-6 w-6 text-amber-400 shrink-0" />
          <div>
            <h2 className="text-base font-bold text-amber-300 uppercase tracking-wider flex items-center gap-2">
              <span>Module 23: What-If Counterfactual Scenario Simulator</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 font-mono">SANDBOX</span>
            </h2>
            <p className="text-xs text-amber-200/80 mt-1 font-mono">
              STRICT METEOROLOGICAL INTEGRITY: Counterfactual simulations test sensitivity to altered ocean-atmospheric parameters. 
              Not an operational forecast.
            </p>
          </div>
        </div>
      </div>

      {/* CONTROLS & PERTURBATION SLIDERS */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 1/3: Slider Controls */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md space-y-5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Sliders className="h-4 w-4 text-cyan-400" />
              <span>Perturbation Controls</span>
            </h3>
            <button
              onClick={() => {
                setDeltaSST(0);
                setDeltaShear(0);
                setTransSpeed(12);
                setApproachAngle(60);
              }}
              className="text-xs text-slate-400 hover:text-cyan-400 flex items-center gap-1 transition-colors"
            >
              <RefreshCw className="h-3 w-3" />
              <span>Reset</span>
            </button>
          </div>

          {/* SST Anomaly */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300 flex items-center gap-1.5">
                <Flame className="h-3.5 w-3.5 text-rose-400" />
                <span>Δ Sea Surface Temp (SST)</span>
              </span>
              <span className={`font-mono font-bold ${deltaSST > 0 ? "text-rose-400" : deltaSST < 0 ? "text-cyan-400" : "text-slate-400"}`}>
                {deltaSST > 0 ? "+" : ""}{deltaSST.toFixed(1)}°C
              </span>
            </div>
            <input
              type="range"
              min="-2.0"
              max="3.0"
              step="0.5"
              value={deltaSST}
              onChange={(e) => setDeltaSST(parseFloat(e.target.value))}
              className="w-full accent-rose-500 bg-slate-800 rounded-lg cursor-pointer h-1.5"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>-2.0°C (Upwelling)</span>
              <span>Baseline</span>
              <span>+3.0°C (Marine Heatwave)</span>
            </div>
          </div>

          {/* Wind Shear Perturbation */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300 flex items-center gap-1.5">
                <Wind className="h-3.5 w-3.5 text-cyan-400" />
                <span>Δ Vertical Wind Shear</span>
              </span>
              <span className={`font-mono font-bold ${deltaShear > 0 ? "text-amber-400" : deltaShear < 0 ? "text-rose-400" : "text-slate-400"}`}>
                {deltaShear > 0 ? "+" : ""}{deltaShear.toFixed(0)} kt
              </span>
            </div>
            <input
              type="range"
              min="-20"
              max="30"
              step="5"
              value={deltaShear}
              onChange={(e) => setDeltaShear(parseFloat(e.target.value))}
              className="w-full accent-cyan-500 bg-slate-800 rounded-lg cursor-pointer h-1.5"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>-20 kt (Favorable)</span>
              <span>Baseline</span>
              <span>+30 kt (Disruptive)</span>
            </div>
          </div>

          {/* Forward Translation Speed */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300 flex items-center gap-1.5">
                <Activity className="h-3.5 w-3.5 text-indigo-400" />
                <span>Forward Translation Speed</span>
              </span>
              <span className={`font-mono font-bold ${transSpeed <= 6 ? "text-rose-400" : "text-emerald-400"}`}>
                {transSpeed.toFixed(0)} kt {transSpeed <= 5 ? "(STALL)" : ""}
              </span>
            </div>
            <input
              type="range"
              min="2"
              max="25"
              step="1"
              value={transSpeed}
              onChange={(e) => setTransSpeed(parseFloat(e.target.value))}
              className="w-full accent-indigo-500 bg-slate-800 rounded-lg cursor-pointer h-1.5"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>2 kt (Catastrophic Stall)</span>
              <span>12 kt (Nominal)</span>
              <span>25 kt (Rapid)</span>
            </div>
          </div>

          {/* Landfall Approach Angle */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300 flex items-center gap-1.5">
                <Compass className="h-3.5 w-3.5 text-teal-400" />
                <span>Coastline Incident Angle</span>
              </span>
              <span className="font-mono font-bold text-teal-400">
                {approachAngle}° ({approachAngle >= 80 ? "Perpendicular" : "Oblique"})
              </span>
            </div>
            <input
              type="range"
              min="15"
              max="90"
              step="5"
              value={approachAngle}
              onChange={(e) => setApproachAngle(parseFloat(e.target.value))}
              className="w-full accent-teal-500 bg-slate-800 rounded-lg cursor-pointer h-1.5"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>15° (Glancing)</span>
              <span>55°</span>
              <span>90° (Direct Perpendicular)</span>
            </div>
          </div>
        </div>

        {/* Right 2/3: Simulation Projections */}
        <div className="lg:col-span-2 space-y-5">
          {/* COMPARISON METRICS CARDS */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {/* Wind Speed */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 backdrop-blur-md">
              <div className="text-[11px] font-mono text-slate-400 uppercase">Max Sustained Wind</div>
              <div className="text-2xl font-bold font-mono text-white mt-1">
                {result?.perturbed.vmax_knots || baseWind} <span className="text-xs font-normal text-slate-400">kt</span>
              </div>
              <div className="text-xs font-mono mt-2 flex items-center gap-1">
                <span className="text-slate-500">Base: {baseWind} kt</span>
                {result && result.impact_deltas.delta_vmax_knots !== 0 && (
                  <span className={result.impact_deltas.delta_vmax_knots > 0 ? "text-rose-400 font-bold" : "text-cyan-400 font-bold"}>
                    ({result.impact_deltas.delta_vmax_knots > 0 ? "+" : ""}{result.impact_deltas.delta_vmax_knots} kt)
                  </span>
                )}
              </div>
            </div>

            {/* Central Pressure */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 backdrop-blur-md">
              <div className="text-[11px] font-mono text-slate-400 uppercase">Central Pressure</div>
              <div className="text-2xl font-bold font-mono text-white mt-1">
                {result?.perturbed.central_pressure_hpa || basePressure} <span className="text-xs font-normal text-slate-400">hPa</span>
              </div>
              <div className="text-xs font-mono mt-2 flex items-center gap-1">
                <span className="text-slate-500">Base: {basePressure} hPa</span>
                {result && result.impact_deltas.delta_pressure_hpa !== 0 && (
                  <span className={result.impact_deltas.delta_pressure_hpa < 0 ? "text-rose-400 font-bold" : "text-cyan-400 font-bold"}>
                    ({result.impact_deltas.delta_pressure_hpa > 0 ? "+" : ""}{result.impact_deltas.delta_pressure_hpa} hPa)
                  </span>
                )}
              </div>
            </div>

            {/* Peak Storm Surge */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 backdrop-blur-md">
              <div className="text-[11px] font-mono text-slate-400 uppercase">Storm Surge</div>
              <div className="text-2xl font-bold font-mono text-amber-400 mt-1">
                {result?.surge_projection.estimated_surge_m || "2.1"} <span className="text-xs font-normal text-slate-400">m</span>
              </div>
              <div className="text-xs font-mono mt-2">
                <span className="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 font-bold text-[10px]">
                  {result?.surge_projection.inundation_risk_level || "MODERATE"} SURGE
                </span>
              </div>
            </div>

            {/* Rain Factor */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 backdrop-blur-md">
              <div className="text-[11px] font-mono text-slate-400 uppercase">Rain Accumulation</div>
              <div className="text-2xl font-bold font-mono text-cyan-400 mt-1">
                {result?.rainfall_amplification.accumulation_multiplier || 1.0}x
              </div>
              <div className="text-xs font-mono mt-2">
                <span className={`px-1.5 py-0.5 rounded font-bold text-[10px] ${
                  transSpeed <= 5 ? "bg-rose-500/20 text-rose-300" : "bg-cyan-500/20 text-cyan-300"
                }`}>
                  {result?.rainfall_amplification.risk_classification || "NOMINAL"}
                </span>
              </div>
            </div>
          </div>

          {/* HYDROLOGIC & INUNDATION DETAIL */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Surge Recalculation */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md space-y-3">
              <div className="flex items-center gap-2 text-sm font-semibold text-slate-200">
                <Waves className="h-4 w-4 text-cyan-400" />
                <span>Hydrodynamic Surge Recalculation</span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                Surge scales with wind stress τ ∝ V² and coastal bathymetry slope. At {approachAngle}° angle of attack, 
                onshore wind component produces <strong className="text-white">{result?.surge_projection.estimated_surge_m || 2.1}m</strong> peak surge anomaly above mean sea level.
              </p>
              <div className="p-3 rounded-lg bg-slate-950/50 border border-slate-800/80 text-[11px] font-mono text-slate-300 space-y-1">
                <div>• Coastal Incidence: {result?.surge_projection.coastal_angle_effect}</div>
                <div>• Estuarine Inundation: 2.5 - 6.0 km inland reach</div>
                <div>• Bathymetry: Shallow Bay of Bengal shelf amplification active</div>
              </div>
            </div>

            {/* Rainfall & Stall Risk */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md space-y-3">
              <div className="flex items-center gap-2 text-sm font-semibold text-slate-200">
                <CloudRain className="h-4 w-4 text-indigo-400" />
                <span>Hydrologic Stall Amplification</span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                {result?.rainfall_amplification.commentary || "Slowing translation velocity traps feeder rainbands over coastal watersheds, multiplying localized precipitation depth."}
              </p>
              <div className="p-3 rounded-lg bg-slate-950/50 border border-slate-800/80 text-[11px] font-mono text-slate-300 space-y-1">
                <div>• Translation Velocity: {transSpeed} knots</div>
                <div>• Accumulation Scaling: {result?.rainfall_amplification.accumulation_multiplier}x standard rate</div>
                <div>• Catchment Threat: {transSpeed <= 5 ? "Extreme flash flooding & reservoir cresting" : "Managed riverine runoff"}</div>
              </div>
            </div>
          </div>

          {/* SCIENTIFIC RATIONALE COUPLING */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-md space-y-2">
            <h4 className="text-xs font-semibold text-slate-300 flex items-center gap-1.5 uppercase font-mono">
              <Info className="h-3.5 w-3.5 text-cyan-400" />
              <span>Scientific Coupling Equations</span>
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-slate-400 font-mono">
              <div className="p-2.5 rounded bg-slate-950/40 border border-slate-800">
                <span className="text-slate-300 block font-bold mb-1">Kerry Emanuel MPI</span>
                {result?.scientific_rationale.thermodynamic_mpi}
              </div>
              <div className="p-2.5 rounded bg-slate-950/40 border border-slate-800">
                <span className="text-slate-300 block font-bold mb-1">Shear Ventilation</span>
                {result?.scientific_rationale.dynamic_shear}
              </div>
              <div className="p-2.5 rounded bg-slate-950/40 border border-slate-800">
                <span className="text-slate-300 block font-bold mb-1">Atkinson & Holliday</span>
                {result?.scientific_rationale.barometric_coupling}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
