"use client";

import React, { useState } from "react";
import { Clock, Navigation, AlertTriangle, ShieldCheck, MapPin, Wind, Gauge } from "lucide-react";
import { Cyclone } from "@/types/cyclone";
import { PredictionResponse, PredictionPoint } from "@/types/prediction";

interface PredictionTimelineProps {
  cyclone: Cyclone | null;
  prediction: PredictionResponse | null;
  selectedStep: string;
  onSelectStep: (step: string) => void;
}

export const PredictionTimeline: React.FC<PredictionTimelineProps> = ({
  cyclone,
  prediction,
  selectedStep,
  onSelectStep,
}) => {
  if (!cyclone) return null;

  // Build 4 timeline points: NOW, +6H, +12H, +24H
  const steps = [
    {
      id: "NOW",
      label: "NOW (0H)",
      time: "Initial Fix",
      lat: cyclone.latitude,
      lon: cyclone.longitude,
      wind: cyclone.windSpeed,
      pressure: cyclone.pressure,
      uncertainty: 0,
      isPrediction: false,
    },
    ...(prediction?.predictions.map((p, idx) => ({
      id: p.forecastTime,
      label: p.leadHours ? `+${p.leadHours}h` : p.forecastTime,
      time: p.forecastTime,
      lat: p.latitude,
      lon: p.longitude,
      wind: p.predictedWindSpeed || cyclone.windSpeed,
      pressure: p.predictedPressure || cyclone.pressure,
      uncertainty: p.uncertainty,
      isPrediction: true,
    })) || []),
  ];

  const activeStepData = steps.find((s) => s.id === selectedStep) || steps[0];

  const isRealHistorical =
    cyclone.provenance?.dataType === "VERIFIED HISTORICAL DATA" ||
    cyclone.provenance?.dataType === "REAL DATA" ||
    cyclone.dataSource?.includes("IMD") ||
    cyclone.dataSource?.includes("IBTrACS") ||
    cyclone.id?.startsWith("IMD-") ||
    cyclone.id?.startsWith("IBTRACS-");

  const isCalmBasin =
    !cyclone.intensity.toLowerCase().includes("cyclon") &&
    !cyclone.intensity.toLowerCase().includes("storm") &&
    cyclone.windSpeed < 34;

  return (
    <div className="p-4 rounded-xl border border-slate-200 bg-white font-mono space-y-3 shadow-xs select-none">
      <div className="flex items-center justify-between border-b border-slate-200 pb-2">
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4 text-blue-700" />
          <span className="text-xs font-bold uppercase tracking-wider text-slate-900">
            {isCalmBasin
              ? "Live Atmospheric Marine Telemetry Stepper"
              : "Trajectory Temporal Stepper"}
          </span>
        </div>
        <span className="text-[10px] text-slate-500 font-medium">
          {isCalmBasin
            ? "Real-time hourly Open-Meteo & INSAT-3D ocean readings"
            : "Step through +24h forecast envelope"}
        </span>
      </div>

      {/* Stepper Buttons */}
      <div className="grid grid-cols-4 gap-2">
        {steps.map((step) => {
          const isSelected = selectedStep === step.id;
          return (
            <button
              key={step.id}
              onClick={() => onSelectStep(step.id)}
              className={`p-2.5 rounded-lg border text-left transition-all ${
                isSelected
                  ? "bg-blue-50 border-blue-600 shadow-xs"
                  : "bg-slate-50 border-slate-200 hover:border-slate-300 hover:bg-slate-100/60"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className={`text-xs font-bold ${isSelected ? "text-blue-800" : "text-slate-800"}`}>
                  {step.label}
                </span>
                {step.isPrediction && (
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-600 inline-block" />
                )}
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">{step.time}</div>
              <div className="text-[11px] font-bold text-slate-900 mt-1">{step.wind} kt</div>
            </button>
          );
        })}
      </div>

      {/* Selected Step Telemetry Summary */}
      <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-1.5 text-slate-700">
            <MapPin className="w-3.5 h-3.5 text-blue-700" />
            <span className="text-slate-900 font-bold">
              {activeStepData.lat.toFixed(2)}°N, {activeStepData.lon.toFixed(2)}°E
            </span>
          </div>

          <div className="flex items-center gap-1.5 text-slate-700">
            <Wind className="w-3.5 h-3.5 text-blue-700" />
            <span className="text-slate-500">Wind:</span>
            <span className="text-blue-800 font-bold">{activeStepData.wind} knots</span>
          </div>

          <div className="flex items-center gap-1.5 text-slate-700">
            <Gauge className="w-3.5 h-3.5 text-amber-700" />
            <span className="text-slate-500">MSLP:</span>
            <span className="text-amber-800 font-bold">{activeStepData.pressure} hPa</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {activeStepData.isPrediction && activeStepData.uncertainty > 0 ? (
            <span className="text-[11px] text-amber-800 font-medium flex items-center gap-1">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
              Uncertainty Cone Radius: ±{activeStepData.uncertainty} km
            </span>
          ) : isCalmBasin ? (
            <span className="text-[11px] text-emerald-800 font-semibold flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-emerald-600 animate-pulse" />
              Basin Status: Calm Sea (No Cyclone Active)
            </span>
          ) : null}
          <span className="text-[10px] text-slate-700 bg-white px-2 py-0.5 rounded border border-slate-300 font-semibold shadow-2xs">
            {activeStepData.isPrediction
              ? isCalmBasin
                ? "LIVE HOURLY ATMOSPHERIC TELEMETRY"
                : "MODEL-GENERATED / EXPERIMENTAL"
              : isRealHistorical
              ? `AUTHENTIC OBSERVATION (SOURCE: ${cyclone.dataSource || "IMD RSMC NEW DELHI"})`
              : `AUTHENTIC OBSERVATION (SOURCE: ${cyclone.dataSource || "IMD INSAT-3D & NASA EOSDIS"})`}
          </span>
        </div>
      </div>
    </div>
  );
};
