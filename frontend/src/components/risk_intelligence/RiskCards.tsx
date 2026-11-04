"use client";

import React from "react";
import {
  Wind,
  CloudRain,
  Waves,
  AlertTriangle,
  Users,
  Building2,
  ShieldCheck,
  CheckCircle2,
  HelpCircle,
} from "lucide-react";
import { CompositeRiskAssessment } from "@/types/impact";

interface RiskCardsProps {
  assessment: CompositeRiskAssessment | null;
}

const getLevelBadge = (level: string) => {
  switch (level?.toUpperCase()) {
    case "EXTREME":
      return "bg-red-50 text-red-700 border-red-200";
    case "HIGH":
      return "bg-amber-50 text-amber-700 border-amber-200";
    case "MODERATE":
      return "bg-yellow-50 text-yellow-800 border-yellow-200";
    case "LOW":
    case "VERY LOW":
      return "bg-emerald-50 text-emerald-700 border-emerald-200";
    default:
      return "bg-slate-100 text-slate-700 border-slate-200";
  }
};

const getSourceBadge = (sourceType: string) => {
  switch (sourceType?.toUpperCase()) {
    case "MEASURED OBSERVATION":
    case "AUTHENTIC OBSERVATION":
      return "bg-cyan-50 text-cyan-700 border-cyan-200";
    case "MODEL-DERIVED":
      return "bg-blue-50 text-blue-700 border-blue-200";
    case "CENSUS OF INDIA COASTAL DISTRICT ATLAS":
      return "bg-indigo-50 text-indigo-700 border-indigo-200";
    case "NATIONAL CRITICAL INFRASTRUCTURE REGISTER":
      return "bg-emerald-50 text-emerald-700 border-emerald-200";
    case "USER PERTURBATION":
      return "bg-amber-50 text-amber-700 border-amber-200";
    case "EXPERIMENTAL":
      return "bg-purple-50 text-purple-700 border-purple-200";
    default:
      return "bg-slate-100 text-slate-600 border-slate-200";
  }
};

export const RiskCards: React.FC<RiskCardsProps> = ({ assessment }) => {
  if (!assessment) return null;

  const {
    windRisk,
    rainfallRisk,
    floodRisk,
    coastalRisk,
    populationExposure,
    infrastructureExposure,
  } = assessment;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 font-mono">
      {/* 1. WIND RISK */}
      <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-xs flex flex-col justify-between space-y-3">
        <div>
          <div className="flex items-center justify-between pb-2 border-b border-slate-200">
            <div className="flex items-center gap-2">
              <Wind className="w-4 h-4 text-blue-700" />
              <span className="text-xs font-bold text-slate-900">WIND RISK</span>
            </div>
            <span className={`text-[10px] px-2 py-0.5 rounded border font-bold ${getLevelBadge(windRisk.riskLevel)}`}>
              {windRisk.riskLevel}
            </span>
          </div>

          <div className="mt-2.5 flex items-center justify-between text-[11px]">
            <span className="text-slate-500 font-medium">DATA STATUS:</span>
            <span className={`px-1.5 py-0.5 rounded text-[9px] border font-bold ${getSourceBadge(windRisk.sourceType)}`}>
              {windRisk.sourceType}
            </span>
          </div>

          <div className="mt-2 text-xs text-slate-700 line-clamp-3 leading-relaxed">
            {windRisk.explanation}
          </div>

          {windRisk.available && windRisk.metrics && (
            <div className="mt-3 p-2.5 rounded-lg bg-slate-50 border border-slate-200 text-[11px] grid grid-cols-2 gap-1.5 text-slate-700">
              <div>Peak Wind: <span className="text-blue-700 font-bold">{windRisk.metrics.max_wind_speed_knots} kt</span></div>
              <div>Velocity: <span className="text-slate-900 font-bold">{windRisk.metrics.max_wind_speed_kmh} km/h</span></div>
              {windRisk.metrics.r34_max_km && (
                <div>R34 Gale: <span className="text-slate-800 font-medium">{windRisk.metrics.r34_max_km} km</span></div>
              )}
              {windRisk.metrics.r64_max_km && (
                <div>R64 Core: <span className="text-red-700 font-bold">{windRisk.metrics.r64_max_km} km</span></div>
              )}
            </div>
          )}
        </div>

        <div className="pt-2 border-t border-slate-100 text-[10px] text-slate-500">
          <div>Source: {windRisk.source}</div>
          {windRisk.disclaimer && (
            <div className="text-amber-800 font-medium mt-1">{windRisk.disclaimer}</div>
          )}
        </div>
      </div>

      {/* 2. RAINFALL RISK */}
      <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-xs flex flex-col justify-between space-y-3">
        <div>
          <div className="flex items-center justify-between pb-2 border-b border-slate-200">
            <div className="flex items-center gap-2">
              <CloudRain className="w-4 h-4 text-blue-700" />
              <span className="text-xs font-bold text-slate-900">RAINFALL RISK</span>
            </div>
            <span className={`text-[10px] px-2 py-0.5 rounded border font-bold ${getLevelBadge(rainfallRisk.riskLevel)}`}>
              {rainfallRisk.riskLevel}
            </span>
          </div>

          <div className="mt-2.5 flex items-center justify-between text-[11px]">
            <span className="text-slate-500 font-medium">DATA STATUS:</span>
            <span className={`px-1.5 py-0.5 rounded text-[9px] border font-bold ${getSourceBadge(rainfallRisk.sourceType)}`}>
              {rainfallRisk.sourceType}
            </span>
          </div>

          <div className="mt-2 text-xs text-slate-700 line-clamp-3 leading-relaxed">
            {rainfallRisk.explanation}
          </div>

          {rainfallRisk.available && rainfallRisk.metrics && (
            <div className="mt-3 p-2.5 rounded-lg bg-slate-50 border border-slate-200 text-[11px] space-y-1 text-slate-700">
              <div>24h Precip: <span className="text-blue-700 font-bold">{rainfallRisk.metrics.accumulated_24h_rainfall_mm} mm</span></div>
              <div>IMD Scale: <span className="text-slate-900 font-bold">{rainfallRisk.metrics.imd_rainfall_classification}</span></div>
            </div>
          )}
        </div>

        <div className="pt-2 border-t border-slate-100 text-[10px] text-slate-500">
          <div>Source: {rainfallRisk.source}</div>
          {rainfallRisk.disclaimer && (
            <div className="text-amber-800 font-medium mt-1">{rainfallRisk.disclaimer}</div>
          )}
        </div>
      </div>

      {/* 3. FLOOD RISK */}
      <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-xs flex flex-col justify-between space-y-3">
        <div>
          <div className="flex items-center justify-between pb-2 border-b border-slate-200">
            <div className="flex items-center gap-2">
              <Waves className="w-4 h-4 text-cyan-700" />
              <span className="text-xs font-bold text-slate-900">FLOOD RISK</span>
            </div>
            <span className={`text-[10px] px-2 py-0.5 rounded border font-bold ${getLevelBadge(floodRisk.riskLevel)}`}>
              {floodRisk.riskLevel}
            </span>
          </div>

          <div className="mt-2.5 flex items-center justify-between text-[11px]">
            <span className="text-slate-500 font-medium">DATA STATUS:</span>
            <span className={`px-1.5 py-0.5 rounded text-[9px] border font-bold ${getSourceBadge(floodRisk.sourceType)}`}>
              {floodRisk.sourceType}
            </span>
          </div>

          <div className="mt-2 text-xs text-slate-700 line-clamp-3 leading-relaxed">
            {floodRisk.explanation}
          </div>

          {floodRisk.available && floodRisk.metrics && (
            <div className="mt-3 p-2.5 rounded-lg bg-slate-50 border border-slate-200 text-[11px] space-y-1 text-slate-700">
              <div>Effective Runoff: <span className="text-cyan-700 font-bold">{floodRisk.metrics.estimated_inundation_potential}</span></div>
              <div>Drainage Stress: <span className="text-slate-900 font-bold">{floodRisk.metrics.drainage_stress_index}</span></div>
            </div>
          )}
        </div>

        <div className="pt-2 border-t border-slate-100 text-[10px] text-slate-500">
          <div>Method: {floodRisk.method}</div>
          {floodRisk.disclaimer && (
            <div className="text-amber-800 font-medium mt-1">{floodRisk.disclaimer}</div>
          )}
        </div>
      </div>

      {/* 4. COASTAL STORM SURGE */}
      <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-xs flex flex-col justify-between space-y-3">
        <div>
          <div className="flex items-center justify-between pb-2 border-b border-slate-200">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-purple-700" />
              <span className="text-xs font-bold text-slate-900">COASTAL STORM SURGE</span>
            </div>
            <span className={`text-[10px] px-2 py-0.5 rounded border font-bold ${getLevelBadge(coastalRisk.riskLevel)}`}>
              {coastalRisk.riskLevel}
            </span>
          </div>

          <div className="mt-2.5 flex items-center justify-between text-[11px]">
            <span className="text-slate-500 font-medium">DATA STATUS:</span>
            <span className={`px-1.5 py-0.5 rounded text-[9px] border font-bold ${getSourceBadge(coastalRisk.sourceType)}`}>
              {coastalRisk.sourceType}
            </span>
          </div>

          <div className="mt-2 text-xs text-slate-700 line-clamp-3 leading-relaxed">
            {coastalRisk.explanation}
          </div>

          {coastalRisk.available && coastalRisk.metrics && (
            <div className="mt-3 p-2.5 rounded-lg bg-slate-50 border border-slate-200 text-[11px] grid grid-cols-2 gap-1.5 text-slate-700">
              <div>Peak Surge: <span className="text-purple-700 font-bold">{coastalRisk.metrics.estimated_surge_peak_meters} m</span></div>
              <div>Barometer IB: <span className="text-slate-900">{coastalRisk.metrics.inverted_barometer_component_m ?? "N/A"} m</span></div>
              <div>Wind Setup: <span className="text-slate-900">{coastalRisk.metrics.shelf_wind_setup_component_m ?? "N/A"} m</span></div>
              <div>Coast Dist: <span className="text-slate-700">{coastalRisk.metrics.assumed_distance_to_coast_km ?? "N/A"} km</span></div>
            </div>
          )}
        </div>

        <div className="pt-2 border-t border-slate-100 text-[10px] text-slate-500">
          <div>Source: {coastalRisk.source}</div>
          {coastalRisk.disclaimer && (
            <div className="text-amber-800 font-medium mt-1">{coastalRisk.disclaimer}</div>
          )}
        </div>
      </div>

      {/* 5. POPULATION EXPOSURE */}
      <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-xs flex flex-col justify-between space-y-3">
        <div>
          <div className="flex items-center justify-between pb-2 border-b border-slate-200">
            <div className="flex items-center gap-2">
              <Users className="w-4 h-4 text-amber-700" />
              <span className="text-xs font-bold text-slate-900">POPULATION EXPOSURE</span>
            </div>
            <span
              className={`text-[10px] px-2 py-0.5 rounded border font-bold ${
                populationExposure.available && (populationExposure.totalExposedPopulation ?? 0) > 0
                  ? "bg-amber-50 text-amber-800 border-amber-200"
                  : "bg-emerald-50 text-emerald-700 border-emerald-200"
              }`}
            >
              {populationExposure.available && populationExposure.settlements.length > 0
                ? `${populationExposure.settlements.length} DISTRICTS`
                : "0 DISTRICTS (CALM / CLEAR)"}
            </span>
          </div>

          <div className="mt-2.5 flex items-center justify-between text-[11px]">
            <span className="text-slate-500 font-medium">DATA STATUS:</span>
            <span className={`px-1.5 py-0.5 rounded text-[9px] border font-bold ${getSourceBadge(populationExposure.sourceType)}`}>
              {populationExposure.sourceType}
            </span>
          </div>

          <div className="mt-2 text-xs text-slate-700 line-clamp-3 leading-relaxed">
            {populationExposure.explanation}
          </div>

          {populationExposure.available && (
            <div className="mt-3 p-2.5 rounded-lg bg-slate-50 border border-slate-200 text-[11px] space-y-1 text-slate-700">
              <div>Total Exposed Pop: <span className="text-amber-800 font-bold">~{(populationExposure.totalExposedPopulation || 0).toLocaleString()}</span></div>
              <div className="text-[10px] text-slate-600 truncate">
                Loci: {populationExposure.settlements.length > 0 ? populationExposure.settlements.map((s) => s.name).join(", ") : "None (No active gale radii)"}
              </div>
            </div>
          )}
        </div>

        <div className="pt-2 border-t border-slate-100 text-[10px] text-slate-500">
          <div>Source: {populationExposure.source}</div>
          {populationExposure.disclaimer && (
            <div className="text-amber-800 font-medium mt-1">{populationExposure.disclaimer}</div>
          )}
        </div>
      </div>

      {/* 6. INFRASTRUCTURE EXPOSURE */}
      <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-xs flex flex-col justify-between space-y-3">
        <div>
          <div className="flex items-center justify-between pb-2 border-b border-slate-200">
            <div className="flex items-center gap-2">
              <Building2 className="w-4 h-4 text-emerald-700" />
              <span className="text-xs font-bold text-slate-900">CRITICAL INFRASTRUCTURE</span>
            </div>
            <span
              className={`text-[10px] px-2 py-0.5 rounded border font-bold ${
                infrastructureExposure.available && infrastructureExposure.criticalCount > 0
                  ? "bg-amber-50 text-amber-800 border-amber-200"
                  : "bg-emerald-50 text-emerald-700 border-emerald-200"
              }`}
            >
              {infrastructureExposure.available && infrastructureExposure.criticalCount > 0
                ? `${infrastructureExposure.criticalCount} ASSETS`
                : "0 ASSETS (CLEAR / SAFE)"}
            </span>
          </div>

          <div className="mt-2.5 flex items-center justify-between text-[11px]">
            <span className="text-slate-500 font-medium">DATA STATUS:</span>
            <span className={`px-1.5 py-0.5 rounded text-[9px] border font-bold ${getSourceBadge(infrastructureExposure.sourceType)}`}>
              {infrastructureExposure.sourceType}
            </span>
          </div>

          <div className="mt-2 text-xs text-slate-700 line-clamp-3 leading-relaxed">
            {infrastructureExposure.explanation}
          </div>

          {infrastructureExposure.available && infrastructureExposure.items && (
            <div className="mt-3 p-2.5 rounded-lg bg-slate-50 border border-slate-200 text-[11px] space-y-1 text-slate-700">
              <div>Critical Facilities: <span className="text-emerald-700 font-bold">{infrastructureExposure.criticalCount} identified</span></div>
              <div className="text-[10px] text-slate-600 truncate">
                Includes: {infrastructureExposure.items.length > 0 ? infrastructureExposure.items.map((i) => `${i.name} (${i.type})`).join(", ") : "None within warning contour"}
              </div>
            </div>
          )}
        </div>

        <div className="pt-2 border-t border-slate-100 text-[10px] text-slate-500">
          <div>Source: {infrastructureExposure.source}</div>
          {infrastructureExposure.disclaimer && (
            <div className="text-amber-800 font-medium mt-1">{infrastructureExposure.disclaimer}</div>
          )}
        </div>
      </div>
    </div>
  );
};

