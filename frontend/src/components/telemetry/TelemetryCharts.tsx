"use client";

import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { TrackPoint } from "@/types/cyclone";
import { PredictionResponse } from "@/types/prediction";
import { TrendingUp, Radio } from "lucide-react";

interface TelemetryChartsProps {
  track: TrackPoint[];
  prediction: PredictionResponse | null;
}

export const TelemetryCharts: React.FC<TelemetryChartsProps> = ({ track, prediction }) => {
  // Combine historical and projected data points for a smooth timeline
  const chartData = [
    ...track.map((pt) => ({
      time: new Date(pt.timestamp).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        hour: "2-digit",
      }),
      observedWind: pt.windSpeed,
      observedPressure: pt.pressure,
      projectedWind: null,
      projectedPressure: null,
      type: "OBSERVED",
    })),
    ...(prediction?.predictions.map((p) => ({
      time: p.forecastTime,
      observedWind: null,
      observedPressure: null,
      projectedWind: p.predictedWindSpeed || null,
      projectedPressure: p.predictedPressure || null,
      type: "PROJECTED",
    })) || []),
  ];

  return (
    <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-xs font-sans">
      {/* Chart Header */}
      <div className="flex flex-wrap items-center justify-between border-b border-slate-200 pb-2.5 mb-3 gap-2">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-blue-600 shrink-0" />
          <div>
            <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wide">
              Atmospheric Telemetry Timeline
            </h3>
            <span className="text-[10px] text-slate-500 block -mt-0.5">
              48-Hour Continuous Sustained Wind (kt) &amp; Surface Pressure (hPa) Series
            </span>
          </div>
        </div>

        {/* Legend Indicators */}
        <div className="flex items-center gap-3 text-[11px] font-mono text-slate-600">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-1 bg-blue-600 inline-block rounded" />
            <span className="font-semibold text-slate-700">Wind (knots)</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-1 bg-amber-600 inline-block rounded" />
            <span className="font-semibold text-slate-700">Pressure (hPa)</span>
          </span>
        </div>
      </div>

      {/* Responsive Line Chart Frame */}
      <div className="w-full h-52 flex items-center justify-center">
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
              <CartesianGrid strokeDasharray="2 2" stroke="#e2e8f0" />
              <XAxis
                dataKey="time"
                stroke="#94a3b8"
                tick={{ fontSize: 10, fontFamily: "monospace", fill: "#475569" }}
                interval={chartData.length > 20 ? 4 : "preserveEnd"}
              />
              <YAxis
                yAxisId="left"
                stroke="#0284c7"
                domain={["auto", "auto"]}
                tick={{ fontSize: 10, fontFamily: "monospace", fill: "#0284c7" }}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                stroke="#d97706"
                domain={["auto", "auto"]}
                tick={{ fontSize: 10, fontFamily: "monospace", fill: "#d97706" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#ffffff",
                  borderColor: "#cbd5e1",
                  borderRadius: "6px",
                  fontFamily: "monospace",
                  fontSize: "11px",
                  color: "#0f172a",
                  boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
                }}
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="observedWind"
                stroke="#0284c7"
                strokeWidth={2.5}
                dot={chartData.length > 20 ? false : { r: 2.5, fill: "#0284c7" }}
                activeDot={{ r: 4 }}
                name="Observed Wind (kt)"
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="projectedWind"
                stroke="#0284c7"
                strokeWidth={2.5}
                strokeDasharray="4 4"
                dot={chartData.length > 20 ? false : { r: 2.5, fill: "#0369a1" }}
                activeDot={{ r: 4 }}
                name="Forecast Wind (kt)"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="observedPressure"
                stroke="#d97706"
                strokeWidth={2.5}
                dot={chartData.length > 20 ? false : { r: 2.5, fill: "#d97706" }}
                activeDot={{ r: 4 }}
                name="Observed MSLP (hPa)"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="projectedPressure"
                stroke="#d97706"
                strokeWidth={2.5}
                strokeDasharray="4 4"
                dot={chartData.length > 20 ? false : { r: 2.5, fill: "#b45309" }}
                activeDot={{ r: 4 }}
                name="Forecast MSLP (hPa)"
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex flex-col items-center justify-center text-slate-500 font-mono text-xs gap-2 py-8">
            <Radio className="w-5 h-5 text-blue-600 animate-pulse" />
            <span>Synchronizing atmospheric telemetry stream...</span>
          </div>
        )}
      </div>

      {/* Axis Footer */}
      <div className="flex items-center justify-between text-[10px] font-mono text-slate-500 pt-2 border-t border-slate-200 mt-1">
        <span className="text-blue-700 font-bold">LEFT AXIS: SUSTAINED WIND (KNOTS)</span>
        <span className="text-slate-500">Source: Open-Meteo High-Resolution Atmospheric Model</span>
        <span className="text-amber-700 font-bold">RIGHT AXIS: CENTRAL PRESSURE (hPa)</span>
      </div>
    </div>
  );
};
