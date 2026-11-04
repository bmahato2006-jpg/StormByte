"use client";

import dynamic from "next/dynamic";
import React from "react";
import { CenterFix, TrajectoryForecast } from "@/types/pipeline";
import { Loader2 } from "lucide-react";

const PipelineMapComponent = dynamic(
  () => import("./PipelineMap").then((mod) => mod.PipelineMap),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-[480px] rounded-xl border border-command-cardBorder bg-command-card flex flex-col items-center justify-center gap-3 font-mono">
        <Loader2 className="w-7 h-7 text-command-accent animate-spin" />
        <span className="text-xs text-slate-400">INITIALIZING LEAFLET GEOSPATIAL MAP...</span>
      </div>
    ),
  }
);

interface PipelineMapWrapperProps {
  centerFix: CenterFix | null;
  forecast: TrajectoryForecast | null;
  trackHistory?: Array<{ latitude: number; longitude: number; timestamp?: string }>;
  requestedLocation?: { latitude: number; longitude: number } | null;
  observationCoverage?: {
    min_latitude: number;
    max_latitude: number;
    min_longitude: number;
    max_longitude: number;
    granule_id?: string;
    satellite?: string;
  } | null;
  showCoverageBounds?: boolean;
  imergObservation?: {
    requested_latitude: number;
    requested_longitude: number;
    grid_latitude?: number | null;
    grid_longitude?: number | null;
    precipitation?: number | null;
    precipitation_mm_per_hr?: number | null;
    unit: string;
    product: string;
    observation_time?: string | null;
    source: string;
  } | null;
  satelliteName?: string | null;
  isLiveNrt?: boolean;
}

export const PipelineMapWrapper: React.FC<PipelineMapWrapperProps> = (props) => {
  return <PipelineMapComponent {...props} />;
};

