"use client";

import dynamic from "next/dynamic";
import React from "react";
import { Cyclone, TrackPoint } from "@/types/cyclone";
import { PredictionResponse } from "@/types/prediction";
import { RiskAssessment, AffectedDistrict } from "@/types/risk";
import { Loader2 } from "lucide-react";

const CycloneMapComponent = dynamic(
  () => import("./CycloneMap").then((mod) => mod.CycloneMap),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-[560px] lg:h-[640px] rounded-xl border border-command-cardBorder bg-command-card flex flex-col items-center justify-center gap-3 font-mono">
        <Loader2 className="w-8 h-8 text-command-accent animate-spin" />
        <span className="text-xs text-slate-400">INITIALIZING GEODETIC GIS ENGINE (LEAFLET)...</span>
      </div>
    ),
  }
);

interface MapWrapperProps {
  cyclone: Cyclone | null;
  track: TrackPoint[];
  prediction: PredictionResponse | null;
  selectedStep?: string;
  risk?: RiskAssessment | null;
  affectedDistricts?: AffectedDistrict[];
}

export const MapWrapper: React.FC<MapWrapperProps> = (props) => {
  return <CycloneMapComponent {...props} />;
};

