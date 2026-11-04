"use client";

import dynamic from "next/dynamic";
import React from "react";
import { Cyclone } from "@/types/cyclone";
import { CompositeRiskAssessment, RiskMapGeoJSON } from "@/types/impact";
import { Loader2 } from "lucide-react";

const RiskMapComponent = dynamic(
  () => import("./RiskMapInner").then((mod) => mod.RiskMapInner),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-[540px] rounded-xl border border-command-cardBorder bg-command-card flex flex-col items-center justify-center gap-3 font-mono">
        <Loader2 className="w-8 h-8 text-command-accent animate-spin" />
        <span className="text-xs text-slate-400">INITIALIZING DISASTER RISK GIS ENGINE (LEAFLET)...</span>
      </div>
    ),
  }
);

interface RiskMapProps {
  cyclone: Cyclone | null;
  assessment: CompositeRiskAssessment | null;
  geoJsonData: RiskMapGeoJSON | null;
}

export const RiskMap: React.FC<RiskMapProps> = (props) => {
  return <RiskMapComponent {...props} />;
};

