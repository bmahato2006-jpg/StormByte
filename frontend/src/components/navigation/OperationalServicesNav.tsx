"use client";

import React, { useState } from "react";
import {
  Satellite,
  LayoutDashboard,
  Navigation,
  ShieldAlert,
  BrainCircuit,
  Database,
  Cpu,
  Radar,
  Radio,
} from "lucide-react";

export type OperationalServiceId =
  | "satellite_ai"
  | "dashboard"
  | "track"
  | "risk"
  | "ai_pipeline"
  | "datasets"
  | "data_intelligence"
  | "local_radar";

export interface OperationalServiceItem {
  id: OperationalServiceId;
  label: string;
  subLabel?: string;
  icon: React.ElementType;
  badge?: string;
  badgeVariant?: "blue" | "indigo" | "amber" | "emerald" | "rose" | "cyan";
  isLive?: boolean;
}

export const OPERATIONAL_SERVICES: OperationalServiceItem[] = [
  {
    id: "satellite_ai",
    label: "Live Satellite AI",
    subLabel: "INSAT-3D",
    icon: Satellite,
    badge: "INSAT-3D",
    badgeVariant: "blue",
    isLive: true,
  },
  {
    id: "dashboard",
    label: "Operations Dashboard",
    icon: LayoutDashboard,
  },
  {
    id: "track",
    label: "Track & Trajectory",
    subLabel: "TRANSFORMER",
    icon: Navigation,
    badge: "TRANSFORMER",
    badgeVariant: "indigo",
  },
  {
    id: "risk",
    label: "Risk & Impact",
    icon: ShieldAlert,
    badgeVariant: "rose",
  },
  {
    id: "ai_pipeline",
    label: "Scientific AI Pipeline",
    subLabel: "CNN V3",
    icon: BrainCircuit,
    badge: "CNN V3",
    badgeVariant: "indigo",
  },
  {
    id: "datasets",
    label: "Data & Provenance Hub",
    subLabel: "IMD/NOAA",
    icon: Database,
    badge: "IMD/NOAA",
    badgeVariant: "amber",
  },
  {
    id: "data_intelligence",
    label: "Data Intelligence",
    icon: Cpu,
    badgeVariant: "cyan",
  },
  {
    id: "local_radar",
    label: "Live Location Radar",
    subLabel: "DWR NRT",
    icon: Radar,
    badge: "DWR NRT",
    badgeVariant: "emerald",
    isLive: true,
  },
];

interface OperationalServicesNavProps {
  activeService?: OperationalServiceId;
  onSelectService?: (serviceId: OperationalServiceId) => void;
  className?: string;
}

export const OperationalServicesNav: React.FC<OperationalServicesNavProps> = ({
  activeService: controlledActiveService,
  onSelectService,
  className = "",
}) => {
  const [internalActiveService, setInternalActiveService] =
    useState<OperationalServiceId>("satellite_ai");

  const currentActive = controlledActiveService ?? internalActiveService;

  const handleSelect = (serviceId: OperationalServiceId) => {
    if (controlledActiveService === undefined) {
      setInternalActiveService(serviceId);
    }
    onSelectService?.(serviceId);
  };

  const getBadgeStyle = (
    variant?: OperationalServiceItem["badgeVariant"],
    isActive?: boolean
  ) => {
    if (isActive) {
      return "bg-cyan-500/20 text-cyan-200 border-cyan-400/40 shadow-[0_0_8px_rgba(6,182,212,0.25)]";
    }
    switch (variant) {
      case "emerald":
        return "bg-emerald-500/15 text-emerald-300 border-emerald-500/30";
      case "blue":
        return "bg-sky-500/15 text-sky-300 border-sky-500/30";
      case "indigo":
        return "bg-indigo-500/15 text-indigo-300 border-indigo-500/30";
      case "amber":
        return "bg-amber-500/15 text-amber-300 border-amber-500/30";
      case "rose":
        return "bg-rose-500/15 text-rose-300 border-rose-500/30";
      case "cyan":
      default:
        return "bg-slate-800 text-slate-300 border-slate-700";
    }
  };

  return (
    <nav
      aria-label="Operational Services"
      className={`w-full max-w-full select-none overflow-x-hidden border-b border-slate-800/80 bg-slate-950/90 shadow-lg backdrop-blur-md transition-colors ${className}`}
    >
      <div className="mx-auto flex h-12 w-full max-w-[1920px] items-center justify-between px-2 sm:px-3 lg:px-4">
        {/* Left Branding Tag: ISRO / IMD Scientific Indicator */}
        <div className="hidden 2xl:flex items-center gap-2 border-r border-slate-800 pr-3 mr-2 shrink-0">
          <div className="flex h-6 w-6 items-center justify-center rounded bg-slate-900 border border-slate-800 text-cyan-400 shadow-inner">
            <Radio className="h-3.5 w-3.5 animate-pulse text-cyan-400" />
          </div>
          <div className="leading-tight">
            <div className="text-[9px] font-mono font-bold tracking-widest text-slate-400 uppercase">
              GRID OPS
            </div>
            <div className="text-[10px] font-mono font-black text-cyan-400">
              8 SERVICES
            </div>
          </div>
        </div>

        {/* 8 Operational Services in a Strict Single Row (Zero Horizontal Overflow) */}
        <div className="flex flex-1 items-center justify-between sm:justify-evenly gap-x-1 sm:gap-x-1.5 lg:gap-x-2 min-w-0">
          {OPERATIONAL_SERVICES.map((service) => {
            const isActive = currentActive === service.id;
            const Icon = service.icon;

            return (
              <button
                key={service.id}
                type="button"
                onClick={() => handleSelect(service.id)}
                className={`group relative flex flex-1 items-center justify-center gap-1.5 rounded-lg px-1.5 py-1.5 sm:px-2 sm:py-1.5 transition-all duration-200 outline-none select-none min-w-0 ${
                  isActive
                    ? "border border-cyan-500/50 bg-gradient-to-b from-cyan-950/60 to-slate-900/90 text-white shadow-[0_0_15px_rgba(6,182,212,0.18)]"
                    : "border border-transparent text-slate-400 hover:border-slate-800 hover:bg-slate-900/60 hover:text-slate-200"
                }`}
              >
                {/* Active Indicator Top Glow Line */}
                {isActive && (
                  <span className="absolute -top-[1px] left-2 right-2 h-[2px] rounded-full bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_8px_#22d3ee]" />
                )}

                {/* Service Icon with Micro Indicator */}
                <div className="relative shrink-0 flex items-center justify-center">
                  <Icon
                    className={`h-3.5 w-3.5 sm:h-4 sm:w-4 transition-transform duration-200 group-hover:scale-110 ${
                      isActive
                        ? "text-cyan-400"
                        : "text-slate-400 group-hover:text-slate-200"
                    }`}
                  />
                  {service.isLive && (
                    <span className="absolute -top-0.5 -right-0.5 flex h-1.5 w-1.5">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                      <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500" />
                    </span>
                  )}
                </div>

                {/* Service Label Text */}
                <span
                  className={`truncate text-[11px] sm:text-xs font-semibold tracking-tight transition-colors ${
                    isActive
                      ? "text-cyan-100 font-bold"
                      : "text-slate-300 group-hover:text-white"
                  }`}
                >
                  {service.label}
                </span>

                {/* Scientific Metadata / Version Badge */}
                {service.badge && (
                  <span
                    className={`hidden xl:inline-flex items-center px-1.5 py-0.2 rounded text-[9px] font-mono font-bold tracking-tight border uppercase shrink-0 transition-colors ${getBadgeStyle(
                      service.badgeVariant,
                      isActive
                    )}`}
                  >
                    {service.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </nav>
  );
};

export default OperationalServicesNav;
