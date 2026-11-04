"use client";

import React, { useState, useEffect } from "react";
import {
  LayoutDashboard,
  Radar,
  Satellite,
  Navigation,
  ShieldAlert,
  Cpu,
  Sliders,
  History,
  LineChart,
  BrainCircuit,
  FileText,
  Activity,
  Database,
  ChevronDown,
  ChevronRight,
} from "lucide-react";

export type NavView =
  | "dashboard"
  | "historical_satellite"
  | "ai_pipeline"
  | "datasets"
  | "data_intelligence"
  | "satellite_ai"
  | "monitor"
  | "satellite"
  | "track"
  | "risk"
  | "digital_twin"
  | "simulator"
  | "replay"
  | "performance"
  | "analyst"
  | "reports"
  | "health";

interface SidebarProps {
  activeView: NavView;
  onSelectView: (view: NavView) => void;
  isBackendHealthy: boolean;
}

interface NavItem {
  id: NavView;
  label: string;
  icon: React.ElementType;
  badge?: string;
  badgeColor?: string;
}

// Clean, streamlined primary navigation
const PRIMARY_NAV_ITEMS: NavItem[] = [
  {
    id: "satellite_ai",
    label: "Live Satellite AI",
    icon: Satellite,
    badge: "INSAT-3D",
    badgeColor: "text-blue-800 border-blue-300 bg-blue-100",
  },
  {
    id: "dashboard",
    label: "Operations Dashboard",
    icon: LayoutDashboard,
  },
  {
    id: "track",
    label: "Track & Trajectory",
    icon: Navigation,
    badge: "TRANSFORMER V1",
    badgeColor: "text-indigo-800 border-indigo-300 bg-indigo-100",
  },
  {
    id: "risk",
    label: "Risk & Impact Assessment",
    icon: ShieldAlert,
  },
  {
    id: "ai_pipeline",
    label: "Scientific AI Pipeline",
    icon: BrainCircuit,
    badge: "CNN V3",
    badgeColor: "text-indigo-800 border-indigo-300 bg-indigo-100",
  },
  {
    id: "datasets",
    label: "Data & Provenance Hub",
    icon: Database,
    badge: "IMD / NOAA",
    badgeColor: "text-amber-900 border-amber-300 bg-amber-100",
  },
  {
    id: "health",
    label: "System Diagnostics",
    icon: Activity,
  },
];

// Secondary analytical modules
const SECONDARY_NAV_ITEMS: NavItem[] = [
  { id: "historical_satellite", label: "Historical Satellite Archive", icon: History },
  { id: "data_intelligence", label: "Data Intelligence & Schema", icon: Database },
  { id: "monitor", label: "Radar Coastal Network", icon: Radar },
  { id: "digital_twin", label: "Digital Twin Vortex Physics", icon: Cpu },
  { id: "simulator", label: "Scenario Simulation Engine", icon: Sliders },
  { id: "replay", label: "Historical Track Replay", icon: History },
  { id: "performance", label: "Model Verification Metrics", icon: LineChart },
  { id: "reports", label: "Disaster Briefing Generator", icon: FileText },
];

export const Sidebar: React.FC<SidebarProps> = ({
  activeView,
  onSelectView,
  isBackendHealthy,
}) => {
  const isSecondaryActive = SECONDARY_NAV_ITEMS.some((item) => item.id === activeView);
  const [isSecondaryOpen, setIsSecondaryOpen] = useState<boolean>(isSecondaryActive);

  useEffect(() => {
    if (isSecondaryActive) {
      setIsSecondaryOpen(true);
    }
  }, [isSecondaryActive]);

  return (
    <aside className="w-64 bg-white border-r border-slate-200 flex flex-col shrink-0 select-none font-sans shadow-xs">
      {/* Portal Directory Header */}
      <div className="px-4 py-3 border-b border-slate-200 bg-[#f8fafc] flex items-center justify-between">
        <span className="text-[11px] font-bold tracking-wider text-slate-700 uppercase font-mono">
          Portal Navigation
        </span>
        <span className="text-[10px] text-slate-500 font-mono">v2.1</span>
      </div>

      {/* Primary Nav List */}
      <nav className="flex-1 overflow-y-auto p-2.5 space-y-1">
        <div className="px-2 pt-1 pb-1 text-[10px] font-bold tracking-wider text-slate-500 uppercase font-mono">
          Operational Services
        </div>

        {PRIMARY_NAV_ITEMS.map((item) => {
          const isActive = activeView === item.id;
          const Icon = item.icon;

          return (
            <button
              key={item.id}
              onClick={() => onSelectView(item.id)}
              className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                isActive
                  ? "bg-blue-50 text-blue-700 font-bold border border-blue-200 shadow-2xs"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-100 border border-transparent"
              }`}
            >
              <div className="flex items-center gap-2.5">
                <Icon
                  className={`w-4 h-4 shrink-0 ${
                    isActive ? "text-blue-600" : "text-slate-400"
                  }`}
                />
                <span className="truncate">{item.label}</span>
              </div>
              {item.badge && (
                <span
                  className={`px-1.5 py-0.5 rounded text-[9px] font-mono border font-bold ${
                    item.badgeColor || "text-slate-600 border-slate-300 bg-slate-100"
                  }`}
                >
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}

        {/* Collapsible Secondary Research Modules */}
        <div className="pt-2">
          <button
            type="button"
            onClick={() => setIsSecondaryOpen(!isSecondaryOpen)}
            className="w-full flex items-center justify-between px-2 py-1.5 text-[10px] font-bold tracking-wider text-slate-500 uppercase hover:text-slate-800 transition-colors font-mono"
          >
            <span>Research &amp; Diagnostics</span>
            {isSecondaryOpen ? (
              <ChevronDown className="w-3.5 h-3.5" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5" />
            )}
          </button>

          {isSecondaryOpen && (
            <div className="mt-1 pl-1 space-y-0.5 border-l border-slate-200 ml-2">
              {SECONDARY_NAV_ITEMS.map((item) => {
                const isActive = activeView === item.id;
                const Icon = item.icon;

                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => onSelectView(item.id)}
                    className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      isActive
                        ? "bg-blue-50 text-blue-700 font-bold border border-blue-200 shadow-2xs"
                        : "text-slate-600 hover:text-slate-900 hover:bg-slate-100 border border-transparent"
                    }`}
                  >
                    <div className="flex items-center gap-2 truncate">
                      <Icon
                        className={`w-3.5 h-3.5 shrink-0 ${
                          isActive ? "text-blue-600" : "text-slate-400"
                        }`}
                      />
                      <span className="truncate">{item.label}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </nav>

      {/* Bottom Official Disclaimer Strip */}
      <div className="p-3 border-t border-slate-200 bg-[#f8fafc] text-[10px] text-slate-600 space-y-1">
        <div className="flex items-center justify-between">
          <span className="text-slate-700 font-medium font-mono text-[11px]">Gateway Service</span>
          <span className="flex items-center gap-1.5 font-mono text-emerald-700 font-bold">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 animate-pulse" />
            {isBackendHealthy ? "ONLINE (200 OK)" : "OFFLINE"}
          </span>
        </div>
        <div className="text-[9px] text-slate-500 pt-1 border-t border-slate-200 leading-tight font-mono">
          CYCLONE-X Intelligence Platform
        </div>
      </div>
    </aside>
  );
};
