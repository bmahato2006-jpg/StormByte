export const SCIENTIFIC_DISCLAIMER =
  "EXPERIMENTAL RESEARCH SYSTEM - NOT AN OFFICIAL WEATHER FORECAST. DOES NOT REPLACE ADVISORIES FROM IMD, NHC, OR RSMC. ALL DATA LABELED ACCORDING TO PROVENANCE.";

export const CATEGORY_COLORS: Record<string, { badge: string; border: string; glow: string; text: string }> = {
  "Super Cyclonic Storm": {
    badge: "bg-pink-100 text-pink-900 border-pink-300",
    border: "#db2777",
    glow: "rgba(219, 39, 119, 0.2)",
    text: "text-pink-900 font-bold",
  },
  "Extremely Severe Cyclonic Storm": {
    badge: "bg-red-100 text-red-900 border-red-300",
    border: "#dc2626",
    glow: "rgba(220, 38, 38, 0.2)",
    text: "text-red-900 font-bold",
  },
  "Very Severe Cyclonic Storm": {
    badge: "bg-orange-100 text-orange-900 border-orange-300",
    border: "#ea580c",
    glow: "rgba(234, 88, 12, 0.2)",
    text: "text-orange-900 font-bold",
  },
  "Severe Cyclonic Storm": {
    badge: "bg-amber-100 text-amber-950 border-amber-300",
    border: "#d97706",
    glow: "rgba(217, 119, 6, 0.2)",
    text: "text-amber-950 font-bold",
  },
  "Cyclonic Storm": {
    badge: "bg-sky-100 text-sky-950 border-sky-300",
    border: "#0284c7",
    glow: "rgba(2, 132, 199, 0.2)",
    text: "text-sky-950 font-bold",
  },
  "Deep Depression": {
    badge: "bg-blue-100 text-blue-950 border-blue-300",
    border: "#2563eb",
    glow: "rgba(37, 99, 235, 0.2)",
    text: "text-blue-950 font-bold",
  },
  "Depression": {
    badge: "bg-slate-100 text-slate-800 border-slate-300",
    border: "#64748b",
    glow: "rgba(100, 116, 139, 0.2)",
    text: "text-slate-800 font-bold",
  },
  "Basin Quiet • Satellite Monitoring": {
    badge: "bg-white text-emerald-950 border-emerald-300 shadow-2xs",
    border: "#10b981",
    glow: "rgba(16, 185, 129, 0.15)",
    text: "text-emerald-950 font-bold",
  },
  "Basin Quiet": {
    badge: "bg-white text-emerald-950 border-emerald-300 shadow-2xs",
    border: "#10b981",
    glow: "rgba(16, 185, 129, 0.15)",
    text: "text-emerald-950 font-bold",
  },
};

export const RISK_LEVEL_COLORS: Record<string, string> = {
  EXTREME: "bg-red-100 text-red-900 border-red-300 font-bold",
  HIGH: "bg-orange-100 text-orange-900 border-orange-300 font-bold",
  MODERATE: "bg-amber-100 text-amber-950 border-amber-300 font-bold",
  LOW: "bg-emerald-100 text-emerald-900 border-emerald-300 font-bold",
};

