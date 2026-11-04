"use client";

import React, { useEffect, useState } from "react";
import { Clock, Satellite } from "lucide-react";

interface GovHeaderProps {
  isBackendHealthy?: boolean;
}

export const GovHeader: React.FC<GovHeaderProps> = ({ isBackendHealthy = true }) => {
  const [currentTimeUtc, setCurrentTimeUtc] = useState<string>("");
  const [currentTimeIst, setCurrentTimeIst] = useState<string>("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setCurrentTimeUtc(now.toUTCString().replace("GMT", "UTC"));
      setCurrentTimeIst(
        now.toLocaleTimeString("en-IN", {
          timeZone: "Asia/Kolkata",
          hour12: false,
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        }) + " IST"
      );
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="w-full bg-white border-b border-slate-200 select-none shadow-xs">
      {/* 1. National Tricolor Ribbon */}
      <div className="w-full h-1 bg-gradient-to-r from-[#FF9933] via-white to-[#138808]" />

      {/* 2. Top Government Utility Strip */}
      <div className="bg-[#f8fafc] border-b border-slate-200 px-6 py-1.5 flex items-center justify-between text-[11px] font-sans text-slate-700">
        <div className="flex items-center gap-3">
          <span className="font-bold tracking-wide text-slate-800 font-mono text-[11px] uppercase">
            National Disaster Early Warning Portal
          </span>
        </div>

        <div className="flex items-center gap-4 font-mono text-[11px]">
          <div className="hidden sm:flex items-center gap-1.5 text-slate-600">
            <Clock className="w-3 h-3 text-blue-600" />
            <span className="text-slate-800 font-bold">{currentTimeIst || "IST"}</span>
            <span className="text-slate-400">/</span>
            <span className="text-slate-600">{currentTimeUtc ? currentTimeUtc.slice(17, 25) + " UTC" : "UTC"}</span>
          </div>
          <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded bg-white border border-slate-200 shadow-2xs">
            <span
              className={`w-2 h-2 rounded-full ${
                isBackendHealthy ? "bg-emerald-600 animate-pulse" : "bg-rose-500"
              }`}
            />
            <span className="text-[10px] text-slate-700 font-sans font-medium">
              {isBackendHealthy ? "Portal Operational" : "Service Offline"}
            </span>
          </div>
        </div>
      </div>

      {/* 3. Main Brand & Ministry of Earth Sciences Banner with Blended Light Indian Flag Background */}
      <div className="relative px-6 py-3.5 bg-white flex items-center justify-between gap-4 overflow-hidden border-b border-slate-200">
        {/* Blended Light Indian National Flag Background (Tricolor Wash) */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: `
              linear-gradient(180deg, 
                rgba(255, 153, 51, 0.12) 0%, 
                rgba(255, 153, 51, 0.05) 28%, 
                rgba(255, 255, 255, 0.70) 45%, 
                rgba(255, 255, 255, 0.70) 55%, 
                rgba(19, 136, 8, 0.05) 72%, 
                rgba(19, 136, 8, 0.12) 100%
              ),
              linear-gradient(90deg, 
                rgba(255, 153, 51, 0.09) 0%, 
                rgba(255, 255, 255, 0) 35%, 
                rgba(255, 255, 255, 0) 65%, 
                rgba(19, 136, 8, 0.09) 100%
              )
            `,
          }}
        />

        {/* Soft Decorative Indian Flag Wave Geometry (Subtle & Light) */}
        <div className="absolute inset-0 pointer-events-none opacity-20 overflow-hidden">
          <svg
            className="w-full h-full"
            preserveAspectRatio="none"
            viewBox="0 0 1200 120"
            fill="none"
          >
            <path
              d="M 0,0 C 300,22 600,-12 900,16 C 1050,28 1150,14 1200,8 L 1200,0 Z"
              fill="#FF9933"
              fillOpacity="0.4"
            />
            <path
              d="M 0,120 C 300,98 600,132 900,104 C 1050,92 1150,106 1200,112 L 1200,120 Z"
              fill="#138808"
              fillOpacity="0.4"
            />
          </svg>
        </div>

        {/* Faint Ashoka Chakra Watermark in Background */}
        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none opacity-[0.035]">
          <svg viewBox="0 0 100 100" className="w-44 h-44 text-blue-900">
            <circle cx="50" cy="50" r="46" fill="none" stroke="currentColor" strokeWidth="2.5" />
            <circle cx="50" cy="50" r="10" fill="none" stroke="currentColor" strokeWidth="2" />
            <circle cx="50" cy="50" r="4" fill="currentColor" />
            {[...Array(24)].map((_, i) => (
              <line
                key={i}
                x1="50"
                y1="10"
                x2="50"
                y2="50"
                stroke="currentColor"
                strokeWidth="1.2"
                transform={`rotate(${i * 15} 50 50)`}
              />
            ))}
          </svg>
        </div>

        {/* LEFT: CYCLONE-X in Left Corner */}
        <div className="relative z-10 flex items-center gap-3.5">
          {/* Satellite Symbol in Left Corner */}
          <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-blue-50 border border-blue-200 p-2 shadow-xs backdrop-blur-2xs group hover:bg-blue-100 transition-colors shrink-0">
            <Satellite className="w-6 h-6 text-blue-600 animate-pulse" />
          </div>

          <div className="text-left">
            <div className="flex items-center gap-2">
              <h1 className="text-xl md:text-2xl font-black tracking-tight text-[#0a2540] font-sans">
                CYCLONE<span className="text-blue-600 font-black">-X</span>
              </h1>
              <span className="px-2 py-0.5 rounded text-[10px] font-sans font-bold bg-blue-100 border border-blue-300 text-blue-800">
                AI EARLY WARNING
              </span>
            </div>
            <p className="text-xs text-slate-800 font-sans font-semibold tracking-wide">
              National Tropical Cyclone Intelligence &amp; Decision-Support System
            </p>
            <p className="text-[11px] text-slate-600 font-sans">
              Real-Time INSAT-3D/3DR &amp; NASA Constellation Georeferencing
            </p>
          </div>
        </div>

        {/* RIGHT: NRT Satellite Surveillance Status */}
        <div className="relative z-10 hidden sm:flex items-center gap-3.5 pl-5 border-l-2 border-slate-200">
          <div className="text-right">
            <div className="flex items-center justify-end gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-600 animate-pulse" />
              <span className="text-xs font-bold text-slate-900 font-sans tracking-wide">
                NRT SATELLITE SURVEILLANCE
              </span>
            </div>
            <p className="text-[11px] text-slate-600 font-sans mt-0.5">
              Multi-Spectral TIR1 • GPM IMERG • GDACS Feeds
            </p>
            <p className="text-[10px] text-slate-500 font-mono">
              Continuous North Indian Ocean Oceanic Monitoring
            </p>
          </div>
        </div>
      </div>
    </header>
  );
};
