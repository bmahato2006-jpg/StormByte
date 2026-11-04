"use client";

import React, { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { Layers, ShieldAlert, AlertTriangle, Building2, Users } from "lucide-react";
import { Cyclone } from "@/types/cyclone";
import { CompositeRiskAssessment, RiskMapGeoJSON } from "@/types/impact";

interface RiskMapInnerProps {
  cyclone: Cyclone | null;
  assessment: CompositeRiskAssessment | null;
  geoJsonData: RiskMapGeoJSON | null;
}

export const RiskMapInner: React.FC<RiskMapInnerProps> = ({
  cyclone,
  assessment,
  geoJsonData,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const layersGroupRef = useRef<L.LayerGroup | null>(null);

  // Active layer toggles
  const [showWindHazard, setShowWindHazard] = useState(true);
  const [showCoastalSurge, setShowCoastalSurge] = useState(true);
  const [showPopulation, setShowPopulation] = useState(true);
  const [showInfrastructure, setShowInfrastructure] = useState(true);
  const [isLayerControlOpen, setIsLayerControlOpen] = useState(true);

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    const initialLat = cyclone ? cyclone.latitude : 20.0;
    const initialLon = cyclone ? cyclone.longitude : 87.0;

    const map = L.map(mapContainerRef.current, {
      center: [initialLat, initialLon],
      zoom: 6,
      minZoom: 3,
      maxZoom: 13,
      zoomControl: false,
    });

    // High-Resolution ESRI World Imagery (Satellite Earth Observation)
    L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      {
        attribution: "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
        maxZoom: 18,
      }
    ).addTo(map);

    // Administrative boundaries & placenames overlay for clear situational context
    L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
      {
        attribution: "",
        maxZoom: 18,
        opacity: 0.85,
      }
    ).addTo(map);


    L.control.zoom({ position: "bottomright" }).addTo(map);

    const layerGroup = L.layerGroup().addTo(map);
    layersGroupRef.current = layerGroup;
    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update Layers
  useEffect(() => {
    const map = mapInstanceRef.current;
    const layerGroup = layersGroupRef.current;
    if (!map || !layerGroup || !cyclone) return;

    layerGroup.clearLayers();

    // Check whether this target is an active cyclonic storm or calm ambient stream
    const hasActiveCyclone = Boolean(
      cyclone &&
        !cyclone.name.includes("Active Satellite Stream") &&
        !cyclone.name.includes("NORTH INDIAN OCEAN") &&
        !cyclone.name.includes("Real-Time Satellite Stream") &&
        !cyclone.intensity.toLowerCase().includes("quiet") &&
        !cyclone.intensity.toLowerCase().includes("no active cyclone") &&
        !cyclone.intensity.toLowerCase().includes("ambient") &&
        !cyclone.intensity.toLowerCase().includes("normal") &&
        (cyclone.intensity.toLowerCase().includes("cyclon") ||
          cyclone.intensity.toLowerCase().includes("storm") ||
          (cyclone.intensity.toLowerCase().includes("depression") && !cyclone.intensity.toLowerCase().includes("no active")) ||
          (cyclone.windSpeed && cyclone.windSpeed >= 34))
    );

    // 1. Center Pin: Red Pulsing Vortex Eye for active cyclones, Green Station Marker for calm ambient observation
    if (hasActiveCyclone) {
      const eyeIcon = L.divIcon({
        className: "cyclone-risk-eye",
        html: `
          <div class="relative flex items-center justify-center w-8 h-8">
            <span class="absolute w-8 h-8 rounded-full bg-red-500/30 animate-ping"></span>
            <span class="absolute w-5 h-5 rounded-full bg-red-600/60 animate-pulse"></span>
            <span class="relative w-3.5 h-3.5 rounded-full bg-red-500 border-2 border-white shadow-[0_0_10px_#ef4444]"></span>
          </div>
        `,
        iconSize: [32, 32],
        iconAnchor: [16, 16],
      });

      const eyeMarker = L.marker([cyclone.latitude, cyclone.longitude], { icon: eyeIcon });
      eyeMarker.bindPopup(
        `<div class="p-3 font-mono text-xs bg-white text-slate-800 rounded-lg border border-red-400 shadow-md min-w-[220px]">
          <div class="font-bold text-red-700 text-sm">${cyclone.name} (VORTEX CENTER)</div>
          <div class="text-slate-600 mt-1">${cyclone.latitude.toFixed(2)}°N, ${cyclone.longitude.toFixed(2)}°E</div>
          <div>Sustained Wind: <span class="text-blue-700 font-bold">${cyclone.windSpeed} kt</span></div>
          <div>Central MSLP: <span class="text-slate-900 font-bold">${cyclone.pressure} hPa</span></div>
          <div class="text-[9px] text-amber-800 font-bold mt-2 pt-1 border-t border-slate-200 uppercase">
            ACTIVE CYCLONIC LOCUS
          </div>
        </div>`
      );
      layerGroup.addLayer(eyeMarker);
    } else {
      const calmIcon = L.divIcon({
        className: "calm-risk-station",
        html: `
          <div class="relative flex items-center justify-center w-8 h-8">
            <span class="absolute w-7 h-7 rounded-full bg-emerald-400/30 animate-ping"></span>
            <span class="relative w-4 h-4 rounded-full bg-emerald-500 border-2 border-white shadow-[0_0_8px_#10b981]"></span>
          </div>
        `,
        iconSize: [32, 32],
        iconAnchor: [16, 16],
      });

      const calmMarker = L.marker([cyclone.latitude, cyclone.longitude], { icon: calmIcon });
      calmMarker.bindPopup(
        `<div class="p-3 font-mono text-xs bg-white text-slate-800 rounded-lg border border-emerald-400 shadow-md min-w-[220px]">
          <div class="font-bold text-emerald-700 text-sm">${cyclone.name}</div>
          <div class="text-slate-600 mt-1">${cyclone.latitude.toFixed(2)}°N, ${cyclone.longitude.toFixed(2)}°E</div>
          <div>Ambient Wind: <span class="text-emerald-700 font-bold">${cyclone.windSpeed} kt</span> (Trade Winds)</div>
          <div>Surface Pressure: <span class="text-slate-900 font-bold">${cyclone.pressure} hPa</span></div>
          <div class="text-[9px] text-emerald-800 font-bold mt-2 pt-1 border-t border-slate-200 uppercase">
            NRT MARINE SURVEILLANCE LOCUS — BASIN QUIET
          </div>
        </div>`
      );
      layerGroup.addLayer(calmMarker);
    }

    // 2. Render GeoJSON features from Impact Engine
    if (geoJsonData && geoJsonData.features) {
      geoJsonData.features.forEach((feat) => {
        const props = feat.properties || {};
        const layerType = props.layer;

        // Wind Hazard Layer
        if (layerType === "WIND_RISK" && showWindHazard && feat.geometry.type === "Polygon") {
          const isExtreme = props.risk_level === "EXTREME";
          const poly = L.geoJSON(feat as any, {
            style: {
              color: isExtreme ? "#dc2626" : "#0284c7",
              fillColor: isExtreme ? "#dc2626" : "#0284c7",
              fillOpacity: isExtreme ? 0.22 : 0.12,
              weight: isExtreme ? 2 : 1.5,
              dashArray: isExtreme ? undefined : "4, 4",
            },
          });
          poly.bindPopup(
            `<div class="p-2.5 font-mono text-xs bg-white text-slate-800 rounded-lg border border-blue-300 shadow-md min-w-[240px]">
              <div class="font-bold text-blue-700 uppercase">${props.hazard_type}</div>
              <div>Radius: <span class="text-slate-900 font-bold">${props.radius_km} km</span></div>
              <div>Severity Rating: <span class="${isExtreme ? "text-red-700" : "text-amber-700"} font-bold">${props.risk_level}</span></div>
              <div class="text-[10px] text-blue-800 mt-1">Status: ${props.data_status}</div>
              <div class="text-[9px] text-amber-800 font-bold uppercase mt-1 pt-1 border-t border-slate-200">
                ${props.disclaimer}
              </div>
            </div>`
          );
          layerGroup.addLayer(poly);
        }

        // Coastal Surge Layer
        if (layerType === "COASTAL_SURGE" && showCoastalSurge && feat.geometry.type === "Polygon") {
          const poly = L.geoJSON(feat as any, {
            style: {
              color: "#9333ea",
              fillColor: "#9333ea",
              fillOpacity: 0.22,
              weight: 2,
            },
          });
          poly.bindPopup(
            `<div class="p-2.5 font-mono text-xs bg-white text-slate-800 rounded-lg border border-purple-300 shadow-md min-w-[240px]">
              <div class="font-bold text-purple-700 uppercase">${props.hazard_type}</div>
              <div>Estimated Peak Surge: <span class="text-slate-900 font-bold">${props.surge_amplitude_m} m</span></div>
              <div>Coastal Risk: <span class="text-red-700 font-bold">${props.risk_level}</span></div>
              <div class="text-[10px] text-purple-800 mt-1">Status: ${props.data_status}</div>
              <div class="text-[9px] text-amber-800 font-bold uppercase mt-1 pt-1 border-t border-slate-200">
                ${props.disclaimer}
              </div>
            </div>`
          );
          layerGroup.addLayer(poly);
        }

        // Population Settlements Layer
        if (layerType === "POPULATION_EXPOSURE" && showPopulation && feat.geometry.type === "Point") {
          const [lon, lat] = feat.geometry.coordinates;
          const marker = L.circleMarker([lat, lon], {
            radius: 6,
            color: "#d97706",
            fillColor: "#f59e0b",
            fillOpacity: 0.9,
            weight: 2,
          });
          marker.bindPopup(
            `<div class="p-2.5 font-mono text-xs bg-white text-slate-800 rounded-lg border border-amber-300 shadow-md min-w-[220px]">
              <div class="font-bold text-amber-800">${props.settlement_name} (${props.state})</div>
              <div>Estimated Pop: <span class="text-slate-900 font-bold">${props.population?.toLocaleString()}</span></div>
              <div>Distance to Core: <span class="text-slate-600">${props.distance_km} km</span></div>
              <div>Hazard Severity: <span class="text-red-700 font-bold">${props.hazard_level}</span></div>
              <div class="text-[9px] text-amber-800 font-bold uppercase mt-1 pt-1 border-t border-slate-200">
                ${props.disclaimer}
              </div>
            </div>`
          );
          layerGroup.addLayer(marker);
        }

        // Critical Infrastructure Layer
        if (layerType === "INFRASTRUCTURE_EXPOSURE" && showInfrastructure && feat.geometry.type === "Point") {
          const [lon, lat] = feat.geometry.coordinates;
          const marker = L.circleMarker([lat, lon], {
            radius: 6,
            color: "#0284c7",
            fillColor: "#0284c7",
            fillOpacity: 0.9,
            weight: 2,
          });
          marker.bindPopup(
            `<div class="p-2.5 font-mono text-xs bg-white text-slate-800 rounded-lg border border-blue-300 shadow-md min-w-[230px]">
              <div class="font-bold text-blue-700 uppercase">${props.facility_name}</div>
              <div>Asset Category: <span class="text-slate-900 font-bold">${props.facility_type}</span></div>
              <div>Distance: <span class="text-slate-600">${props.distance_km} km</span></div>
              <div>Vulnerability: <span class="text-red-700 font-bold">${props.risk_level}</span></div>
              <div class="text-[9px] text-amber-800 font-bold uppercase mt-1 pt-1 border-t border-slate-200">
                ${props.disclaimer}
              </div>
            </div>`
          );
          layerGroup.addLayer(marker);
        }
      });
    }
  }, [
    cyclone,
    assessment,
    geoJsonData,
    showWindHazard,
    showCoastalSurge,
    showPopulation,
    showInfrastructure,
  ]);

  return (
    <div className="relative w-full h-[540px] rounded-xl overflow-hidden border border-slate-200 bg-white shadow-xs">
      <div ref={mapContainerRef} className="w-full h-full z-0" />

      {/* Floating Layer Controls Drawer */}
      <div className="absolute top-3 right-3 z-[1000] bg-white/95 backdrop-blur-md border border-slate-200 rounded-lg p-3 shadow-md text-xs font-mono max-w-[280px] space-y-2.5 select-none text-slate-800">
        <div className="flex items-center justify-between border-b border-slate-200 pb-2">
          <div className="flex items-center gap-1.5 text-slate-900 font-bold">
            <Layers className="w-3.5 h-3.5 text-blue-700" />
            <span>DISASTER RISK LAYERS</span>
          </div>
          <button
            onClick={() => setIsLayerControlOpen(!isLayerControlOpen)}
            className="text-slate-500 hover:text-slate-800 text-[10px] px-1 font-semibold"
          >
            {isLayerControlOpen ? "MINIMIZE" : "EXPAND"}
          </button>
        </div>

        {isLayerControlOpen && (
          <div className="space-y-2.5 max-h-[360px] overflow-y-auto pr-1">
            <label className="flex items-center justify-between cursor-pointer text-slate-700 hover:text-slate-900">
              <span className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={showWindHazard}
                  onChange={(e) => setShowWindHazard(e.target.checked)}
                  className="rounded border-slate-300 text-blue-600 focus:ring-0"
                />
                <span className="text-[11px] font-medium">Wind Swaths (R34/R64)</span>
              </span>
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200 font-bold">
                MODEL-DERIVED
              </span>
            </label>

            <label className="flex items-center justify-between cursor-pointer text-slate-700 hover:text-slate-900">
              <span className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={showCoastalSurge}
                  onChange={(e) => setShowCoastalSurge(e.target.checked)}
                  className="rounded border-slate-300 text-purple-600 focus:ring-0"
                />
                <span className="text-[11px] font-medium">Coastal Storm Surge</span>
              </span>
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-purple-50 text-purple-700 border border-purple-200 font-bold">
                PHYSICAL SETUP
              </span>
            </label>

            <label className="flex items-center justify-between cursor-pointer text-slate-700 hover:text-slate-900">
              <span className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={showPopulation}
                  onChange={(e) => setShowPopulation(e.target.checked)}
                  className="rounded border-slate-300 text-amber-600 focus:ring-0"
                />
                <span className="text-[11px] font-medium">Population Exposure</span>
              </span>
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-50 text-amber-800 border border-amber-200 font-bold">
                CENSUS ATLAS
              </span>
            </label>

            <label className="flex items-center justify-between cursor-pointer text-slate-700 hover:text-slate-900">
              <span className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={showInfrastructure}
                  onChange={(e) => setShowInfrastructure(e.target.checked)}
                  className="rounded border-slate-300 text-emerald-600 focus:ring-0"
                />
                <span className="text-[11px] font-medium">Critical Infrastructure</span>
              </span>
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200 font-bold">
                INFRA REGISTER
              </span>
            </label>
          </div>
        )}
      </div>
    </div>
  );
};

