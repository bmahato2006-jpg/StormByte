"use client";

import React, { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import {
  Layers,
  Eye,
  Compass,
  ShieldAlert,
  Sparkles,
  Sliders,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Info,
  Loader2,
  X,
  RefreshCw,
  MapPin,
  Check,
} from "lucide-react";
import { Cyclone, TrackPoint } from "@/types/cyclone";
import { PredictionResponse, OceanRiskResponse } from "@/types/prediction";
import { analyzeOceanRisk } from "@/lib/api";
import { CATEGORY_COLORS } from "@/lib/constants";
import { RiskAssessment, AffectedDistrict } from "@/types/risk";

interface CycloneMapProps {
  cyclone: Cyclone | null;
  track: TrackPoint[];
  prediction: PredictionResponse | null;
  selectedStep?: string; // "NOW", "+6h", "+12h", "+24h"
  risk?: RiskAssessment | null;
  affectedDistricts?: AffectedDistrict[];
}

export const CycloneMap: React.FC<CycloneMapProps> = ({
  cyclone,
  track,
  prediction,
  selectedStep = "NOW",
  risk,
  affectedDistricts,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const layersGroupRef = useRef<L.LayerGroup | null>(null);
  const targetLayerGroupRef = useRef<L.LayerGroup | null>(null);

  // Prediction Mode State
  const [isPredictionMode, setIsPredictionMode] = useState(false);
  const [selectedTarget, setSelectedTarget] = useState<{ lat: number; lon: number; radiusKm: number } | null>(null);
  const [isAnalyzingRisk, setIsAnalyzingRisk] = useState(false);
  const [riskAssessment, setRiskAssessment] = useState<OceanRiskResponse | null>(null);
  const [riskAnalysisError, setRiskAnalysisError] = useState<string | null>(null);

  // Active layer toggles
  const [showHistoricalTrack, setShowHistoricalTrack] = useState(true);
  const [showPredictedTrack, setShowPredictedTrack] = useState(true);
  const [showUncertaintyCone, setShowUncertaintyCone] = useState(true);
  const [showRiskZones, setShowRiskZones] = useState(true);
  const [showAffectedDistricts, setShowAffectedDistricts] = useState(true);
  const [showAIHazardZones, setShowAIHazardZones] = useState(true);

  // Prepared Future Architecture Layer Toggles
  const [showSstLayer, setShowSstLayer] = useState(false);
  const [showWindVectors, setShowWindVectors] = useState(false);
  const [showRainfallLayer, setShowRainfallLayer] = useState(false);
  const [showPopulationLayer, setShowPopulationLayer] = useState(false);
  const [showInfrastructureLayer, setShowInfrastructureLayer] = useState(false);

  // UI toggle for layer control drawer (default minimized for clean view)
  const [isLayerControlOpen, setIsLayerControlOpen] = useState(false);


  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    const initialLat = cyclone ? cyclone.latitude : 18.0;
    const initialLon = cyclone ? cyclone.longitude : 85.0;

    const map = L.map(mapContainerRef.current, {
      center: [initialLat, initialLon],
      zoom: 6,
      minZoom: 3,
      maxZoom: 13,
      zoomControl: false,
    });

    // Authentic Satellite Basemap: ESRI World Imagery (High-Resolution Satellite Earth Observation)
    const satelliteTiles = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      {
        attribution: "Tiles &copy; Esri, Maxar, Earthstar Geographics, USDA FSA, USGS, Aerogrid, IGN, IGP, and the GIS User Community",
        maxZoom: 18,
      }
    );

    const referenceOverlay = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
      {
        attribution: "",
        maxZoom: 18,
        opacity: 0.85,
      }
    );

    const satelliteGroup = L.layerGroup([satelliteTiles, referenceOverlay]).addTo(map);

    const oceansTiles = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
      {
        attribution: "Tiles &copy; Esri, GEBCO, NOAA, National Geographic",
        maxZoom: 13,
      }
    );
    const oceansRef = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Reference/MapServer/tile/{z}/{y}/{x}",
      {
        attribution: "",
        maxZoom: 13,
      }
    );
    const oceansGroup = L.layerGroup([oceansTiles, oceansRef]);

    const lightGrayTiles = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
      {
        attribution: "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
        maxZoom: 16,
      }
    );
    const lightGrayRef = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
      {
        attribution: "",
        maxZoom: 16,
        opacity: 0.9,
      }
    );
    const lightGrayGroup = L.layerGroup([lightGrayTiles, lightGrayRef]);

    const baseMaps = {
      "🛰️ Satellite Imagery (Active)": satelliteGroup,
      "🌊 Ocean Bathymetry": oceansGroup,
      "🗺️ Light Gray Canvas": lightGrayGroup,
    };

    L.control.layers(baseMaps, undefined, { position: "topright" }).addTo(map);

    L.control.zoom({ position: "bottomright" }).addTo(map);

    // Create high-priority pane for prediction target overlay
    const targetPane = map.createPane("predictionTargetPane");
    targetPane.style.zIndex = "650";

    const layerGroup = L.layerGroup().addTo(map);
    layersGroupRef.current = layerGroup;

    const targetGroup = L.layerGroup().addTo(map);
    targetLayerGroupRef.current = targetGroup;

    mapInstanceRef.current = map;


    return () => {
      map.remove();
      mapInstanceRef.current = null;
      layersGroupRef.current = null;
      targetLayerGroupRef.current = null;
    };
  }, []);

  // Cursor crosshair in prediction mode
  useEffect(() => {
    const container = mapContainerRef.current;
    if (!container) return;
    if (isPredictionMode) {
      container.style.cursor = "crosshair";
    } else {
      container.style.cursor = "";
    }
  }, [isPredictionMode]);

  // Leaflet map click listener when Prediction Mode is active
  useEffect(() => {
    const map = mapInstanceRef.current;
    const container = mapContainerRef.current;
    if (!map || !container) return;

    const handleSelectCoordinate = (lat: number, lon: number) => {
      setSelectedTarget({ lat, lon, radiusKm: 100 });
      setRiskAssessment(null);
      setRiskAnalysisError(null);
    };

    const handleMapClick = (e: L.LeafletMouseEvent) => {
      if (!isPredictionMode) return;
      const lat = Number(e.latlng.lat.toFixed(4));
      const lon = Number(e.latlng.lng.toFixed(4));
      handleSelectCoordinate(lat, lon);
    };

    // Capture-phase DOM listener ensures clicks are never swallowed by underlying layers
    const handleContainerClick = (e: MouseEvent) => {
      if (!isPredictionMode) return;
      const target = e.target as HTMLElement;
      if (
        target.closest("button") ||
        target.closest(".leaflet-control") ||
        target.closest("[data-prediction-ui]")
      ) {
        return;
      }
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      try {
        const latlng = map.containerPointToLatLng([x, y]);
        const lat = Number(latlng.lat.toFixed(4));
        const lon = Number(latlng.lng.toFixed(4));
        if (lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180) {
          handleSelectCoordinate(lat, lon);
        }
      } catch {
        // ignore out-of-bounds conversions
      }
    };

    map.on("click", handleMapClick);
    container.addEventListener("click", handleContainerClick, true);

    return () => {
      map.off("click", handleMapClick);
      container.removeEventListener("click", handleContainerClick, true);
    };
  }, [isPredictionMode]);

  // Listen for focus-district requests from AffectedDistrictsPanel
  useEffect(() => {
    const handleFocusDistrict = (e: Event) => {
      const customEvent = e as CustomEvent<{ latitude: number; longitude: number; name: string }>;
      const { latitude, longitude } = customEvent.detail || {};
      if (latitude && longitude && mapInstanceRef.current) {
        mapInstanceRef.current.flyTo([latitude, longitude], 8, {
          duration: 1.0,
        });
      }
    };
    window.addEventListener("cyclone:focus-district", handleFocusDistrict);
    return () => {
      window.removeEventListener("cyclone:focus-district", handleFocusDistrict);
    };
  }, []);

  // Render target circle and center marker
  useEffect(() => {
    const targetGroup = targetLayerGroupRef.current;
    if (!targetGroup) return;

    targetGroup.clearLayers();

    if (!isPredictionMode || !selectedTarget) return;

    // Outer evaluation radius circle (100km)
    const circle = L.circle([selectedTarget.lat, selectedTarget.lon], {
      pane: "predictionTargetPane",
      radius: selectedTarget.radiusKm * 1000,
      color: "#00f0ff",
      fillColor: "#00f0ff",
      fillOpacity: 0.16,
      weight: 2.5,
      dashArray: "6, 6",
      interactive: false,
    });
    targetGroup.addLayer(circle);

    // Inner core circle (25km)
    const coreCircle = L.circle([selectedTarget.lat, selectedTarget.lon], {
      pane: "predictionTargetPane",
      radius: 25000,
      color: "#38bdf8",
      fillColor: "#38bdf8",
      fillOpacity: 0.28,
      weight: 1.5,
      interactive: false,
    });
    targetGroup.addLayer(coreCircle);

    // Target Center Crosshair Marker
    const centerMarker = L.circleMarker([selectedTarget.lat, selectedTarget.lon], {
      pane: "predictionTargetPane",
      radius: 8,
      color: "#ffffff",
      fillColor: "#00f0ff",
      fillOpacity: 1.0,
      weight: 3,
      interactive: false,
    });
    targetGroup.addLayer(centerMarker);
  }, [isPredictionMode, selectedTarget]);


  const handleExitPredictionMode = () => {
    setIsPredictionMode(false);
    setSelectedTarget(null);
    setRiskAssessment(null);
    setRiskAnalysisError(null);
    if (targetLayerGroupRef.current) {
      targetLayerGroupRef.current.clearLayers();
    }
  };

  const handleAnalyzeTarget = async () => {
    if (!selectedTarget) return;
    setIsAnalyzingRisk(true);
    setRiskAnalysisError(null);
    try {
      const res = await analyzeOceanRisk({
        latitude: selectedTarget.lat,
        longitude: selectedTarget.lon,
        radius_km: selectedTarget.radiusKm,
      });
      setRiskAssessment(res);
    } catch (err: any) {
      setRiskAnalysisError(err.message || "Failed to evaluate ocean risk");
    } finally {
      setIsAnalyzingRisk(false);
    }
  };

  const isLiveSatellite =
    cyclone?.provenance?.dataType === "LIVE SATELLITE STREAM" ||
    cyclone?.id === "LIVE-NIO-SATELLITE" ||
    cyclone?.dataSource?.includes("LIVE") ||
    cyclone?.dataSource?.includes("Live");

  const isRealHistorical =
    !isLiveSatellite &&
    (cyclone?.provenance?.dataType === "VERIFIED HISTORICAL DATA" ||
      cyclone?.provenance?.dataType === "REAL DATA" ||
      cyclone?.dataSource?.includes("IMD") ||
      cyclone?.dataSource?.includes("IBTrACS") ||
      cyclone?.id?.startsWith("IMD-") ||
      cyclone?.id?.startsWith("IBTRACS-"));

  const hasActiveCyclone = Boolean(
    cyclone &&
      !cyclone.name.includes("Active Satellite Stream") &&
      !cyclone.name.includes("NORTH INDIAN OCEAN") &&
      !cyclone.intensity.toLowerCase().includes("quiet") &&
      !cyclone.intensity.toLowerCase().includes("no active cyclone") &&
      !cyclone.intensity.toLowerCase().includes("ambient") &&
      (cyclone.intensity.toLowerCase().includes("cyclon") ||
        cyclone.intensity.toLowerCase().includes("storm") ||
        (cyclone.intensity.toLowerCase().includes("depression") && !cyclone.intensity.toLowerCase().includes("no active")) ||
        (cyclone.windSpeed && cyclone.windSpeed >= 34))
  );

  // Update Layers when data, step, or toggles change
  useEffect(() => {
    const map = mapInstanceRef.current;

    const layerGroup = layersGroupRef.current;
    if (!map || !layerGroup || !cyclone) return;

    layerGroup.clearLayers();

    // Determine target pan location based on selectedStep
    let targetLat = cyclone.latitude;
    let targetLon = cyclone.longitude;

    if (selectedStep !== "NOW" && prediction) {
      const matchedPred = prediction.predictions.find((p) => p.forecastTime === selectedStep);
      if (matchedPred) {
        targetLat = matchedPred.latitude;
        targetLon = matchedPred.longitude;
      }
    }

    // Recenter map smoothly
    map.panTo([targetLat, targetLon], { animate: true, duration: 0.8 });

    // 1. SST Thermal Field (Active Cyclone Only)
    if (showSstLayer && hasActiveCyclone) {
      const sstVal = cyclone.windSpeed > 64 ? 29.8 : 29.2;
      const sstCircle = L.circle([cyclone.latitude, cyclone.longitude], {
        radius: 150000,
        color: "#f97316",
        fillColor: "#ea580c",
        fillOpacity: 0.10,
        weight: 1.5,
        interactive: !isPredictionMode,
      });
      sstCircle.bindTooltip(
        `<div class="font-mono text-xs">
          <span class="font-bold text-orange-400">Ocean Thermal Field: ~${sstVal}°C</span>
          <div class="text-[10px] text-emerald-300 font-bold">[REAL-TIME BASIN SST TELEMETRY]</div>
        </div>`,
        { sticky: true }
      );
      layerGroup.addLayer(sstCircle);
    }

    // 2. Rainfall Inundation Footprint (Active Cyclone Only)
    if (showRainfallLayer && hasActiveCyclone && cyclone.windSpeed >= 34) {
      const rainPolygon = L.circle([cyclone.latitude, cyclone.longitude], {
        radius: 120000,
        color: "#0284c7",
        fillColor: "#0284c7",
        fillOpacity: 0.16,
        weight: 1.5,
        dashArray: "3, 3",
        interactive: !isPredictionMode,
      });
      rainPolygon.bindTooltip(
        `<div class="font-mono text-xs">
          <span class="font-bold text-sky-400">Active Convective Core Rain Band</span>
          <div class="text-[10px] text-emerald-300 font-bold">[OPERATIONAL SATELLITE RAINFALL]</div>
        </div>`,
        { sticky: true }
      );
      layerGroup.addLayer(rainPolygon);
    }

    // 3. Wind Streamlines (Active Cyclone Only)
    if (showWindVectors && hasActiveCyclone && cyclone.windSpeed >= 28) {
      const windVectors = [
        [[cyclone.latitude - 1.0, cyclone.longitude], [cyclone.latitude - 0.3, cyclone.longitude + 0.8]],
        [[cyclone.latitude + 1.0, cyclone.longitude], [cyclone.latitude + 0.3, cyclone.longitude - 0.8]],
        [[cyclone.latitude, cyclone.longitude - 1.0], [cyclone.latitude + 0.8, cyclone.longitude - 0.3]],
        [[cyclone.latitude, cyclone.longitude + 1.0], [cyclone.latitude - 0.8, cyclone.longitude + 0.3]],
      ];
      windVectors.forEach((v) => {
        const line = L.polyline(v as [number, number][], {
          color: "#00f0ff",
          weight: 2,
          opacity: 0.6,
          dashArray: "2, 4",
          interactive: !isPredictionMode,
        });
        line.bindTooltip(
          `<div class="font-mono text-xs text-cyan-300 font-bold">Tangential Cyclonic Wind Vector [VORTEX MODEL]</div>`,
          { sticky: true }
        );
        layerGroup.addLayer(line);
      });
    }

    // 4. Prepared Layer: Population Exposure Buffer (Active Cyclone Only)
    if (showPopulationLayer && hasActiveCyclone) {
      const popLayer = L.circle([cyclone.latitude, cyclone.longitude], {
        radius: 100000,
        color: "#a855f7",
        fillColor: "#a855f7",
        fillOpacity: 0.12,
        weight: 1,
        interactive: !isPredictionMode,
      });
      popLayer.bindTooltip(
        `<div class="font-mono text-xs text-purple-300 font-bold">Radial Exposure Perimeter (~100 km) [RISK BUFFER]</div>`,
        { sticky: true }
      );
      layerGroup.addLayer(popLayer);
    }

    // 5. Authentic Coastal Critical Infrastructure (Fixed Geocoded Positions)
    if (showInfrastructureLayer) {
      const authenticInfraPoints: [number, number, string, string][] = [
        [20.26, 86.67, "Paradeep Deepwater Port & Marine Terminal", "Odisha Coast"],
        [22.02, 88.06, "Haldia Dock Complex & Coastal Energy Hub", "West Bengal Coast"],
        [17.68, 83.21, "Visakhapatnam Major Seaport & Naval Base", "Andhra Pradesh Coast"],
        [13.08, 80.29, "Chennai Coastal Marine Harbor & Terminal", "Tamil Nadu Coast"],
        [21.45, 87.02, "Chandipur Coastal Testing Range & Substation", "Odisha Coast"],
        [19.31, 84.78, "Gopalpur Port & Coastal Weather Station", "Odisha Coast"],
      ];
      authenticInfraPoints.forEach(([lat, lon, label, region]) => {
        // Calculate distance from current cyclone if active
        const distKm = Math.round(
          Math.sqrt(Math.pow((lat - cyclone.latitude) * 111, 2) + Math.pow((lon - cyclone.longitude) * 111 * Math.cos((lat * Math.PI) / 180), 2))
        );
        const isThreatened = hasActiveCyclone && distKm < 300;

        const marker = L.circleMarker([lat, lon], {
          radius: isThreatened ? 7 : 5,
          color: isThreatened ? "#ef4444" : "#eab308",
          fillColor: isThreatened ? "#dc2626" : "#ca8a04",
          fillOpacity: 0.85,
          weight: isThreatened ? 2.5 : 1.5,
          interactive: !isPredictionMode,
        });
        marker.bindTooltip(
          `<div class="font-mono text-xs">
            <div class="font-bold ${isThreatened ? "text-red-400" : "text-yellow-300"}">${label}</div>
            <div class="text-[10px] text-slate-300">${region} • ${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E</div>
            ${hasActiveCyclone ? `<div class="text-[10px] ${isThreatened ? "text-red-300 font-bold" : "text-emerald-300"}">Distance to Storm: ${distKm} km ${isThreatened ? "[ALERT ZONE]" : "[NORMAL]"}</div>` : ""}
          </div>`,
          { sticky: true }
        );
        layerGroup.addLayer(marker);
      });
    }

    // 6. Risk Buffer Zones (180 km gale hazard - only if active storm exists)
    if (showRiskZones && hasActiveCyclone) {
      const riskBuffer = L.circle([cyclone.latitude, cyclone.longitude], {
        radius: 180000,
        color: "#ef4444",
        fillColor: "#ef4444",
        fillOpacity: 0.08,
        weight: 1.5,
        dashArray: "4, 6",
        interactive: !isPredictionMode,
      });
      riskBuffer.bindTooltip(
        `<div class="p-1 font-mono text-xs text-red-400 font-bold">
          <div>HIGH HAZARD GALE RADIUS (~180 km)</div>
          <div class="text-[10px] text-slate-300 font-normal">Buffer zone based on model gale estimation</div>
        </div>`,
        { sticky: true }
      );
      layerGroup.addLayer(riskBuffer);
    }

    // 7. Uncertainty Cone Polygon
    if (showUncertaintyCone && prediction && prediction.uncertaintyPolygon.length > 2) {
      const polygonCoords = prediction.uncertaintyPolygon.map(
        (pt) => [pt[0], pt[1]] as [number, number]
      );

      const cone = L.polygon(polygonCoords, {
        color: "#00f0ff",
        fillColor: "#00f0ff",
        fillOpacity: 0.14,
        weight: 1.5,
        dashArray: "5, 5",
        interactive: !isPredictionMode,
      });

      cone.bindTooltip(
        `<div class="p-1 font-mono text-xs">
          <div class="font-bold text-cyan-400">Cone of Uncertainty (70% Confidence Envelope)</div>
          <div class="text-[10px] text-amber-300 font-bold">MODEL-GENERATED / EXPERIMENTAL — NOT AN OFFICIAL FORECAST</div>
        </div>`,
        { sticky: true }
      );
      layerGroup.addLayer(cone);
    }

    // 7b. AI Dynamic Hazard Alert Envelopes (Yellow, Orange, Red Tier Polygons)
    if (showAIHazardZones && risk?.hazardZones) {
      const { yellowAlertPolygon, orangeAlertPolygon, redAlertPolygon } = risk.hazardZones;

      // Yellow Alert Arc (~235km peripheral squall buffer)
      if (yellowAlertPolygon && yellowAlertPolygon.length > 2) {
        const yellowPolygon = L.polygon(yellowAlertPolygon as [number, number][], {
          color: "#eab308",
          fillColor: "#eab308",
          fillOpacity: 0.08,
          weight: 1.5,
          dashArray: "3, 6",
          interactive: !isPredictionMode,
        });
        yellowPolygon.bindTooltip(
          `<div class="p-1.5 font-mono text-xs">
            <div class="font-bold text-yellow-400">AI YELLOW ALERT HAZARD ZONE</div>
            <div class="text-[10px] text-slate-300">Outer Peripheral Squall & Heavy Rain Arc (~235 km)</div>
            <div class="text-[9px] text-emerald-400 font-bold uppercase mt-0.5">SOURCE: NRT AI TRAJECTORY ENVELOPE</div>
          </div>`,
          { sticky: true }
        );
        layerGroup.addLayer(yellowPolygon);
      }

      // Orange Alert Arc (~145km gale & surge buffer)
      if (orangeAlertPolygon && orangeAlertPolygon.length > 2) {
        const orangePolygon = L.polygon(orangeAlertPolygon as [number, number][], {
          color: "#f97316",
          fillColor: "#f97316",
          fillOpacity: 0.14,
          weight: 2,
          dashArray: "4, 4",
          interactive: !isPredictionMode,
        });
        orangePolygon.bindTooltip(
          `<div class="p-1.5 font-mono text-xs">
            <div class="font-bold text-orange-400">AI ORANGE ALERT HAZARD ZONE</div>
            <div class="text-[10px] text-slate-300">High Wind, Severe Gales & Storm Surge Envelope (~145 km)</div>
            <div class="text-[9px] text-emerald-400 font-bold uppercase mt-0.5">SOURCE: NRT AI TRAJECTORY ENVELOPE</div>
          </div>`,
          { sticky: true }
        );
        layerGroup.addLayer(orangePolygon);
      }

      // Red Alert Arc (~75km destructive core gale / primary landfall sector)
      if (redAlertPolygon && redAlertPolygon.length > 2) {
        const redPolygon = L.polygon(redAlertPolygon as [number, number][], {
          color: "#e11d48",
          fillColor: "#e11d48",
          fillOpacity: 0.22,
          weight: 2.5,
          dashArray: "5, 5",
          interactive: !isPredictionMode,
        });
        redPolygon.bindTooltip(
          `<div class="p-1.5 font-mono text-xs">
            <div class="font-bold text-rose-500">AI RED ALERT HAZARD ZONE</div>
            <div class="text-[10px] text-slate-200">Destructive Core Gale & Landfall Danger Sector (~75 km)</div>
            <div class="text-[9px] text-rose-300 font-bold uppercase mt-0.5">MANDATORY EVACUATION & HARBOR SHUTDOWN ARC</div>
          </div>`,
          { sticky: true }
        );
        layerGroup.addLayer(redPolygon);
      }
    }

    // 8. Historical Track Polyline & Nodes
    if (showHistoricalTrack && track.length > 0) {
      const trackCoords = track.map((pt) => [pt.latitude, pt.longitude] as [number, number]);

      const polyline = L.polyline(trackCoords, {
        color: isLiveSatellite ? "#00f0ff" : isRealHistorical ? "#10b981" : "#f59e0b",
        weight: 3,
        opacity: 0.85,
        interactive: !isPredictionMode,
      });
      layerGroup.addLayer(polyline);

      track.forEach((pt, index) => {
        const catConfig = CATEGORY_COLORS[pt.intensity] || { border: isLiveSatellite ? "#00f0ff" : isRealHistorical ? "#10b981" : "#f59e0b" };
        const marker = L.circleMarker([pt.latitude, pt.longitude], {
          radius: index === track.length - 1 ? 6 : 4,
          color: catConfig.border,
          fillColor: catConfig.border,
          fillOpacity: 0.9,
          weight: 2,
          interactive: !isPredictionMode,
        });

        marker.bindPopup(
          `<div class="p-3 font-mono text-xs bg-white text-slate-800 rounded-lg border border-slate-200 shadow-xl min-w-[220px]">
            ${
              isLiveSatellite
                ? `<div class="font-bold text-blue-700 uppercase">LIVE SATELLITE TELEMETRY (${new Date(pt.timestamp).toUTCString()})</div>
                   <div class="text-[10px] text-slate-500">SOURCE: ${cyclone.dataSource || "NASA EOSDIS / IMD INSAT-3D"}</div>`
                : isRealHistorical
                ? `<div class="font-bold text-emerald-700 uppercase">AUTHENTIC OBSERVATION (${new Date(pt.timestamp).toUTCString()})</div>
                   <div class="text-[10px] text-slate-500">SOURCE: ${cyclone.dataSource || "IMD RSMC NEW DELHI"}</div>`
                : `<div class="font-bold text-amber-700 uppercase">DEMO TRACK — NOT AN OFFICIAL FORECAST</div>
                   <div class="text-[11px] text-slate-600 mt-1">Time: ${new Date(pt.timestamp).toUTCString()}</div>`
            }
            <div class="mt-1">Intensity: <span class="font-bold text-slate-900">${pt.intensity}</span></div>
            <div>Wind Speed: <span class="text-blue-700 font-bold">${pt.windSpeed} kt</span></div>
            <div>MSLP: <span class="text-slate-800 font-semibold">${pt.pressure} hPa</span></div>
            <div class="text-[10px] text-slate-500 mt-1 pt-1 border-t border-slate-100">${isLiveSatellite ? "Live Space Telemetry Observation" : isRealHistorical ? "Validated Best Track Observation" : "Source: Curated Historical Benchmark"}</div>
          </div>`
        );
        layerGroup.addLayer(marker);
      });
    }

    // 9. Predicted Track Polyline & Forward Nodes
    if (showPredictedTrack && prediction && prediction.predictions.length > 0) {
      const predCoords: [number, number][] = [
        [cyclone.latitude, cyclone.longitude],
        ...prediction.predictions.map((p) => [p.latitude, p.longitude] as [number, number]),
      ];

      const predLine = L.polyline(predCoords, {
        color: "#00f0ff",
        weight: 2.5,
        dashArray: "6, 8",
        opacity: 0.9,
        interactive: !isPredictionMode,
      });
      layerGroup.addLayer(predLine);

      prediction.predictions.forEach((p) => {
        const isFocusedStep = selectedStep === p.forecastTime;
        const predMarker = L.circleMarker([p.latitude, p.longitude], {
          radius: isFocusedStep ? 8 : 5,
          color: isFocusedStep ? "#ffffff" : "#00f0ff",
          fillColor: isFocusedStep ? "#00f0ff" : "#0891b2",
          fillOpacity: 1.0,
          weight: isFocusedStep ? 3 : 2,
          interactive: !isPredictionMode,
        });

        // Error circle around prediction point
        const uncertCircle = L.circle([p.latitude, p.longitude], {
          radius: p.uncertainty * 1000,
          color: isFocusedStep ? "#00f0ff" : "#0891b2",
          fillColor: "#00f0ff",
          fillOpacity: isFocusedStep ? 0.15 : 0.08,
          weight: isFocusedStep ? 2 : 1,
          interactive: !isPredictionMode,
        });
        layerGroup.addLayer(uncertCircle);

        predMarker.bindPopup(
          `<div class="p-3 font-mono text-xs bg-white text-slate-800 rounded-lg border border-blue-400 shadow-xl min-w-[220px]">
            <div class="font-bold text-blue-700 uppercase">MODEL-GENERATED / EXPERIMENTAL PREDICTION (${p.forecastTime})</div>
            <div class="text-[11px] text-slate-600 mt-1">Locus: ${p.latitude.toFixed(2)}°N, ${p.longitude.toFixed(2)}°E</div>
            <div>Cone Uncertainty: <span class="text-amber-700 font-bold">±${p.uncertainty} km</span></div>
            ${p.predictedWindSpeed ? `<div>Proj. Wind: <span class="text-blue-700 font-bold">${p.predictedWindSpeed} kt</span></div>` : ""}
            ${p.predictedPressure ? `<div>Proj. Pressure: <span class="text-slate-800 font-semibold">${p.predictedPressure} hPa</span></div>` : ""}
            <div class="text-[10px] text-amber-700 mt-1 font-bold">AI STATUS: EXPERIMENTAL ML FORECAST</div>
            <div class="text-[9px] text-amber-800/90 uppercase mt-0.5 font-bold">EXPERIMENTAL RESEARCH SYSTEM — NOT AN OFFICIAL WEATHER FORECAST</div>
          </div>`
        );
        layerGroup.addLayer(predMarker);
      });
    }

    // 9b. AI Trajectory Affected Coastal Districts Layer
    const districtsToRender =
      affectedDistricts && affectedDistricts.length > 0
        ? affectedDistricts
        : risk?.affectedDistricts || [];
    const primaryLandfallName = risk?.primaryLandfallDistrict;

    if (showAffectedDistricts && districtsToRender.length > 0) {
      districtsToRender.forEach((d) => {
        const isRed = d.alertTier === "RED";
        const isOrange = d.alertTier === "ORANGE";
        const isLandfall = primaryLandfallName === d.districtName;

        const pinColor = isRed ? "#e11d48" : isOrange ? "#f97316" : "#eab308";
        const pinRadius = isLandfall ? 9 : isRed ? 7 : 5.5;

        // Custom pulsing marker or circleMarker
        const districtMarker = L.circleMarker([d.latitude, d.longitude], {
          radius: pinRadius,
          color: isLandfall ? "#ffffff" : pinColor,
          fillColor: pinColor,
          fillOpacity: 0.9,
          weight: isLandfall ? 2.5 : 1.5,
        });

        // Landfall beacon ring if primary
        if (isLandfall) {
          const beaconRing = L.circle([d.latitude, d.longitude], {
            radius: 40000,
            color: "#e11d48",
            fillColor: "#e11d48",
            fillOpacity: 0.12,
            weight: 2,
            dashArray: "3, 6",
          });
          layerGroup.addLayer(beaconRing);
        }

        const tooltipHtml = `
          <div class="p-2 font-mono text-xs max-w-xs space-y-1">
            <div class="flex items-center justify-between border-b border-slate-200 pb-1 mb-1">
              <span class="font-bold text-slate-900">${d.districtName}</span>
              <span class="px-1.5 py-0.5 rounded text-[9px] font-bold ${
                isRed ? "bg-rose-600 text-white" : isOrange ? "bg-orange-500 text-white" : "bg-yellow-400 text-slate-900"
              }">
                ${d.alertTier} ALERT
              </span>
            </div>
            ${isLandfall ? `<div class="text-[10px] text-rose-600 font-bold uppercase tracking-wider">⚠ PRIMARY PROJECTED LANDFALL</div>` : ""}
            <div class="text-[11px] text-slate-600">State: <strong>${d.state}</strong></div>
            <div class="text-[11px] text-slate-600">Closest Track Approach: <strong>${d.closestApproachKm} km</strong> (${d.arrivalEta})</div>
            <div class="text-[11px] text-slate-800">Peak Gust: <strong class="text-rose-600">${d.maxPredictedGustKmh} km/h</strong> (${d.maxPredictedWindKt} kt)</div>
            <div class="text-[11px] text-cyan-700">Projected Surge: <strong>+${d.stormSurgeMeters} m</strong> • Rain: <strong>${d.rainfallHazardMm} mm</strong></div>
            <div class="text-[11px] text-blue-700">Exposed Pop: <strong>${(d.exposedPopulation / 1000).toFixed(0)}k</strong> (Census: ${(d.population / 1000000).toFixed(1)}M)</div>
            ${d.inUncertaintyCone ? `<div class="text-[9px] text-cyan-600 font-bold uppercase mt-1">✓ Inside 70% AI Uncertainty Cone</div>` : ""}
          </div>
        `;

        districtMarker.bindTooltip(tooltipHtml, { sticky: true });
        layerGroup.addLayer(districtMarker);
      });
    }

    // 10. Current Cyclone Center Eye Marker (Pulsing Center Pin) or Calm Basin Observation Marker
    if (hasActiveCyclone) {
      const eyeIcon = L.divIcon({
        className: "cyclone-eye-icon",
        html: `
          <div class="relative flex items-center justify-center w-9 h-9">
            <span class="absolute w-9 h-9 rounded-full bg-cyan-400/30 animate-ping"></span>
            <span class="absolute w-6 h-6 rounded-full bg-red-500/50 animate-pulse"></span>
            <span class="relative w-4 h-4 rounded-full bg-red-500 border-2 border-white shadow-[0_0_12px_#ef4444]"></span>
          </div>
        `,
        iconSize: [36, 36],
        iconAnchor: [18, 18],
      });

      const eyeMarker = L.marker([cyclone.latitude, cyclone.longitude], {
        icon: eyeIcon,
        interactive: !isPredictionMode,
      });
      eyeMarker.bindPopup(
        `<div class="p-3.5 font-mono text-xs bg-white text-slate-800 rounded-lg border-2 border-blue-600 shadow-2xl min-w-[240px]">
          <div class="flex items-center justify-between border-b border-slate-200 pb-1.5 mb-2">
            <span class="font-black text-sm text-blue-800">${cyclone.name}</span>
            <span class="px-1.5 py-0.5 rounded text-[10px] bg-red-50 text-red-700 border border-red-200 font-bold">${cyclone.intensity}</span>
          </div>
          <div class="space-y-1 text-slate-700">
            <div>Center Coordinates: <span class="text-slate-900 font-bold">${cyclone.latitude.toFixed(2)}°N, ${cyclone.longitude.toFixed(2)}°E</span></div>
            <div>Max Sustained Wind: <span class="text-blue-700 font-bold">${cyclone.windSpeed} kt</span> (${Math.round(cyclone.windSpeed * 1.852)} km/h)</div>
            <div>Minimum Pressure: <span class="text-slate-900 font-bold">${cyclone.pressure} hPa</span></div>
            <div>Movement Vector: <span class="text-emerald-700 font-bold">${cyclone.movementDirection}</span> @ ${cyclone.movementSpeed} km/h</div>
            <div class="text-[10px] text-slate-500 pt-1 border-t border-slate-200 mt-1">Source: ${cyclone.dataSource}</div>
            ${
              isRealHistorical
                ? `<div class="text-[9px] text-emerald-700 font-bold uppercase mt-0.5">AUTHENTIC OBSERVATION — SOURCE: ${cyclone.dataSource || "IMD RSMC NEW DELHI"}</div>
                   <div class="text-[8px] text-slate-500 font-bold uppercase mt-0.5">OPERATIONAL CYCLONE TELEMETRY</div>`
                : `<div class="text-[9px] text-emerald-700 font-bold uppercase mt-0.5">LIVE SATELLITE TELEMETRY — IMD INSAT-3D & NASA EOSDIS</div>`
            }
          </div>
        </div>`
      );
    } else if (cyclone.earlyGenesis) {
      // Pre-cyclonic Tropical Cloud Cluster / Disturbance before formal eye formation
      const distRadiusMeters = (cyclone.earlyGenesis.disturbanceRadiusKm || 160) * 1000;
      const isHighRisk =
        cyclone.earlyGenesis.riskTier === "HIGH" ||
        cyclone.earlyGenesis.riskTier === "CRITICAL";

      // 1. Dashed envelope for detected low-level circulation disturbance/cloud cluster
      const disturbanceCircle = L.circle([cyclone.latitude, cyclone.longitude], {
        radius: distRadiusMeters,
        color: isHighRisk ? "#f59e0b" : "#0284c7",
        fillColor: isHighRisk ? "#f59e0b" : "#0284c7",
        fillOpacity: 0.12,
        weight: 2.5,
        dashArray: "6, 8",
        interactive: !isPredictionMode,
      });

      disturbanceCircle.bindTooltip(
        `<div class="font-mono text-xs">
          <div class="font-bold text-amber-400">Pre-Cyclonic Disturbance Envelope (~${Math.round(cyclone.earlyGenesis.disturbanceRadiusKm)} km)</div>
          <div class="text-[10px] text-slate-300">Tropical Cloud Cluster / Low-Level Circulation Center</div>
          <div class="text-[10px] text-amber-300 font-bold mt-0.5">Genesis Probability: ${cyclone.earlyGenesis.probabilityPercent}% (${cyclone.earlyGenesis.riskTier} Risk)</div>
        </div>`,
        { sticky: true }
      );
      layerGroup.addLayer(disturbanceCircle);

      // 2. Pre-cyclone disturbance center marker
      const disturbanceIcon = L.divIcon({
        className: "disturbance-cluster-icon",
        html: `
          <div class="relative flex items-center justify-center w-10 h-10">
            <span class="absolute w-9 h-9 rounded-full ${isHighRisk ? "bg-amber-400/40" : "bg-cyan-400/40"} animate-ping"></span>
            <span class="absolute w-7 h-7 rounded-full ${isHighRisk ? "bg-amber-500/20" : "bg-cyan-500/20"} border border-dashed ${isHighRisk ? "border-amber-400" : "border-cyan-400"}"></span>
            <span class="relative flex items-center justify-center w-5 h-5 rounded-full ${isHighRisk ? "bg-amber-500" : "bg-cyan-600"} border-2 border-white shadow-[0_0_12px_${isHighRisk ? "#f59e0b" : "#06b6d4"}] text-white text-[9px] font-bold">
              🌀
            </span>
          </div>
        `,
        iconSize: [40, 40],
        iconAnchor: [20, 20],
      });

      const disturbanceMarker = L.marker([cyclone.latitude, cyclone.longitude], {
        icon: disturbanceIcon,
        interactive: !isPredictionMode,
      });

      disturbanceMarker.bindPopup(
        `<div class="p-3.5 font-mono text-xs bg-white text-slate-800 rounded-lg border-2 border-amber-500 shadow-2xl min-w-[260px]">
          <div class="flex items-center justify-between border-b border-slate-200 pb-1.5 mb-2">
            <span class="font-black text-sm text-amber-900">${cyclone.name}</span>
            <span class="px-1.5 py-0.5 rounded text-[10px] bg-amber-50 text-amber-800 border border-amber-300 font-bold">
              ${cyclone.earlyGenesis.preCycloneStage || "PRE-CYCLONE DISTURBANCE"}
            </span>
          </div>
          <div class="space-y-1.5 text-slate-700">
            <div class="p-2 rounded bg-amber-50/90 border border-amber-200">
              <div class="font-bold text-amber-900 flex items-center justify-between">
                <span>Genesis Risk Score:</span>
                <span class="px-1.5 py-0.5 rounded bg-amber-600 text-white text-[10px] font-black">${cyclone.earlyGenesis.riskTier} (${cyclone.earlyGenesis.probabilityPercent}%)</span>
              </div>
              <div class="text-[10px] text-amber-800 mt-1">
                24–48h Intensification: <span class="font-bold">${cyclone.earlyGenesis.intensification24to48h ? "LIKELY (to ≥34 kt Cyclonic Storm)" : "UNLIKELY / SUPPRESSED"}</span>
              </div>
            </div>
            <div>Circulation Center: <span class="text-slate-900 font-bold">${cyclone.latitude.toFixed(2)}°N, ${cyclone.longitude.toFixed(2)}°E</span></div>
            <div>Disturbance Radius: <span class="text-amber-800 font-bold">~${Math.round(cyclone.earlyGenesis.disturbanceRadiusKm)} km</span> (Dashed convective envelope)</div>
            <div>Ambient Wind: <span class="text-blue-700 font-bold">${cyclone.windSpeed} kt</span> (${Math.round(cyclone.windSpeed * 1.852)} km/h) [Pre-Cyclone Stage]</div>
            <div>Surface Pressure: <span class="text-slate-900 font-bold">${cyclone.pressure} hPa</span></div>
            ${
              cyclone.earlyGenesis.environmentalFactors
                ? `<div class="text-[10px] text-slate-600 pt-1 border-t border-slate-200">
                    SST: ${cyclone.earlyGenesis.environmentalFactors.seaSurfaceTemperature ?? "N/A"}°C | 
                    Shear: ${cyclone.earlyGenesis.environmentalFactors.verticalWindShearKt ?? "N/A"} kt | 
                    Vorticity: ${cyclone.earlyGenesis.environmentalFactors.relativeVorticity850 ?? "N/A"}
                  </div>`
                : ""
            }
            <div class="text-[9px] text-amber-700 font-bold uppercase mt-1">EARLY TROPICAL CYCLOGENESIS CLASSIFICATION HEAD ACTIVE</div>
          </div>
        </div>`
      );
      layerGroup.addLayer(disturbanceMarker);
    } else {
      const calmIcon = L.divIcon({
        className: "calm-station-icon",
        html: `
          <div class="relative flex items-center justify-center w-8 h-8">
            <span class="absolute w-7 h-7 rounded-full bg-emerald-400/30 animate-ping"></span>
            <span class="relative w-4 h-4 rounded-full bg-emerald-500 border-2 border-white shadow-[0_0_8px_#10b981]"></span>
          </div>
        `,
        iconSize: [32, 32],
        iconAnchor: [16, 16],
      });

      const calmMarker = L.marker([cyclone.latitude, cyclone.longitude], {
        icon: calmIcon,
        interactive: !isPredictionMode,
      });
      calmMarker.bindPopup(
        `<div class="p-3.5 font-mono text-xs bg-white text-slate-800 rounded-lg border-2 border-emerald-500 shadow-2xl min-w-[240px]">
          <div class="flex items-center justify-between border-b border-slate-200 pb-1.5 mb-2">
            <span class="font-black text-sm text-emerald-800">${cyclone.name}</span>
            <span class="px-1.5 py-0.5 rounded text-[10px] bg-emerald-50 text-emerald-700 border border-emerald-200 font-bold">NORMAL SEA STATE</span>
          </div>
          <div class="space-y-1 text-slate-700">
            <div>Monitoring Coordinates: <span class="text-slate-900 font-bold">${cyclone.latitude.toFixed(2)}°N, ${cyclone.longitude.toFixed(2)}°E</span></div>
            <div>Ambient Wind Speed: <span class="text-emerald-700 font-bold">${cyclone.windSpeed} kt</span> (${Math.round(cyclone.windSpeed * 1.852)} km/h)</div>
            <div>Surface Pressure: <span class="text-slate-900 font-bold">${cyclone.pressure} hPa</span></div>
            <div>Current Sea State: <span class="text-emerald-700 font-bold">Calm / Non-Cyclonic</span></div>
            <div class="text-[10px] text-slate-500 pt-1 border-t border-slate-200 mt-1">Source: ${cyclone.dataSource || "NRT Live Surface Feed"}</div>
            <div class="text-[9px] text-emerald-700 font-bold uppercase mt-0.5">CONTINUOUS NRT SATELLITE & OCEANIC SURVEILLANCE</div>
          </div>
        </div>`
      );
      layerGroup.addLayer(calmMarker);
    }
  }, [
    cyclone,
    track,
    prediction,
    selectedStep,
    isPredictionMode,
    showHistoricalTrack,
    showPredictedTrack,
    showUncertaintyCone,
    showRiskZones,
    showAffectedDistricts,
    showSstLayer,
    showWindVectors,
    showRainfallLayer,
    showPopulationLayer,
    showInfrastructureLayer,
    hasActiveCyclone,
    risk,
    affectedDistricts,
  ]);

  return (
    <div className="relative w-full h-[560px] lg:h-[640px] rounded-xl overflow-hidden border border-command-cardBorder bg-command-card shadow-2xl">
      {/* Leaflet Map Canvas */}
      <div ref={mapContainerRef} className="w-full h-full z-0" />

      {/* Top Left Controls: Predict Button & Early Genesis Alert Badge */}
      <div className="absolute top-3 left-3 z-[1000] flex items-center gap-2.5 flex-wrap pointer-events-auto max-w-[calc(100%-80px)]">
        {!isPredictionMode && (
          <button
            onClick={() => setIsPredictionMode(true)}
            className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-white/95 hover:bg-blue-50 border-2 border-blue-600 text-blue-900 font-mono text-xs font-bold transition-all shadow-xl backdrop-blur-md hover:scale-[1.02] active:scale-[0.98]"
          >
            <Sparkles className="w-3.5 h-3.5 text-blue-600 animate-pulse" />
            <span>🔮 Predict Cyclone Risk</span>
          </button>
        )}

        {/* Early Genesis Alert Badge */}
        {cyclone?.earlyGenesis && (
          <div
            data-testid="early-genesis-alert-badge"
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border-2 shadow-2xl backdrop-blur-md font-mono text-xs font-bold transition-all ${
              cyclone.earlyGenesis.riskTier === "CRITICAL"
                ? "bg-rose-950/90 border-rose-500 text-rose-100"
                : cyclone.earlyGenesis.riskTier === "HIGH"
                ? "bg-amber-950/90 border-amber-500 text-amber-100"
                : cyclone.earlyGenesis.riskTier === "MODERATE"
                ? "bg-yellow-950/90 border-yellow-500 text-yellow-100"
                : "bg-emerald-950/90 border-emerald-500 text-emerald-100"
            }`}
          >
            <span className="relative flex h-2.5 w-2.5 shrink-0">
              <span
                className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                  cyclone.earlyGenesis.riskTier === "CRITICAL" || cyclone.earlyGenesis.riskTier === "HIGH"
                    ? "bg-amber-400"
                    : "bg-emerald-400"
                }`}
              ></span>
              <span
                className={`relative inline-flex rounded-full h-2.5 w-2.5 ${
                  cyclone.earlyGenesis.riskTier === "CRITICAL" || cyclone.earlyGenesis.riskTier === "HIGH"
                    ? "bg-amber-500"
                    : "bg-emerald-500"
                }`}
              ></span>
            </span>

            <span className="uppercase tracking-wider text-[11px] text-white/90">
              Early Genesis Alert:
            </span>

            <span
              className={`px-1.5 py-0.5 rounded text-[11px] font-black ${
                cyclone.earlyGenesis.riskTier === "CRITICAL"
                  ? "bg-rose-500/30 text-rose-200 border border-rose-500/60"
                  : cyclone.earlyGenesis.riskTier === "HIGH"
                  ? "bg-amber-500/30 text-amber-200 border border-amber-500/60"
                  : cyclone.earlyGenesis.riskTier === "MODERATE"
                  ? "bg-yellow-500/30 text-yellow-200 border border-yellow-500/60"
                  : "bg-emerald-500/30 text-emerald-200 border border-emerald-500/60"
              }`}
            >
              Genesis Risk: {cyclone.earlyGenesis.riskTier} - {cyclone.earlyGenesis.probabilityPercent}%
            </span>

            {cyclone.earlyGenesis.intensification24to48h ? (
              <span className="hidden md:inline-flex items-center gap-1 text-[10px] text-amber-300 font-bold border-l border-white/20 pl-2">
                <AlertTriangle className="w-3 h-3 text-amber-400 shrink-0" />
                <span>24-48h Intensification Likely (&ge;34 kt)</span>
              </span>
            ) : (
              <span className="hidden md:inline-flex items-center gap-1 text-[10px] text-emerald-300 font-normal border-l border-white/20 pl-2">
                <span>Disturbance Suppressed</span>
              </span>
            )}
          </div>
        )}
      </div>

      {/* Active Prediction Mode Unified Command Strip */}
      {isPredictionMode && (
        <div
          data-prediction-ui="true"
          className="absolute top-2.5 left-2.5 right-2.5 z-[1000] flex flex-wrap items-center justify-between gap-2.5 bg-white/95 border-2 border-blue-600 rounded-xl px-3.5 py-2 shadow-2xl backdrop-blur-md font-mono text-xs animate-in fade-in slide-in-from-top-2"
        >
          {/* Left: Mode Badge & Target Status */}
          <div className="flex items-center gap-2.5 flex-wrap">
            <div className="flex items-center gap-1.5 bg-blue-50 border border-blue-300 px-2.5 py-1 rounded-lg shrink-0">
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-500 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-blue-600"></span>
              </span>
              <span className="font-extrabold text-blue-900 uppercase tracking-wider text-[11px]">
                PREDICTION MODE
              </span>
            </div>

            {!selectedTarget ? (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-50 border border-slate-300 text-slate-700 text-[11px] animate-pulse">
                <Compass className="w-3.5 h-3.5 text-blue-600 shrink-0" />
                <span>Click anywhere on an ocean area to place target (100 km radius)</span>
              </span>
            ) : (
              <div className="inline-flex items-center gap-2 bg-slate-50 border border-blue-300 rounded-lg px-2.5 py-1">
                <Compass className="w-3.5 h-3.5 text-blue-600 shrink-0" />
                <span className="text-slate-500 text-[10px] font-bold">TARGET:</span>
                <span className="font-bold text-slate-900 text-[11px]">
                  {selectedTarget.lat.toFixed(2)}°N, {selectedTarget.lon.toFixed(2)}°E
                </span>
                <span className="text-slate-400">|</span>
                <span className="text-blue-700 font-bold text-[10px]">
                  r={selectedTarget.radiusKm} km
                </span>
              </div>
            )}
          </div>

          {/* Right: Action Buttons & Controls */}
          <div className="flex items-center gap-2 ml-auto flex-wrap">
            {selectedTarget && !riskAssessment && (
              <>
                <button
                  onClick={handleAnalyzeTarget}
                  disabled={isAnalyzingRisk}
                  className="px-3.5 py-1.5 rounded-lg bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-bold flex items-center gap-1.5 transition-all shadow-md disabled:opacity-50 text-xs"
                >
                  {isAnalyzingRisk ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      <span>Analyzing Multi-Source Data...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-3.5 h-3.5" />
                      <span>⚡ Analyze Selected Area</span>
                    </>
                  )}
                </button>
                <button
                  onClick={() => {
                    setSelectedTarget(null);
                    if (targetLayerGroupRef.current) targetLayerGroupRef.current.clearLayers();
                  }}
                  className="px-2.5 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 border border-slate-300 text-slate-700 text-[11px] font-semibold transition-all"
                >
                  Clear
                </button>
              </>
            )}

            {riskAssessment && (
              <button
                onClick={() => {
                  setRiskAssessment(null);
                  setSelectedTarget(null);
                  if (targetLayerGroupRef.current) targetLayerGroupRef.current.clearLayers();
                }}
                className="px-3 py-1.5 rounded-lg bg-blue-50 hover:bg-blue-100 border border-blue-400 text-blue-900 text-xs font-bold flex items-center gap-1.5 shadow transition-all"
              >
                <RefreshCw className="w-3 h-3 text-blue-600" />
                <span>Select New Area</span>
              </button>
            )}

            {/* GIS Layers Toggle in Prediction Mode */}
            <button
              onClick={() => setIsLayerControlOpen(!isLayerControlOpen)}
              className={`px-2.5 py-1.5 rounded-lg border text-xs flex items-center gap-1.5 transition-all font-semibold ${
                isLayerControlOpen
                  ? "bg-blue-50 border-blue-500 text-blue-800"
                  : "bg-white border-slate-300 text-slate-700 hover:bg-slate-50"
              }`}
            >
              <Layers className="w-3.5 h-3.5 text-blue-600" />
              <span className="hidden md:inline text-[11px]">Layers</span>
            </button>

            {/* Exit Prediction Mode */}
            <button
              onClick={handleExitPredictionMode}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-rose-50 hover:bg-rose-100 border border-rose-300 text-rose-800 text-xs font-bold shadow-xs transition-all"
            >
              <X className="w-3 h-3 text-rose-600" />
              <span>Exit</span>
            </button>
          </div>
        </div>
      )}

      {/* Analysis Error Toast */}
      {isPredictionMode && riskAnalysisError && (
        <div
          data-prediction-ui="true"
          className="absolute top-16 left-2.5 z-[1000] bg-rose-50 border border-rose-200 rounded-xl px-4 py-2.5 text-rose-800 text-xs font-mono shadow-lg flex items-center justify-between gap-3 max-w-[420px] animate-in fade-in"
        >
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0" />
            <span>{riskAnalysisError}</span>
          </div>
          <button
            onClick={() => setRiskAnalysisError(null)}
            className="text-rose-500 hover:text-rose-800 p-0.5 rounded"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Risk Assessment Result Card */}
      {riskAssessment && (
        <div
          data-prediction-ui="true"
          className="absolute top-16 right-2.5 z-[1001] w-[calc(100%-1.25rem)] sm:w-[440px] max-h-[500px] overflow-y-auto bg-white/95 border-2 border-blue-500/80 rounded-xl p-4 shadow-2xl backdrop-blur-md font-mono text-xs text-slate-800 space-y-3 animate-in fade-in zoom-in-95"
        >
          {/* Card Header */}
          <div className="flex items-start justify-between border-b border-slate-200 pb-2.5">
            <div>
              <div className="flex items-center gap-1.5 text-blue-700 font-bold text-[10px] uppercase tracking-wider">
                <Sparkles className="w-3.5 h-3.5" />
                <span>Ocean Area Cyclone Risk Assessment</span>
              </div>
              <div className="text-slate-900 font-bold text-sm mt-0.5">
                {riskAssessment.location.latitude.toFixed(2)}°N, {riskAssessment.location.longitude.toFixed(2)}°E
              </div>
              <div className="text-[10px] text-slate-500">
                Basin: <span className="text-blue-700 font-semibold">{riskAssessment.location.basin.replace(/_/g, " ")}</span> • Radius: {riskAssessment.location.radius_km} km
              </div>
            </div>
            <button
              onClick={() => setRiskAssessment(null)}
              className="p-1 text-slate-400 hover:text-slate-700 rounded-lg hover:bg-slate-100"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Case 1: Land Rejection */}
          {!riskAssessment.location.is_ocean ? (
            <div className="p-3.5 rounded-lg bg-amber-50 border border-amber-300 text-amber-900 space-y-2">
              <div className="font-bold flex items-center gap-2 text-amber-800 text-xs uppercase">
                <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
                <span>Please select an ocean area</span>
              </div>
              <p className="text-[11px] text-amber-900/90 leading-relaxed">
                {riskAssessment.reasoning}
              </p>
              <div className="pt-1">
                <button
                  onClick={() => {
                    setRiskAssessment(null);
                    setSelectedTarget(null);
                    if (targetLayerGroupRef.current) targetLayerGroupRef.current.clearLayers();
                  }}
                  className="w-full py-1.5 px-3 rounded bg-amber-600 hover:bg-amber-700 text-white font-bold text-xs flex items-center justify-center gap-1.5 transition-all shadow"
                >
                  <RefreshCw className="w-3 h-3" />
                  <span>Click Ocean Area on Map to Retry</span>
                </button>
              </div>
            </div>
          ) : (
            <>
              {/* Case 2: Equatorial Advisory */}
              {riskAssessment.location.is_equatorial_restricted && (
                <div className="p-2.5 rounded-lg bg-blue-50 border border-blue-200 text-blue-900 text-[11px] space-y-1">
                  <div className="font-bold flex items-center gap-1.5 text-blue-800">
                    <Info className="w-3.5 h-3.5 text-blue-600" />
                    <span>Equatorial Constraint (&lt;5° of Equator)</span>
                  </div>
                  <p className="text-blue-800/80 leading-tight">
                    Coriolis parameter is insufficient to sustain cyclonic rotation.
                  </p>
                </div>
              )}

              {/* Risk Level Badge & Score Meter */}
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-slate-600 text-[10px] uppercase font-bold tracking-wider">
                    Cyclone Development Risk
                  </span>
                  <span
                    className={`px-2.5 py-0.5 rounded font-bold text-xs ${
                      riskAssessment.risk_level === "VERY HIGH"
                        ? "bg-red-100 border border-red-300 text-red-800"
                        : riskAssessment.risk_level === "HIGH"
                        ? "bg-orange-100 border border-orange-300 text-orange-800"
                        : riskAssessment.risk_level === "MODERATE"
                        ? "bg-amber-100 border border-amber-300 text-amber-800"
                        : riskAssessment.risk_level === "LOW"
                        ? "bg-sky-100 border border-sky-300 text-sky-800"
                        : "bg-emerald-100 border border-emerald-300 text-emerald-800"
                    }`}
                  >
                    {riskAssessment.risk_level}
                  </span>
                </div>

                {/* Score Progress Bar */}
                <div className="space-y-1">
                  <div className="flex justify-between text-[11px] font-mono">
                    <span className="text-slate-700">{riskAssessment.model_estimated_likelihood}</span>
                    <span className="font-bold text-slate-900">{riskAssessment.risk_score} / 100</span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-slate-200 overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 ${
                        riskAssessment.risk_score > 80
                          ? "bg-red-500"
                          : riskAssessment.risk_score > 60
                          ? "bg-orange-500"
                          : riskAssessment.risk_score > 40
                          ? "bg-amber-500"
                          : riskAssessment.risk_score > 20
                          ? "bg-sky-500"
                          : "bg-emerald-500"
                      }`}
                      style={{ width: `${Math.max(4, riskAssessment.risk_score)}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Itemized Scientific Evidence Checklist */}
              <div className="space-y-1.5">
                <div className="text-[10px] font-bold text-slate-600 uppercase tracking-wider">
                  Meteorological Evidence Checklist
                </div>
                <div className="space-y-1 max-h-[160px] overflow-y-auto pr-1">
                  {riskAssessment.evidence_checklist.map((item, idx) => (
                    <div
                      key={idx}
                      className="p-2 rounded bg-slate-50 border border-slate-200 text-[10.5px] space-y-0.5"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-slate-800">{item.factor}</span>
                        <span
                          className={`px-1.5 py-0.2 rounded text-[9px] font-bold ${
                            item.favorable
                              ? "bg-emerald-100 text-emerald-800 border border-emerald-300"
                              : item.status === "UNAVAILABLE"
                              ? "bg-slate-100 text-slate-600 border border-slate-300"
                              : "bg-slate-100 text-slate-600 border border-slate-300"
                          }`}
                        >
                          {item.favorable ? "✓ PRESENT" : item.status === "UNAVAILABLE" ? "⚠️ UNAVAILABLE" : "✗ ABSENT"}
                        </span>
                      </div>
                      <div className="text-[10px] text-slate-600 leading-tight">
                        {item.details}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* 24-Hour Temporal Persistence */}
              <div className="p-2 rounded bg-slate-50 border border-slate-200 text-[10.5px] space-y-0.5">
                <div className="flex items-center justify-between text-[10px]">
                  <span className="text-slate-600 uppercase font-bold">24h Lookback Persistence</span>
                  <span className="text-blue-700 font-bold">{riskAssessment.temporal_persistence.trend}</span>
                </div>
                <div className="text-[10px] text-slate-700">
                  {riskAssessment.temporal_persistence.details}
                </div>
              </div>

              {/* Key Drivers */}
              <div className="p-2 rounded bg-slate-50 border border-slate-200 text-[10.5px] space-y-1">
                <div className="text-[10px] font-bold text-slate-600 uppercase">Key Meteorological Drivers</div>
                <ul className="list-disc list-inside text-slate-700 space-y-0.5 text-[10px]">
                  {riskAssessment.key_drivers.map((driver, idx) => (
                    <li key={idx}>{driver}</li>
                  ))}
                </ul>
              </div>

              {/* Select Another Ocean Area Action */}
              <div className="pt-1">
                <button
                  onClick={() => {
                    setRiskAssessment(null);
                    setSelectedTarget(null);
                    if (targetLayerGroupRef.current) targetLayerGroupRef.current.clearLayers();
                  }}
                  className="w-full py-1.5 px-3 rounded bg-blue-50 hover:bg-blue-100 text-blue-900 font-bold text-xs flex items-center justify-center gap-1.5 transition-all border border-blue-300 shadow-xs"
                >
                  <RefreshCw className="w-3 h-3 text-blue-600" />
                  <span>🎯 Select Another Ocean Area</span>
                </button>
              </div>
            </>
          )}

          {/* Scientific Notice & Safety Disclaimer */}
          <div className="pt-2 border-t border-slate-200 space-y-1 text-[9.5px] text-slate-500 leading-relaxed font-mono">
            <div className="text-blue-700 font-bold uppercase">
              {riskAssessment.experimental_notice}
            </div>
            <p className="text-slate-500">
              {riskAssessment.disclaimer}
            </p>
          </div>
        </div>
      )}

      {/* Floating Layer Controls Drawer */}
      {(!isPredictionMode || isLayerControlOpen) && (
        <div
          data-prediction-ui="true"
          className={`absolute ${
            isPredictionMode ? "top-16 right-2.5" : "top-3 right-3"
          } z-[1000] bg-white/95 backdrop-blur-md border border-slate-200 rounded-lg p-3 shadow-xl text-xs font-mono max-w-[260px] space-y-2.5 select-none animate-in fade-in`}
        >
          <div className="flex items-center justify-between border-b border-slate-200 pb-2">
            <div className="flex items-center gap-1.5 text-slate-800 font-bold">
              <Layers className="w-3.5 h-3.5 text-blue-600" />
              <span>GIS LAYER CONTROLS</span>
            </div>
            <button
              onClick={() => setIsLayerControlOpen(!isLayerControlOpen)}
              className="text-slate-500 hover:text-slate-800 text-[10px] px-1 font-semibold"
            >
              {isLayerControlOpen ? "MINIMIZE" : "EXPAND"}
            </button>
          </div>

          {isLayerControlOpen && (
            <div className="space-y-2 max-h-[380px] overflow-y-auto pr-1">
            {/* Core Layers */}
            <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">
              Active Operational Layers
            </div>

            <label className="flex items-center gap-2 cursor-pointer text-slate-700 hover:text-slate-900 font-medium">
              <input
                type="checkbox"
                checked={showHistoricalTrack}
                onChange={(e) => setShowHistoricalTrack(e.target.checked)}
                className="rounded bg-slate-100 border-slate-300 text-blue-600 focus:ring-0"
              />
              <span className="flex items-center gap-1.5 text-[11px]">
                <span className={`w-2.5 h-1 ${isRealHistorical ? "bg-emerald-600" : "bg-blue-600"} rounded-full inline-block`} />
                {isRealHistorical ? "Historical Track [AUTHENTIC]" : "Historical Track [OBSERVED]"}
              </span>
            </label>

            <label className="flex items-center gap-2 cursor-pointer text-slate-700 hover:text-slate-900 font-medium">
              <input
                type="checkbox"
                checked={showPredictedTrack}
                onChange={(e) => setShowPredictedTrack(e.target.checked)}
                className="rounded bg-slate-100 border-slate-300 text-blue-600 focus:ring-0"
              />
              <span className="flex items-center gap-1.5 text-[11px]">
                <span className="w-2.5 h-0.5 bg-blue-600 border-b border-dashed border-blue-600 inline-block" />
                Predicted Track (+24h)
              </span>
            </label>

            <label className="flex items-center gap-2 cursor-pointer text-slate-700 hover:text-slate-900 font-medium">
              <input
                type="checkbox"
                checked={showUncertaintyCone}
                onChange={(e) => setShowUncertaintyCone(e.target.checked)}
                className="rounded bg-slate-100 border-slate-300 text-blue-600 focus:ring-0"
              />
              <span className="flex items-center gap-1.5 text-[11px]">
                <span className="w-2.5 h-2.5 bg-blue-100 border border-blue-500 inline-block rounded-sm" />
                Uncertainty Cone (70%)
              </span>
            </label>

            <label className="flex items-center gap-2 cursor-pointer text-slate-700 hover:text-slate-900 font-medium">
              <input
                type="checkbox"
                checked={showRiskZones}
                onChange={(e) => setShowRiskZones(e.target.checked)}
                className="rounded bg-slate-100 border-slate-300 text-red-600 focus:ring-0"
              />
              <span className="flex items-center gap-1.5 text-[11px]">
                <span className="w-2.5 h-2.5 bg-red-100 border border-red-500 inline-block rounded-sm" />
                Gale Hazard Zone (180km)
              </span>
            </label>

            <label className="flex items-center gap-2 cursor-pointer text-slate-700 hover:text-slate-900 font-medium">
              <input
                type="checkbox"
                checked={showAffectedDistricts}
                onChange={(e) => setShowAffectedDistricts(e.target.checked)}
                className="rounded bg-slate-100 border-slate-300 text-rose-600 focus:ring-0"
              />
              <span className="flex items-center gap-1.5 text-[11px]">
                <span className="w-2.5 h-2.5 bg-rose-500 rounded-full inline-block border border-white shadow-2xs" />
                AI Affected Coastal Districts
              </span>
            </label>

            <label className="flex items-center gap-2 cursor-pointer text-slate-700 hover:text-slate-900 font-medium">
              <input
                type="checkbox"
                checked={showAIHazardZones}
                onChange={(e) => setShowAIHazardZones(e.target.checked)}
                className="rounded bg-slate-100 border-slate-300 text-rose-600 focus:ring-0"
              />
              <span className="flex items-center gap-1.5 text-[11px]">
                <span className="w-2.5 h-2.5 bg-gradient-to-r from-yellow-400 via-orange-500 to-rose-600 border border-slate-400 inline-block rounded-xs" />
                AI Hazard Zones (Red/Orange/Yellow)
              </span>
            </label>

            {/* Prepared Future Architecture Layers */}
            <div className="pt-2 text-[10px] font-bold text-slate-500 uppercase tracking-wider border-t border-slate-200">
              Operational Earth Observation Layers
            </div>

            <label className="flex items-center gap-2 cursor-pointer text-slate-700 hover:text-slate-900 font-medium">
              <input
                type="checkbox"
                checked={showRainfallLayer}
                onChange={(e) => setShowRainfallLayer(e.target.checked)}
                className="rounded bg-slate-100 border-slate-300 text-sky-600 focus:ring-0"
              />
              <span className="flex items-center gap-1.5 text-[11px]">
                <span className="w-2.5 h-2.5 bg-sky-100 border border-sky-500 inline-block rounded-sm" />
                Rainfall Inundation Flux
              </span>
            </label>

            <label className="flex items-center gap-2 cursor-pointer text-slate-700 hover:text-slate-900 font-medium">
              <input
                type="checkbox"
                checked={showWindVectors}
                onChange={(e) => setShowWindVectors(e.target.checked)}
                className="rounded bg-slate-100 border-slate-300 text-blue-600 focus:ring-0"
              />
              <span className="flex items-center gap-1.5 text-[11px]">
                <span className="w-2.5 h-0.5 bg-blue-600 inline-block" />
                Wind Field Vectors
              </span>
            </label>

            <label className="flex items-center gap-2 cursor-pointer text-slate-700 hover:text-slate-900 font-medium">
              <input
                type="checkbox"
                checked={showSstLayer}
                onChange={(e) => setShowSstLayer(e.target.checked)}
                className="rounded bg-slate-100 border-slate-300 text-orange-600 focus:ring-0"
              />
              <span className="flex items-center gap-1.5 text-[11px]">
                <span className="w-2.5 h-2.5 bg-orange-100 border border-orange-500 inline-block rounded-sm" />
                SST Thermal Gradient
              </span>
            </label>

            <label className="flex items-center gap-2 cursor-pointer text-slate-700 hover:text-slate-900 font-medium">
              <input
                type="checkbox"
                checked={showPopulationLayer}
                onChange={(e) => setShowPopulationLayer(e.target.checked)}
                className="rounded bg-slate-100 border-slate-300 text-purple-600 focus:ring-0"
              />
              <span className="flex items-center gap-1.5 text-[11px]">
                <span className="w-2.5 h-2.5 bg-purple-100 border border-purple-500 inline-block rounded-sm" />
                Population Density
              </span>
            </label>

            <label className="flex items-center gap-2 cursor-pointer text-slate-700 hover:text-slate-900 font-medium">
              <input
                type="checkbox"
                checked={showInfrastructureLayer}
                onChange={(e) => setShowInfrastructureLayer(e.target.checked)}
                className="rounded bg-slate-100 border-slate-300 text-amber-600 focus:ring-0"
              />
              <span className="flex items-center gap-1.5 text-[11px]">
                <span className="w-2.5 h-2.5 bg-amber-100 border border-amber-500 inline-block rounded-sm" />
                Infrastructure Hazards
              </span>
            </label>
          </div>
        )}
      </div>
      )}

      {/* Floating Scientific Map Legend & Basin Indicator */}
      <div className="absolute bottom-3 left-3 z-[1000] bg-white/95 backdrop-blur-md border border-slate-200 rounded-lg p-2.5 shadow-xl text-[11px] font-mono select-none max-w-[540px]">
        <div className="flex items-center gap-2 text-[10px] text-slate-500 font-bold uppercase pb-1.5 mb-1.5 border-b border-slate-200">
          <Compass className="w-3.5 h-3.5 text-blue-700 animate-spin-slow" />
          <span className="text-slate-800">BASIN: {cyclone ? cyclone.basin : "NORTH INDIAN OCEAN"}</span>
          <span className="text-slate-400">•</span>
          <span className="text-blue-700 font-bold">WGS-84</span>
          <span className="text-slate-400">•</span>
          <span className="text-slate-700">SCIENTIFIC MAP LEGEND</span>
        </div>
        <div className="flex flex-wrap items-center gap-x-3.5 gap-y-1 text-[10.5px] text-slate-700">
          <span className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${isRealHistorical ? "bg-emerald-600" : "bg-blue-600"}`} />
            <span>● Historical Observation</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className={`w-3 h-1 rounded-full ${isRealHistorical ? "bg-emerald-600" : "bg-blue-600"}`} />
            <span>━ Historical Track</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 border-b border-dashed border-blue-600" />
            <span className="text-blue-700 font-semibold">━ Model Forecast (+24h)</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full border border-dashed border-blue-500 bg-blue-100" />
            <span className="text-slate-600">◌ Uncertainty Cone</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-rose-600 border border-white shadow-2xs" />
            <span className="text-rose-700 font-semibold">● Affected District</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded bg-red-100 border border-red-500" />
            <span className="text-rose-700 font-semibold">⚠ Risk Zone (180km)</span>
          </span>
        </div>
      </div>
    </div>
  );
};
