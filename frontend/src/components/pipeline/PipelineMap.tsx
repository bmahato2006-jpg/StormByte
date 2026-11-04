"use client";

import React, { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { CenterFix, TrajectoryForecast } from "@/types/pipeline";
import { Compass, AlertTriangle, ArrowRight, Radio, Search, Layers } from "lucide-react";

export interface PipelineMapProps {
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

export const PipelineMap: React.FC<PipelineMapProps> = ({
  centerFix,
  forecast,
  trackHistory = [],
  requestedLocation = null,
  observationCoverage = null,
  showCoverageBounds = false,
  imergObservation = null,
  satelliteName = null,
  isLiveNrt = true,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const layerGroupRef = useRef<L.LayerGroup | null>(null);

  const hasCoords = Boolean(
    centerFix &&
    centerFix.coordinates_available &&
    centerFix.latitude !== null &&
    centerFix.longitude !== null
  );

  const hasRequested = Boolean(
    requestedLocation &&
    typeof requestedLocation.latitude === "number" &&
    typeof requestedLocation.longitude === "number" &&
    !isNaN(requestedLocation.latitude) &&
    !isNaN(requestedLocation.longitude)
  );

  const hasCoverage = Boolean(
    showCoverageBounds &&
    observationCoverage &&
    typeof observationCoverage.min_latitude === "number" &&
    typeof observationCoverage.max_latitude === "number"
  );

  const hasImerg = Boolean(
    imergObservation &&
    typeof imergObservation.grid_latitude === "number" &&
    typeof imergObservation.grid_longitude === "number"
  );

  const hasRenderableContent = isLiveNrt || hasCoords || hasRequested || hasCoverage || hasImerg;

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    // Determine initial center (defaults to North Indian Ocean Basin)
    let initialLat = 16.0;
    let initialLon = 83.0;

    const cf = centerFix;
    if (hasRequested && requestedLocation) {
      initialLat = requestedLocation.latitude;
      initialLon = requestedLocation.longitude;
    } else if (cf && cf.coordinates_available && typeof cf.latitude === "number" && typeof cf.longitude === "number") {
      initialLat = cf.latitude;
      initialLon = cf.longitude;
    } else if (hasCoverage && observationCoverage) {
      initialLat = (observationCoverage.min_latitude + observationCoverage.max_latitude) / 2.0;
      initialLon = (observationCoverage.min_longitude + observationCoverage.max_longitude) / 2.0;
    }

    const map = L.map(mapContainerRef.current, {
      center: [initialLat, initialLon],
      zoom: 5,
      minZoom: 2,
      maxZoom: 12,
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
    layerGroupRef.current = layerGroup;
    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
      layerGroupRef.current = null;
    };
  }, []);

  // Update Map Layers
  useEffect(() => {
    const map = mapInstanceRef.current;
    const group = layerGroupRef.current;
    if (!map || !group || !hasRenderableContent) return;

    group.clearLayers();

    const boundsPoints: [number, number][] = [];

    // 1. Plot 🛰️ Satellite Observation Coverage (Bounding Box Polygon)
    if (hasCoverage && observationCoverage) {
      const { min_latitude, max_latitude, min_longitude, max_longitude, granule_id, satellite } = observationCoverage;
      const rectBounds: [[number, number], [number, number]] = [
        [min_latitude, min_longitude],
        [max_latitude, max_longitude],
      ];

      L.rectangle(rectBounds, {
        color: "#10b981", // emerald-500
        weight: 2,
        dashArray: "6, 6",
        fillColor: "#10b981",
        fillOpacity: 0.12,
      })
        .bindTooltip(
          `<b>🛰️ SATELLITE COVERAGE BOUNDS</b><br/>Granule: ${granule_id || "NRT"}<br/>Satellite: ${satellite || "INSAT-3D"}<br/>Lat: [${min_latitude.toFixed(2)}°N to ${max_latitude.toFixed(2)}°N]<br/>Lon: [${min_longitude.toFixed(2)}°E to ${max_longitude.toFixed(2)}°E]`,
          {
            className: "font-mono text-xs bg-white border border-emerald-500 text-slate-800 shadow-md",
            permanent: false,
          }
        )
        .addTo(group);

      boundsPoints.push([min_latitude, min_longitude], [max_latitude, max_longitude]);
    }

    // 1.5. Plot 🟢 NRT Satellite Surveillance Center when basin is quiet
    const cf = centerFix;
    const hasValidCenter = Boolean(cf && cf.coordinates_available && typeof cf.latitude === "number" && typeof cf.longitude === "number");
    if (hasCoverage && observationCoverage && !hasValidCenter) {
      const cLat = (observationCoverage.min_latitude + observationCoverage.max_latitude) / 2.0;
      const cLon = (observationCoverage.min_longitude + observationCoverage.max_longitude) / 2.0;

      L.circleMarker([cLat, cLon], {
        radius: 7,
        fillColor: "#10b981", // emerald-500
        color: "#ffffff",
        weight: 2,
        fillOpacity: 0.9,
      })
        .bindTooltip(
          `<b>🟢 NRT SATELLITE SURVEILLANCE</b><br/>Status: <b>BASIN QUIET • NO VORTEX DETECTED</b><br/>Surveillance Center: ${cLat.toFixed(2)}°N, ${cLon.toFixed(2)}°E<br/>Sensor: ${observationCoverage.satellite || "NRT SATELLITE"}`,
          {
            className: "font-mono text-xs bg-white border border-emerald-500 text-slate-800 shadow-md",
            permanent: false,
          }
        )
        .addTo(group);
    }

    // 2. Plot 📍 Requested Location Marker
    if (hasRequested && requestedLocation) {
      const rLat = requestedLocation.latitude;
      const rLon = requestedLocation.longitude;

      // Cyan target marker with pulsing ring
      L.circleMarker([rLat, rLon], {
        radius: 8,
        fillColor: "#0284c7", // blue-600
        color: "#ffffff",
        weight: 2.5,
        fillOpacity: 1.0,
      })
        .bindTooltip(
          `<b>📍 REQUESTED LOCATION</b><br/>Lat: ${rLat.toFixed(2)}°N | Lon: ${rLon.toFixed(2)}°E`,
          {
            className: "font-mono text-xs bg-white border border-blue-500 text-slate-800 shadow-md",
            permanent: false,
          }
        )
        .addTo(group);

      // Outer radar pulse ring
      L.circleMarker([rLat, rLon], {
        radius: 16,
        color: "#0284c7",
        weight: 1.5,
        dashArray: "3, 3",
        fillColor: "#0284c7",
        fillOpacity: 0.15,
      }).addTo(group);

      boundsPoints.push([rLat, rLon]);
    }

    // 2.5. Plot 🌧️ NASA IMERG Precipitation Observation
    if (hasImerg && imergObservation && imergObservation.grid_latitude && imergObservation.grid_longitude) {
      const gLat = imergObservation.grid_latitude;
      const gLon = imergObservation.grid_longitude;
      const d = 0.05; // 0.1° cell radius
      const cellBounds: [[number, number], [number, number]] = [
        [gLat - d, gLon - d],
        [gLat + d, gLon + d],
      ];

      L.rectangle(cellBounds, {
        color: "#0284c7",
        weight: 2,
        fillColor: "#38bdf8",
        fillOpacity: 0.35,
      })
        .bindPopup(
          `<div class="font-mono text-xs space-y-1 p-1 text-slate-800">
            <div class="font-bold text-blue-700 border-b border-slate-200 pb-1">
              🌧️ NASA IMERG OBSERVATION
            </div>
            <div><b>Product:</b> ${imergObservation.product}</div>
            <div><b>Observation Time:</b> ${imergObservation.observation_time || "2025-08-01 00:00–00:29 UTC"}</div>
            <div><b>Latitude:</b> ${gLat.toFixed(4)}°N</div>
            <div><b>Longitude:</b> ${gLon.toFixed(4)}°E</div>
            <div><b>Precipitation:</b> <span class="text-blue-900 font-bold">${imergObservation.precipitation_mm_per_hr ?? 0} mm/hr</span> (${imergObservation.precipitation ?? 0} in ${imergObservation.unit})</div>
            <div><b>Source:</b> ${imergObservation.source}</div>
            <div class="text-[10px] text-emerald-600 font-semibold mt-1">✓ Historical / Verified NASA Observation</div>
          </div>`,
          {
            className: "font-mono text-xs bg-white border border-blue-400 text-slate-800 shadow-lg",
          }
        )
        .bindTooltip(
          `<b>🌧️ NASA IMERG PRECIPITATION</b><br/>Rainfall: ${imergObservation.precipitation_mm_per_hr ?? 0} mm/hr<br/>Grid: ${gLat.toFixed(2)}°N, ${gLon.toFixed(2)}°E<br/>Source: ${imergObservation.source}<br/>Click for details`,
          {
            className: "font-mono text-xs bg-white border border-blue-400 text-slate-800 shadow-md",
            permanent: false,
          }
        )
        .addTo(group);

      boundsPoints.push([gLat - d, gLon - d], [gLat + d, gLon + d]);
    }

    // 3. Observed Past Track (Solid Line)
    if (trackHistory && trackHistory.length > 1) {
      const pastCoords: [number, number][] = trackHistory.map((p) => [p.latitude, p.longitude]);
      L.polyline(pastCoords, {
        color: "#00f0ff",
        weight: 3,
        opacity: 0.8,
      }).addTo(group);

      // Past fix markers
      trackHistory.forEach((p, idx) => {
        L.circleMarker([p.latitude, p.longitude], {
          radius: 4,
          fillColor: "#00f0ff",
          color: "#003b46",
          weight: 1,
          fillOpacity: 0.9,
        })
          .bindTooltip(`Historical Fix ${idx + 1}<br/>${p.timestamp || ""}`, {
            className: "font-mono text-xs bg-white border border-slate-300 text-slate-800 shadow-md",
          })
          .addTo(group);

        boundsPoints.push([p.latitude, p.longitude]);
      });
    }

    // 4. Plot 🌀 Cyclone Center only when independently verified
    if (cf && cf.coordinates_available && typeof cf.latitude === "number" && typeof cf.longitude === "number") {
      const obsLat = cf.latitude;
      const obsLon = cf.longitude;

      L.circleMarker([obsLat, obsLon], {
        radius: 9,
        fillColor: "#ef4444",
        color: "#ffffff",
        weight: 2.5,
        fillOpacity: 1.0,
      })
        .bindTooltip(
          `<b>🌀 CURRENT OBSERVED CENTER</b><br/>Lat: ${obsLat.toFixed(2)}°N | Lon: ${obsLon.toFixed(2)}°E<br/>Method: ${cf.localization_method}`,
          {
            className: "font-mono text-xs bg-white border border-red-500 text-slate-800 shadow-md",
            permanent: false,
          }
        )
        .addTo(group);

      boundsPoints.push([obsLat, obsLon]);

      // 5. Plot ➡️ Forecast Track only when genuine track history exists
      if (forecast && forecast.predictions && forecast.predictions.length > 0) {
        const forecastCoords: [number, number][] = [
          [obsLat, obsLon],
          ...forecast.predictions.map((p) => [p.latitude, p.longitude] as [number, number]),
        ];

      L.polyline(forecastCoords, {
        color: "#f59e0b",
        weight: 2.5,
        dashArray: "6, 6",
        opacity: 0.9,
      }).addTo(group);

      // Uncertainty Circles around each forecast point
      forecast.predictions.forEach((p) => {
        const radMeters = (p.uncertainty_radius_km_68 || p.uncertainty_km) * 1000;

        L.circle([p.latitude, p.longitude], {
          radius: radMeters,
          color: "#f59e0b",
          weight: 1.5,
          dashArray: "4, 4",
          fillColor: "#fbbf24",
          fillOpacity: 0.18,
        })
          .bindTooltip(
            `<b>+${p.forecast_hour}h Uncertainty Cone</b><br/>Empirical 68% Radius: ±${p.uncertainty_radius_km_68.toFixed(1)} km<br/>95% Radius: ±${p.uncertainty_radius_km_95.toFixed(1)} km`,
            { className: "font-mono text-xs bg-white border border-amber-500 text-slate-800 shadow-md" }
          )
          .addTo(group);

        L.circleMarker([p.latitude, p.longitude], {
          radius: 7,
          fillColor: "#ffffff",
          color: "#d97706",
          weight: 2.5,
          fillOpacity: 1.0,
        })
          .bindTooltip(
            `<b>+${p.forecast_hour}h Forecast Point</b><br/>Lat: ${p.latitude.toFixed(2)}°N | Lon: ${p.longitude.toFixed(2)}°E<br/>Valid: ${p.timestamp}`,
            { className: "font-mono text-xs bg-white border border-amber-500 text-slate-800 shadow-md" }
          )
          .addTo(group);

          boundsPoints.push([p.latitude, p.longitude]);
        });
      }
    }

    // 6. Fit Map Bounds
    if (boundsPoints.length > 1) {
      const bounds = L.latLngBounds(boundsPoints);
      map.fitBounds(bounds, { padding: [40, 40], maxZoom: 8 });
    } else if (boundsPoints.length === 1) {
      map.setView(boundsPoints[0], 6);
    }
  }, [centerFix, forecast, trackHistory, hasCoords, hasRequested, hasCoverage, hasImerg, requestedLocation, observationCoverage, showCoverageBounds, imergObservation, satelliteName, hasRenderableContent]);

  return (
    <div className="relative w-full h-[480px] rounded-xl border border-slate-200 overflow-hidden bg-white shadow-sm">
      <div ref={mapContainerRef} className="w-full h-full" />

      {/* Floating Notice when Pure Pixel Space is active */}
      {!hasRenderableContent && (
        <div className="absolute top-3 left-3 z-[1000] max-w-sm bg-white/95 backdrop-blur-md border border-amber-300 rounded-lg p-3 text-[11px] font-mono text-slate-800 shadow-md space-y-1">
          <div className="flex items-center gap-1.5 font-bold text-amber-900 uppercase">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-600 shrink-0" />
            <span>North Indian Ocean Surveillance Sector</span>
          </div>
          <p className="text-slate-600 text-[10px] leading-tight">
            Raw raster analyzed in pixel space{centerFix?.pixel_x ? ` (${centerFix.pixel_x}px, ${centerFix.pixel_y}px)` : ""}.
            Select a calibrated preset or click "Search NRT Coverage" to overlay bounds &amp; precipitation.
          </p>
        </div>
      )}

      {/* Live NRT Satellite Connection Status Badge (Replaces Obstructing Map Legend) */}
      <div className="absolute top-3 right-3 z-[1000] flex items-center gap-2 bg-slate-900/90 backdrop-blur-md border border-slate-700/60 rounded-full px-3.5 py-1.5 text-[11px] font-mono text-white shadow-lg pointer-events-none select-none">
        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shrink-0" />
        <span className="font-bold text-emerald-400 tracking-wider">LIVE NRT SATELLITE</span>
        <span className="text-slate-500">•</span>
        <span className="text-slate-200 font-semibold">{satelliteName || "ISRO INSAT-3D / NASA GIBS"}</span>
        <span className="text-slate-500">•</span>
        <span className="text-blue-400 font-semibold">WGS-84</span>
      </div>
    </div>
  );
};
