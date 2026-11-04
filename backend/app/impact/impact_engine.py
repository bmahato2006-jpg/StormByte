"""Impact Intelligence Orchestrator (Phase 6).

Coordinates wind, rainfall, coastal, population, and infrastructure analysis modules.
Generates rule-based explainability, GeoJSON risk layers, and chained data provenance.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Dict, List, Optional
from backend.app.core.provenance import create_provenance
from backend.app.impact.coastal_risk import coastal_risk_analyzer
from backend.app.impact.exposure import exposure_analyzer
from backend.app.impact.flood_risk import flood_risk_analyzer
from backend.app.impact.impact_models import (
    CompositeRiskAssessment,
    RiskMapGeoJSON,
    ScenarioRequest,
    WhyExplanation,
)
from backend.app.impact.rainfall_risk import rainfall_risk_analyzer
from backend.app.impact.risk_scoring import risk_scoring_engine
from backend.app.impact.wind_risk import wind_risk_analyzer
from backend.app.ml.pipeline.risk_engine import INDIAN_COASTAL_POINTS, haversine_km
from backend.app.schemas.provenance import DataProvenance


class ImpactIntelligenceEngine:
    """Central orchestrator for Phase 6 Disaster Decision-Support."""

    @classmethod
    def evaluate_impact(
        cls,
        cyclone_data: Dict[str, Any],
        prediction_data: Optional[Dict[str, Any]] = None,
        scenario_params: Optional[ScenarioRequest] = None,
        parent_hash: Optional[str] = None,
    ) -> CompositeRiskAssessment:
        storm_id = cyclone_data.get("id", "CUSTOM_RUN")
        storm_name = cyclone_data.get("name", "UNNAMED_CYCLONE")
        basin = cyclone_data.get("basin", "North Indian Ocean")

        obs = cyclone_data.get("currentObservation", cyclone_data)
        base_lat = float(
            obs.get("latitude")
            or (obs.get("center", {}).get("lat") if isinstance(obs.get("center"), dict) else None)
            or 15.0
        )
        base_lon = float(
            obs.get("longitude")
            or (obs.get("center", {}).get("lon") if isinstance(obs.get("center"), dict) else None)
            or 85.0
        )
        raw_wind = (
            obs.get("windSpeed")
            or obs.get("wind_speed")
            or obs.get("max_wind_speed_knots")
            or obs.get("wind_speed_knots")
            or obs.get("intensity_knots")
        )
        base_wind = float(raw_wind) if raw_wind is not None else None
        raw_press = (
            obs.get("pressure")
            or obs.get("minimum_pressure_hpa")
            or obs.get("central_pressure_hpa")
        )
        base_pressure = float(raw_press) if raw_press is not None else None

        is_scenario = scenario_params is not None
        scenario_perturbations: Dict[str, Any] = {}

        # Apply user scenario perturbations if provided
        eff_lat = base_lat
        eff_lon = base_lon
        eff_wind = base_wind
        eff_pressure = base_pressure
        scenario_rain_mm = None
        scenario_surge_m = None

        if is_scenario and scenario_params:
            if scenario_params.track_offset_lat is not None:
                eff_lat += scenario_params.track_offset_lat
                scenario_perturbations["track_offset_lat"] = scenario_params.track_offset_lat
            if scenario_params.track_offset_lon is not None:
                eff_lon += scenario_params.track_offset_lon
                scenario_perturbations["track_offset_lon"] = scenario_params.track_offset_lon
            if scenario_params.wind_speed_override_knots is not None:
                eff_wind = scenario_params.wind_speed_override_knots
                scenario_perturbations["wind_speed_override_knots"] = scenario_params.wind_speed_override_knots
            if scenario_params.rainfall_scenario_mm is not None:
                scenario_rain_mm = scenario_params.rainfall_scenario_mm
                scenario_perturbations["rainfall_scenario_mm"] = scenario_params.rainfall_scenario_mm
            if scenario_params.coastal_tide_surge_scenario_m is not None:
                scenario_surge_m = scenario_params.coastal_tide_surge_scenario_m
                scenario_perturbations["coastal_tide_surge_scenario_m"] = scenario_params.coastal_tide_surge_scenario_m

        is_cyclonic = (eff_wind is not None and eff_wind >= 34.0)

        # 1. Wind Risk
        vortex_info = cyclone_data.get("digitalTwin", {})
        quad_radii = vortex_info.get("quadrantRadii", {}).get("R34") if vortex_info else None
        wind_comp = wind_risk_analyzer.evaluate(
            wind_speed_knots=eff_wind,
            quadrant_radii=quad_radii if quad_radii else None,
            is_perturbation=is_scenario and scenario_params.wind_speed_override_knots is not None,
        )

        # 2. Rainfall Risk (Connects authentic NASA IMERG precipitation when available)
        measured_rain = cyclone_data.get("measured_rainfall_mm")
        if measured_rain is None:
            env_obs = cyclone_data.get("environmental_obs", {})
            measured_rain = env_obs.get("rainfall_rate_mm_hr") or env_obs.get("measured_rainfall_mm")

        # Automatically check authentic local NASA IMERG observation if coordinate is georeferenced
        if measured_rain is None and not is_scenario:
            try:
                from backend.app.data.geospatial.nasa_imerg_service import nasa_imerg_service
                imerg_lookup = nasa_imerg_service.lookup_location(eff_lat, eff_lon)
                if imerg_lookup.coverage and imerg_lookup.is_valid_measurement and imerg_lookup.precipitation_mm_per_hr is not None:
                    measured_rain = imerg_lookup.precipitation_mm_per_hr * 24.0 if imerg_lookup.precipitation_accum_mm is None else imerg_lookup.precipitation_accum_mm
            except Exception:
                pass

        rain_comp = rainfall_risk_analyzer.evaluate(
            measured_rainfall_mm=measured_rain,
            scenario_rainfall_mm=scenario_rain_mm,
            is_demo_mode=False,
            demo_scenario_mm=None,
        )

        # 3. Flood Risk
        flood_comp = flood_risk_analyzer.evaluate(rain_comp)

        # 4. Coastal Storm Surge Risk (Compute authentic geodesic distance to Indian coastline)
        dist_coast = round(
            min(haversine_km(eff_lat, eff_lon, pt[0], pt[1]) for pt in INDIAN_COASTAL_POINTS), 1
        )
        coastal_comp = coastal_risk_analyzer.evaluate(
            central_pressure_hpa=eff_pressure,
            max_wind_knots=eff_wind,
            distance_to_coast_km=dist_coast,
            scenario_surge_m=scenario_surge_m,
            is_demo_mode=False,
        )

        # 5. Population Exposure (Census of India Coastal District Atlas)
        hazard_r = wind_comp.metrics.get("r34_max_km", 160.0) if wind_comp.available else 120.0
        pop_comp = exposure_analyzer.evaluate_population(
            cyclone_lat=eff_lat,
            cyclone_lon=eff_lon,
            hazard_radius_km=hazard_r,
            max_wind_knots=eff_wind,
            is_scenario=is_scenario,
            basin=basin,
        )

        # 6. Infrastructure Exposure
        infra_comp = exposure_analyzer.evaluate_infrastructure(
            cyclone_lat=eff_lat,
            cyclone_lon=eff_lon,
            hazard_radius_km=hazard_r,
            max_wind_knots=eff_wind,
            is_scenario=is_scenario,
            basin=basin,
        )

        # 7. Composite Risk Synthesis
        comp_level, risk_idx, avail_comps, unavail_comps, method_note = risk_scoring_engine.compute_composite_score(
            wind_risk=wind_comp,
            rainfall_risk=rain_comp,
            flood_risk=flood_comp,
            coastal_risk=coastal_comp,
            population_exposure=pop_comp,
            infrastructure_exposure=infra_comp,
        )

        # If basin is calm / non-cyclonic, adjust composite risk level truthfully
        if not is_cyclonic:
            comp_level = "LOW"
            if risk_idx is not None:
                risk_idx = min(risk_idx, 15.0)

        # 8. AI "Why?" Factor Breakdown Engine (Rule-based)
        factors: List[str] = []
        drivers: List[str] = []

        if wind_comp.available:
            if is_cyclonic:
                factors.append(f"Wind hazard classified as {wind_comp.risk_level}: Maximum sustained winds at {wind_comp.metrics.get('max_wind_speed_knots', 'N/A')} kt ({wind_comp.metrics.get('max_wind_speed_kmh', 'N/A')} km/h).")
                if wind_comp.risk_level in ("HIGH", "EXTREME"):
                    drivers.append(f"Destructive Wind Core ({wind_comp.metrics.get('max_wind_speed_knots', '')} kt)")
            else:
                factors.append(f"Wind state classified as LOW / CALM: Sustained breeze at {eff_wind:.1f} kt. Below tropical depression gale threshold (34 kt).")

        if coastal_comp.available:
            if is_cyclonic:
                factors.append(f"Coastal surge proxy rated {coastal_comp.risk_level}: Peak anomaly estimated at ~{coastal_comp.metrics.get('estimated_surge_peak_meters', 'N/A')} m above normal astronomical tide.")
                if coastal_comp.risk_level in ("HIGH", "EXTREME"):
                    drivers.append(f"Severe Coastal Surge Setup (~{coastal_comp.metrics.get('estimated_surge_peak_meters', '')}m)")
            else:
                factors.append(f"Coastal surge rated LOW: Normal astronomical tidal variance (~{coastal_comp.metrics.get('estimated_surge_peak_meters', 0.2)} m). No storm surge anomaly in basin.")

        if rain_comp.available:
            rain_val = rain_comp.metrics.get('accumulated_24h_rainfall_mm')
            if is_cyclonic or (rain_val is not None and rain_val > 50):
                factors.append(f"Precipitation inundation rated {rain_comp.risk_level} ({rain_comp.source_type}): {rain_val} mm/24h.")
                if rain_comp.risk_level in ("HIGH", "EXTREME"):
                    drivers.append(f"Heavy Inundation Runoff ({rain_val}mm)")
            else:
                factors.append(f"Rainfall rated LOW: Scattered ambient convective showers ({rain_val or 0:.1f} mm/24h).")
        else:
            factors.append("Ambient oceanic precipitation within normal seasonal thresholds.")

        if pop_comp.available and pop_comp.total_exposed_population and pop_comp.total_exposed_population > 0:
            factors.append(f"Population exposure (Census of India): ~{pop_comp.total_exposed_population:,} residents in {len(pop_comp.settlements)} vulnerable coastal districts within hazard reach.")
            if len(pop_comp.settlements) >= 3:
                drivers.append(f"High Coastal Settlement Exposure ({len(pop_comp.settlements)} districts)")
        else:
            factors.append("Zero coastal districts under cyclonic gale threat; maritime open-water circulation poses no onshore population hazard.")

        if not drivers:
            drivers = ["Normal Ambient Trade Winds", "Normal Astronomical Sea State"] if not is_cyclonic else ["General Atmospheric Circulation"]

        headline = f"Composite impact rated {comp_level} (Model-Derived Risk Index: {risk_idx if risk_idx is not None else 'N/A'}/100) — {'Basin Calm / Normal Oceanic State' if not is_cyclonic else 'Active Cyclonic Threat Analysis'}."

        limitations = [
            "Analytical wind field assumes idealized Holland vortex symmetry and does not model localized convective gusts.",
            "Storm-surge metric is an empirical inverted-barometer proxy and does not resolve 2D bathymetric shallow-water wave setup.",
            "Operational Doppler radar and real-time census raster feeds are not connected to this environment.",
            "This assessment is an experimental decision-support model and NOT an official government evacuation advisory.",
        ]

        why_explanation = WhyExplanation(
            headline=headline,
            factor_breakdown=factors,
            primary_drivers=drivers,
            engine_type="RULE-BASED EXPLANATION ENGINE",
            limitations=limitations,
        )

        # 9. Provenance Chaining
        eff_parent_hash = parent_hash
        if not eff_parent_hash:
            raw_str = f"{storm_id}:{obs.get('timestamp')}:{eff_lat}:{eff_lon}:{eff_wind}"
            eff_parent_hash = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]

        prov_dict = create_provenance(
            source="IMPACT_INTELLIGENCE_ENGINE",
            processing_version="v0.6.0-phase6",
            model_version="impact_risk_engine-v0.6.0",
            data_type="RISK_ASSESSMENT",
            parent_hash=eff_parent_hash,
        )
        prov_dict["disclaimer"] = "EXPERIMENTAL AI DECISION-SUPPORT — NOT AN OFFICIAL IMD ADVISORY. ALL OUTPUTS FOR EVALUATION ONLY."
        prov = DataProvenance(**prov_dict)

        return CompositeRiskAssessment(
            storm_id=storm_id,
            storm_name=storm_name,
            risk_level=comp_level,
            risk_index=risk_idx,
            available_components=avail_comps,
            unavailable_components=unavail_comps,
            wind_risk=wind_comp,
            rainfall_risk=rain_comp,
            flood_risk=flood_comp,
            coastal_risk=coastal_comp,
            population_exposure=pop_comp,
            infrastructure_exposure=infra_comp,
            why_explanation=why_explanation,
            method=method_note,
            limitations="Outputs are model-derived indicators. Under no circumstances should operational disaster decisions be made from this tool.",
            provenance=prov,
            is_scenario_output=is_scenario,
            scenario_perturbations=scenario_perturbations if is_scenario else None,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    @classmethod
    def generate_risk_map_geojson(
        cls,
        assessment: CompositeRiskAssessment,
        cyclone_data: Dict[str, Any],
        prediction_data: Optional[Dict[str, Any]] = None,
    ) -> RiskMapGeoJSON:
        """Generates geospatial GeoJSON features representing multi-hazard impact perimeters."""
        features: List[Dict[str, Any]] = []

        obs = cyclone_data.get("currentObservation", cyclone_data)
        center_lat = float(obs.get("latitude", 15.0))
        center_lon = float(obs.get("longitude", 85.0))
        max_wind = assessment.wind_risk.metrics.get("max_wind_speed_knots", 0)
        is_cyclonic = max_wind >= 34.0

        def create_circle_poly(c_lat: float, c_lon: float, radius_km: float, points: int = 36) -> List[List[float]]:
            coords = []
            for i in range(points + 1):
                angle = math.radians((i * 360.0) / points)
                d_lat = (radius_km * math.cos(angle)) / 111.0
                d_lon = (radius_km * math.sin(angle)) / (111.0 * max(0.1, math.cos(math.radians(c_lat))))
                coords.append([round(c_lon + d_lon, 4), round(c_lat + d_lat, 4)])
            return coords

        if not is_cyclonic:
            # Gentle ambient marine observation zone
            features.append({
                "type": "Feature",
                "properties": {
                    "layer": "WIND_RISK",
                    "hazard_type": f"Ambient Marine Surface Observation Zone ({max_wind:.0f} kt)",
                    "radius_km": 140.0,
                    "risk_level": "LOW",
                    "data_status": assessment.wind_risk.source_type,
                    "disclaimer": "Basin Calm — Normal Oceanic Operational Environment",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [create_circle_poly(center_lat, center_lon, 140.0)],
                },
            })
        else:
            # 1. Wind Hazard Rings (R34, R64)
            r34 = assessment.wind_risk.metrics.get("r34_max_km", 180.0)
            features.append({
                "type": "Feature",
                "properties": {
                    "layer": "WIND_RISK",
                    "hazard_type": "Gale-Force Wind Swath (>= 34 kt)",
                    "radius_km": r34,
                    "risk_level": "MODERATE",
                    "data_status": assessment.wind_risk.source_type,
                    "disclaimer": "Holland Vortex Gradient Wind Swath",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [create_circle_poly(center_lat, center_lon, r34)],
                },
            })

            # R64 Hurricane Destructive Core
            if max_wind >= 64.0:
                r64 = assessment.wind_risk.metrics.get("r64_max_km", 55.0)
                features.append({
                    "type": "Feature",
                    "properties": {
                        "layer": "WIND_RISK",
                        "hazard_type": "Core Eyewall Destructive Wind (>= 64 kt)",
                        "radius_km": r64,
                        "risk_level": "EXTREME",
                        "data_status": assessment.wind_risk.source_type,
                        "disclaimer": "Hurricane core destruction radius",
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [create_circle_poly(center_lat, center_lon, r64)],
                    },
                })

            # 2. Coastal Storm Surge Hazard Coastal Buffer
            if assessment.coastal_risk.available and assessment.coastal_risk.risk_level in ("MODERATE", "HIGH", "EXTREME"):
                surge_m = assessment.coastal_risk.metrics.get("estimated_surge_peak_meters", 3.0)
                # Position surge danger sector on nearest coastline
                nearest_pt = min(INDIAN_COASTAL_POINTS, key=lambda p: haversine_km(center_lat, center_lon, p[0], p[1]))
                c_lat, c_lon = nearest_pt
                features.append({
                    "type": "Feature",
                    "properties": {
                        "layer": "COASTAL_SURGE",
                        "hazard_type": f"Coastal Storm Surge Danger Belt (~{surge_m:.1f} m)",
                        "surge_amplitude_m": surge_m,
                        "risk_level": assessment.coastal_risk.risk_level,
                        "data_status": assessment.coastal_risk.source_type,
                        "disclaimer": "INVERTED BAROMETER & SHALLOW-SHELF HYDRODYNAMIC PROXY",
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [round(c_lon - 0.5, 4), round(c_lat - 0.5, 4)],
                            [round(c_lon + 0.6, 4), round(c_lat - 0.2, 4)],
                            [round(c_lon + 0.5, 4), round(c_lat + 0.6, 4)],
                            [round(c_lon - 0.4, 4), round(c_lat + 0.4, 4)],
                            [round(c_lon - 0.5, 4), round(c_lat - 0.5, 4)],
                        ]],
                    },
                })

        # 3. Population Settlements (Census of India Districts)
        if assessment.population_exposure.available and assessment.population_exposure.settlements:
            for s in assessment.population_exposure.settlements:
                features.append({
                    "type": "Feature",
                    "properties": {
                        "layer": "POPULATION_EXPOSURE",
                        "settlement_name": s.name,
                        "state": s.state,
                        "population": s.estimated_population,
                        "distance_km": s.distance_to_core_km,
                        "hazard_level": s.hazard_level,
                        "data_status": assessment.population_exposure.source_type,
                        "disclaimer": "CENSUS OF INDIA OFFICIAL GEODATA",
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [s.longitude, s.latitude],
                    },
                })

        # 4. Critical Infrastructure Points
        if assessment.infrastructure_exposure.available and assessment.infrastructure_exposure.items:
            for item in assessment.infrastructure_exposure.items:
                features.append({
                    "type": "Feature",
                    "properties": {
                        "layer": "INFRASTRUCTURE_EXPOSURE",
                        "facility_id": item.id,
                        "facility_name": item.name,
                        "facility_type": item.type,
                        "risk_level": item.risk_level,
                        "distance_km": item.distance_km,
                        "data_status": assessment.infrastructure_exposure.source_type,
                        "disclaimer": "NATIONAL CRITICAL INFRASTRUCTURE REGISTER",
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [item.longitude, item.latitude],
                    },
                })

        return RiskMapGeoJSON(features=features)


impact_engine = ImpactIntelligenceEngine()

