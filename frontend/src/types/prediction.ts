import { DataProvenance } from "./provenance";

export interface PredictionPoint {
  forecastTime: string; // "+6h", "+12h", "+24h"
  leadHours?: number;
  latitude: number;
  longitude: number;
  uncertainty: number; // radius in km
  semiMajorKm?: number;
  semiMinorKm?: number;
  headingDeg?: number;
  predictedWindSpeed?: number;
  predictedPressure?: number;
  intensityCategory?: string;
  intensityTrend?: string;
  modelVersion: string;
  generatedAt: string;
}

export interface PredictionResponse {
  cycloneId: string;
  predictions: PredictionPoint[];
  uncertaintyPolygon: [number, number][]; // [lat, lon] closed loop
  modelVersion: string;
  generatedAt: string;
  provenance: DataProvenance;
  environmentalObservationsAvailable?: boolean;
  sstAvailable?: boolean;
  windShearAvailable?: boolean;
  environmentalSourceType?: "MEASURED_OBSERVATION" | "DEMO_SCENARIO" | "USER_PERTURBATION" | "NONE" | string;
}

export interface HollandRadialSample {
  radius_km: number;
  wind_speed_knots: number;
}

export interface QuadrantRadii {
  r34_km: number;
  r50_km: number;
  r64_km: number;
}

export interface HollandVortexProfile {
  storm_id: string;
  central_pressure_hpa: number;
  environmental_pressure_hpa: number;
  central_pressure_deficit_hpa: number;
  holland_b: number;
  radius_max_wind_km: number;
  peak_gradient_wind_knots: number;
  quadrant_radii: Record<string, QuadrantRadii>;
  radial_curve: HollandRadialSample[];
  scientific_note: string;
}

export interface TrajectoryForecastRequest {
  storm_id?: string;
  track_points: Array<{
    latitude: number;
    longitude: number;
    timestamp?: string;
    windSpeed?: number;
    pressure?: number;
  }>;
  steering_u_knots?: number;
  steering_v_knots?: number;
  sst_c?: number | null;
  shear_knots?: number | null;
  horizons_hours?: number[];
  environmental_source_type?: string;
}

export interface EvidenceItem {
  factor: string;
  status: "PRESENT" | "ABSENT" | "UNAVAILABLE" | string;
  value?: any;
  threshold: string;
  favorable: boolean;
  details: string;
}

export interface TemporalPersistence {
  window_hours: number;
  trend: "STABLE" | "INTENSIFYING" | "DISSIPATING" | "INSUFFICIENT_HISTORY" | "NOT_APPLICABLE" | string;
  observations_evaluated: number;
  details: string;
}

export interface OceanRiskLocation {
  latitude: number;
  longitude: number;
  radius_km: number;
  basin: string;
  is_ocean: boolean;
  is_equatorial_restricted: boolean;
}

export interface OceanRiskResponse {
  location: OceanRiskLocation;
  risk_level: "VERY LOW" | "LOW" | "MODERATE" | "HIGH" | "VERY HIGH" | string;
  risk_score: number;
  model_estimated_likelihood: string;
  probability_calibrated: boolean;
  evidence_checklist: EvidenceItem[];
  temporal_persistence: TemporalPersistence;
  key_drivers: string[];
  limitations: string[];
  reasoning: string;
  experimental_notice: string;
  disclaimer: string;
  provenance?: Record<string, any>;
}

export interface OceanRiskStatusResponse {
  service_status: string;
  supported_basins: string[];
  data_sources: Record<string, any>;
  scientific_governance: Record<string, any>;
}

