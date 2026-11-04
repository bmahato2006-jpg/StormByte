import { DataProvenance } from "./provenance";

export interface AffectedDistrict {
  districtName: string;
  state: string;
  latitude: number;
  longitude: number;
  population: number;
  exposedPopulation: number;
  alertTier: "RED" | "ORANGE" | "YELLOW" | "GREEN" | string;
  closestApproachKm: number;
  arrivalEta: string;
  maxPredictedWindKt: number;
  maxPredictedGustKmh: number;
  stormSurgeMeters: number;
  rainfallHazardMm: number;
  inUncertaintyCone: boolean;
  criticalInfrastructureCount: number;
  evacuationAdvisory: string;
}

export interface AIHazardZones {
  redAlertPolygon: [number, number][];
  orangeAlertPolygon: [number, number][];
  yellowAlertPolygon: [number, number][];
}

export interface AIModelMetadata {
  detectionModel: string;
  trajectoryModel: string;
  intensityModel: string;
  vortexModel: string;
  nrtSatelliteSource: string;
  nrtSatelliteTimestamp: string;
  isLiveNRT: boolean;
}

export interface RiskAssessment {
  cycloneId: string;
  riskLevel: "LOW" | "MODERATE" | "HIGH" | "EXTREME" | string;
  riskIndex?: number | null;
  windRisk: string;
  rainfallRisk: string;
  coastalRisk: string;
  exposureRisk: string;
  surgeEstimateMeters?: number | null;
  estimatedPopulationAtRisk?: number | null;
  primaryLandfallDistrict?: string | null;
  affectedDistricts?: AffectedDistrict[];
  hazardZones?: AIHazardZones;
  aiModelMetadata?: AIModelMetadata;
  generatedAt: string;
  provenance: DataProvenance;
}


