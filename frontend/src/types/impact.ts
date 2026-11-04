import { DataProvenance } from "./provenance";

export interface ComponentRisk {
  available: boolean;
  riskLevel: "LOW" | "MODERATE" | "HIGH" | "EXTREME" | "UNAVAILABLE" | string;
  status: string;
  source: string;
  sourceType: "MEASURED OBSERVATION" | "MODEL-DERIVED" | "DEMO / SCENARIO" | "USER PERTURBATION" | "UNAVAILABLE" | "EXPERIMENTAL" | string;
  method: string;
  explanation: string;
  metrics: Record<string, any>;
  disclaimer?: string | null;
}

export interface SettlementAtRisk {
  name: string;
  state: string;
  latitude: number;
  longitude: number;
  estimatedPopulation: number;
  distanceToCoreKm: number;
  hazardLevel: string;
}

export interface PopulationExposure {
  available: boolean;
  source: string;
  sourceType: string;
  totalExposedPopulation?: number | null;
  explanation: string;
  settlements: SettlementAtRisk[];
  disclaimer?: string | null;
}

export interface InfrastructureItem {
  id: string;
  name: string;
  type: "hospital" | "port" | "airport" | "power_grid" | "school" | "bridge" | string;
  latitude: number;
  longitude: number;
  riskLevel: string;
  distanceKm: number;
  source: string;
  sourceType: string;
}

export interface InfrastructureExposure {
  available: boolean;
  source: string;
  sourceType: string;
  criticalCount: number;
  items: InfrastructureItem[];
  explanation: string;
  disclaimer?: string | null;
}

export interface WhyExplanation {
  headline: string;
  factorBreakdown: string[];
  primaryDrivers: string[];
  engineType: string;
  limitations: string[];
}

export interface CompositeRiskAssessment {
  stormId: string;
  stormName?: string | null;
  riskLevel: "LOW" | "MODERATE" | "HIGH" | "EXTREME" | "UNKNOWN / INSUFFICIENT DATA" | string;
  riskIndex?: number | null;
  availableComponents: string[];
  unavailableComponents: string[];
  windRisk: ComponentRisk;
  rainfallRisk: ComponentRisk;
  floodRisk: ComponentRisk;
  coastalRisk: ComponentRisk;
  populationExposure: PopulationExposure;
  infrastructureExposure: InfrastructureExposure;
  whyExplanation: WhyExplanation;
  method: string;
  limitations: string;
  provenance: DataProvenance;
  isScenarioOutput: boolean;
  scenarioPerturbations?: Record<string, any> | null;
  generatedAt: string;
}

export interface ScenarioRequest {
  storm_id?: string;
  wind_speed_override_knots?: number;
  track_offset_lat?: number;
  track_offset_lon?: number;
  sst_c?: number;
  shear_knots?: number;
  rainfall_scenario_mm?: number;
  coastal_tide_surge_scenario_m?: number;
  include_demo_exposure?: boolean;
}

export interface RiskMapFeature {
  type: "Feature";
  properties: Record<string, any>;
  geometry: {
    type: "Polygon" | "Point" | "LineString";
    coordinates: any;
  };
}

export interface RiskMapGeoJSON {
  type: "FeatureCollection";
  features: RiskMapFeature[];
}
