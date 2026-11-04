import { DataProvenance } from "./provenance";

export interface EarlyGenesisAlert {
  probability: number;
  probabilityPercent: number;
  riskTier: "LOW" | "MODERATE" | "HIGH" | "CRITICAL" | string;
  intensification24to48h: boolean;
  preCycloneStage: string;
  disturbanceRadiusKm: number;
  circulationCenter: {
    latitude: number;
    longitude: number;
  };
  environmentalFactors?: {
    seaSurfaceTemperature?: number;
    verticalWindShearKt?: number;
    shearFavorable?: boolean;
    relativeVorticity850?: number;
    relativeHumidity700?: number;
    oceanHeatContent?: number;
    tccGrowthRateKm2Hr?: number;
    tccPersistenceHours?: number;
    [key: string]: any;
  };
  headline: string;
}

export interface Cyclone {
  id: string;
  name: string;
  basin: string;
  latitude: floatNumber;
  longitude: floatNumber;
  timestamp: string;
  intensity: string;
  windSpeed: number; // knots
  pressure: number; // hPa
  movementDirection: string;
  movementSpeed: number; // km/h
  dataSource: string;
  dataTimestamp: string;
  provenance: DataProvenance;
  earlyGenesis?: EarlyGenesisAlert;
}

type floatNumber = number;

export interface CycloneSummary {
  id: string;
  name: string;
  basin: string;
  latitude: number;
  longitude: number;
  intensity: string;
  windSpeed: number;
  pressure: number;
  dataType: string;
  timestamp: string;
  earlyGenesis?: EarlyGenesisAlert;
}

export interface TrackPoint {
  timestamp: string;
  latitude: number;
  longitude: number;
  intensity: string;
  windSpeed: number;
  pressure: number;
}

export interface CycloneTrackResponse {
  cycloneId: string;
  name: string;
  track: TrackPoint[];
  provenance: DataProvenance;
}

