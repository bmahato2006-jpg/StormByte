import { DataProvenance } from "./provenance";

export interface QuadrantValues {
  NE: number;
  SE: number;
  SW: number;
  NW: number;
}

export interface QuadrantRadii {
  R34: QuadrantValues;
  R50: QuadrantValues;
  R64: QuadrantValues;
}

export interface RadialWindPoint {
  radiusKm: number;
  windSpeedKnots: number;
}

export interface DigitalTwinResponse {
  cycloneId: string;
  stormName: string;
  radiusMaxWindKm: number;
  centralPressureHpa: number;
  maxWindKnots: number;
  hollandB: number;
  coriolisF: number;
  quadrantRadii: QuadrantRadii;
  radialProfile: RadialWindPoint[];
  rightFrontQuadrantRisk: string;
  provenance: DataProvenance;
}

