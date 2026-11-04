import { DataProvenance } from "./provenance";

export interface EnvironmentalConditions {
  cycloneId: string;
  seaSurfaceTemperature: number; // °C
  verticalWindShear: number; // knots
  meanSeaLevelPressure: number; // hPa
  relativeHumidityMidTroposphere: number; // %
  outflowTemperature: number; // °C
  oceanHeatContent?: number; // kJ/cm²
  timestamp: string;
  provenance: DataProvenance;
}

