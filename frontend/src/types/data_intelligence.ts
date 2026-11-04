/**
 * Cyclone-X Scientific Data Intelligence & Provenance Types.
 * Adheres strictly to Zero-Fabrication and Scientific Integrity rules.
 */

export type DataMode =
  | "DEMO"
  | "HISTORICAL"
  | "USER_UPLOAD"
  | "LIVE_NRT"
  | "MODEL_PREDICTION"
  | "OFFICIAL_INFO";

export type DataSourceStatus =
  | "ACTIVE"
  | "AVAILABLE"
  | "DEMO"
  | "NOT_CONFIGURED"
  | "UNAVAILABLE"
  | "DEGRADED";

export interface CycloneObservation {
  id: string;
  storm_id: string;
  storm_name: string;
  timestamp: string;
  latitude: number;
  longitude: number;
  wind_speed: number;
  central_pressure: number;
  classification: string;
  source: string;
  source_reference?: string;
  data_mode: DataMode;
  created_at: string;
}

export interface SatelliteObservation {
  id: string;
  satellite: string;
  sensor: string;
  product: string;
  timestamp: string;
  file_path: string;
  source_reference?: string;
  latitude_min: number;
  latitude_max: number;
  longitude_min: number;
  longitude_max: number;
  resolution?: string;
  processing_status: string;
  data_mode: DataMode;
  created_at: string;
}

export interface EnvironmentalObservation {
  id: string;
  timestamp: string;
  latitude: number;
  longitude: number;
  sea_surface_temperature?: number;
  rainfall?: number;
  wind_speed?: number;
  wind_direction?: number;
  pressure?: number;
  humidity?: number;
  source: string;
  data_mode: DataMode;
  created_at: string;
}

export interface DataProvenance {
  id: string;
  source: string;
  provider: string;
  product: string;
  dataset_version: string;
  acquisition_time?: string;
  processing_time?: string;
  source_reference?: string;
  license_notes?: string;
  data_mode: DataMode;
  status: string;
}

export interface DataSourceInfo {
  name: string;
  provider: string;
  description: string;
  supported_products: string[];
  status: DataSourceStatus;
  data_mode: DataMode;
  configured: boolean;
  requires_auth: boolean;
  last_checked: string;
  notes: string;
}

export interface DataStatusResponse {
  active_data_mode: DataMode;
  database_status: string;
  is_database_connected: boolean;
  total_cyclone_observations: number;
  total_satellite_observations: number;
  total_environmental_observations: number;
  total_provenance_records: number;
  sources: DataSourceInfo[];
  scientific_disclaimer: string;
}

export interface ImportHistoricalReport {
  records_received: number;
  records_imported: number;
  records_rejected: number;
  validation_errors: string[];
  imported_storm_id?: string;
  data_mode: DataMode;
  timestamp: string;
}

export interface UploadSatelliteResponse {
  id: string;
  filename: string;
  file_path: string;
  file_type: string;
  file_size_bytes: number;
  dimensions?: {
    width: number;
    height: number;
  };
  timestamp?: string;
  data_mode: DataMode;
  message: string;
}

export interface MosdacTelemetryStatus {
  is_live: boolean;
  last_scan_utc: string | null;
  granules_available: number;
  vortex_detected: boolean;
  active_cyclone_id: string | null;
  latency_ms?: number | null;
  operational_health: string;
  latest_granule_id?: string | null;
  latest_vortex_details?: {
    eye_pixel_x: number;
    eye_pixel_y: number;
    eye_confidence: number;
    wind_speed_knots: number;
    central_pressure_hpa: number;
    imd_category: string;
    model_name: string;
  } | null;
  last_poll_utc?: string | null;
  polling_interval_minutes?: number;
}


export interface ImergTelemetryStatus {
  is_live: boolean;
  last_scan_utc: string | null;
  granules_available: number;
  max_rain_rate_mm_hr?: number | null;
  mean_rain_rate_mm_hr?: number | null;
  heavy_rain_area_km2?: number | null;
  latency_ms?: number | null;
  operational_health: string;
  latest_granule_id?: string | null;
  dataset_name?: string;
  tensor_shape?: number[] | null;
  last_poll_utc?: string | null;
}

export interface EnvironmentTelemetryStatus {
  is_live: boolean;
  last_scan_utc?: string | null;
  sea_surface_temperature?: number | null;
  vertical_wind_shear?: number | null;
  mean_sea_level_pressure?: number | null;
  relative_humidity?: number | null;
  ocean_heat_content?: number | null;
  source: string;
  data_mode: string;
  operational_health: string;
  granules_available: number;
  latency_ms?: number | null;
}
