export type ScientificDatasetStatus =
  | "AVAILABLE"
  | "NOT_DOWNLOADED"
  | "DOWNLOAD_REQUIRED"
  | "CREDENTIALS_REQUIRED"
  | "ACCESS_RESTRICTED"
  | "PROCESSING"
  | "VALIDATED"
  | "INVALID"
  | "DEMO_ONLY";

export type ScientificDataMode =
  | "REAL_RESEARCH"
  | "DEMO / SCENARIO"
  | "USER PERTURBATION"
  | "UNAVAILABLE";

export type QualityStatus = "PASS" | "WARNING" | "FAIL";

export interface DatasetMetadata {
  dataset_id: string;
  name: string;
  source: string;
  provider: string;
  description: string;
  variables: string[];
  units: Record<string, string>;
  format: string;
  coverage: string;
  temporal_resolution?: string;
  spatial_resolution?: string;
  time_range?: string;
  license: string;
  access_type: string;
  status: ScientificDatasetStatus;
  local_path?: string;
  source_url: string;
  retrieved_at?: string;
  checksum?: string;
  data_mode: ScientificDataMode;
  provenance: Record<string, any>;
}

export interface DataQualityReport {
  dataset_id: string;
  dataset_name: string;
  total_records: number;
  valid_records: number;
  invalid_records: number;
  missing_values: Record<string, number>;
  duplicate_timestamps: number;
  time_range?: string;
  spatial_coverage?: string;
  variables: string[];
  units: Record<string, string>;
  quality_status: QualityStatus;
  issues: string[];
  generated_at: string;
}

export interface NormalizedTrackRecord {
  storm_id: string;
  storm_name: string;
  basin: string;
  timestamp: string;
  latitude: number;
  longitude: number;
  max_wind_kt?: number;
  central_pressure_hpa?: number;
  movement_direction_deg?: number;
  movement_speed_kmh?: number;
  source: string;
  source_record_id?: string;
  data_mode: ScientificDataMode;
  provenance: Record<string, any>;
}

