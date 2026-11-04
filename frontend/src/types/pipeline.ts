/**
 * TypeScript Interfaces for Cyclone-X Scientific End-to-End AI Pipeline (Phase 10).
 */

export interface GeospatialMetadata {
  latitude_min?: number;
  latitude_max?: number;
  longitude_min?: number;
  longitude_max?: number;
  acquisition_time?: string;
  satellite?: string;
  sensor?: string;
  channel?: string;
  product_id?: string;
  source?: string;
  spatial_reference?: string;
  georeferencing_available: boolean;
}

export interface CenterFix {
  pixel_x: number | null;
  pixel_y: number | null;
  bounding_box_pixels?: [number, number, number, number] | null;
  symmetry_score?: number | null;
  localization_method: string;
  latitude: number | null;
  longitude: number | null;
  coordinates_available: boolean;
  spatial_note: string;
}

export interface ForecastPoint {
  forecast_hour: number;
  timestamp: string;
  latitude: number;
  longitude: number;
  uncertainty_km: number;
  uncertainty_radius_km_68: number;
  uncertainty_radius_km_95: number;
}

export interface TrajectoryForecast {
  model_name: string;
  model_version: string;
  origin_fix: {
    latitude: number;
    longitude: number;
    timestamp: string;
  };
  predictions: ForecastPoint[];
  uncertainty_method: string;
  uncertainty: Record<string, number>;
  is_experimental: boolean;
}

export interface RiskFactors {
  detection_confidence?: {
    status: string;
    value: number;
    score: number;
    weight: number;
    explanation?: string;
  };
  land_proximity?: string | {
    status: string;
    min_distance_to_coast_km: number;
    score: number;
    weight: number;
    assessment: string;
  };
  track_uncertainty?: string | {
    status: string;
    mean_uncertainty_radius_km: number;
    score: number;
    weight: number;
    note: string;
  };
  wind?: string | {
    status: string;
    value_knots: number;
    score: number;
    category: string;
    weight: number;
  };
  rainfall?: string | {
    status: string;
    value_24h_mm: number;
    score: number;
    category: string;
    weight: number;
  };
  sst?: string | {
    status: string;
    value_celsius: number;
    score: number;
    assessment: string;
    weight: number;
  };
  [key: string]: any;
}

export interface PipelineRiskAssessment {
  risk_score: number;
  risk_level: "NONE" | "LOW" | "MODERATE" | "HIGH" | "SEVERE";
  risk_factors: RiskFactors;
  risk_confidence: number;
  data_completeness: number;
  available_factors_count?: number;
  total_factors_count?: number;
  explanation: string;
  disclaimer: string;
}

export type PipelineDataMode = "DEMO" | "HISTORICAL" | "UPLOAD" | "LIVE_NRT" | "SCIENTIFIC_UPLOAD" | "LIVE_MONITOR";

export interface ScientificRasterMetadata {
  source_format: string;
  width: number;
  height: number;
  channels: number;
  dtype: string;
  crs?: string | null;
  transform?: number[] | null;
  bounds?: {
    latitude_min?: number;
    latitude_max?: number;
    longitude_min?: number;
    longitude_max?: number;
  } | null;
  resolution?: number[] | null;
  acquisition_time?: string | null;
  satellite?: string | null;
  sensor?: string | null;
  channel_name?: string | null;
  product_id?: string | null;
  source?: string | null;
  variable_names?: string[];
  attributes?: Record<string, any>;
  georeferencing_available: boolean;
}

export interface SatelliteUploadResponse {
  file_id: string;
  filename: string;
  format: string;
  size_bytes: number;
  metadata: ScientificRasterMetadata;
  georeferencing_available: boolean;
  validation_status: string;
  sha256: string;
  message: string;
}

export interface PipelineProvenance {
  // Unified Scientific Audit Trail & Provenance Fields
  image_source: string;
  image_dataset: string;
  image_id?: string | null;
  image_sha256?: string | null;
  data_mode: string;
  detection_model: string;
  detection_model_version: string;
  detection_checkpoint?: string | null;
  trajectory_model?: string | null;
  trajectory_model_version?: string | null;
  trajectory_checkpoint?: string | null;
  track_source?: string | null;
  track_id?: string | null;
  track_provenance?: string | null;
  analysis_timestamp_utc: string;

  // Legacy Backwards Compatibility Fields
  source?: string;
  acquisition_timestamp?: string;
  satellite?: string;
  sensor?: string;
  channel?: string;
  product_id?: string | null;
  georeferencing_available?: boolean;
  preprocessing_version?: string;
  detection_version?: string;
  inference_timestamp?: string;
  original_filename?: string | null;
  file_format?: string | null;
  crs?: string | null;
  bounds?: Record<string, number> | null;
  resolution?: number[] | null;
  geospatial_status?: string | null;
}

export interface PipelineAnalyzeResponse {
  detected: boolean;
  label: "CYCLONE" | "NO_CYCLONE";
  confidence: number;
  class_probabilities: Record<string, number>;
  structural_type: string;
  center_fix: CenterFix | null;
  geospatial_status: string;
  raster_metadata?: Record<string, any> | null;
  track_history_length: number;
  track_history_required: number;
  trajectory_status: string;
  trajectory_forecast: TrajectoryForecast | null;
  risk_assessment: PipelineRiskAssessment;
  provenance: PipelineProvenance;
  processing_status: string;
  error_detail?: string | null;
  disclaimer: string;
  [key: string]: any;
}

export interface PipelineStatusResponse {
  pipeline_status: string;
  phase9_detection: {
    model_name: string;
    version: string;
    architecture?: string;
    is_trained: boolean;
    status: string;
    checkpoint: string;
    metrics?: {
      accuracy?: number;
      accuracy_percentage?: number;
      precision?: number;
      recall?: number;
      f1_score?: number;
      specificity?: number;
      roc_auc?: number;
      test_samples?: number;
      total_dataset_samples?: number;
      geographic_scope?: string;
      threshold?: number;
      confusion_matrix?: Record<string, number>;
    };
  };
  phase8_trajectory: {
    model_name: string;
    version: string;
    is_trained: boolean;
    status: string;
    checkpoint: string;
    required_history_length: number;
    metrics?: {
      test_sequences?: number;
      overall_ml_mae_km?: number;
      by_horizon?: Record<string, any>;
    };
  };
  intensity_estimation?: {
    model_name: string;
    version: string;
    architecture?: string;
    is_trained: boolean;
    status: string;
    checkpoint: string;
    metrics?: {
      mae_knots?: number;
      rmse_knots?: number;
      medae_knots?: number;
      r2_score?: number;
      sample_count?: number;
      slice_breakdown_by_category?: Record<string, any>;
    };
  };
  risk_engine: {
    status: string;
    engine: string;
    factors_evaluated: string[];
    zero_fabrication_rule: string;
  };
  geospatial_capability: {
    status: string;
    validator: string;
    crs: string;
    zero_fabrication_rule: string;
  };
  supported_data_modes: string[];
  live_nrt_configured: boolean;
  disclaimer: string;
}

export interface RequestedLocation {
  latitude: number;
  longitude: number;
}

export interface NRTSearchObservation {
  latitude: number;
  longitude: number;
  min_latitude: number;
  max_latitude: number;
  min_longitude: number;
  max_longitude: number;
  bounds: {
    min_latitude: number;
    max_latitude: number;
    min_longitude: number;
    max_longitude: number;
  };
  acquisition_time: string;
  granule_id: string;
  satellite?: string;
  sensor?: string;
  dataset_id?: string;
  crs?: string;
  provider?: string;
  channel?: string;
  resolution?: string;
  grid_dimensions?: number[];
  sha256?: string;
}

export interface NRTLatLonSearchResponse {
  requested_location: RequestedLocation;
  observation: NRTSearchObservation | null;
  coverage: boolean;
  distance_km: number | null;
  source: string;
  status: string;
  message?: string;
  detection?: {
    is_cyclone?: boolean;
    confidence?: number;
    model_version?: string;
    intensity_category?: string;
    status?: string;
  } | null;
  provenance?: Record<string, any> | null;
  zero_fabrication_rule: string;
}

export interface NASAIMERGLocationResponse {
  requested_latitude: number;
  requested_longitude: number;
  grid_latitude?: number | null;
  grid_longitude?: number | null;
  precipitation?: number | null;
  precipitation_mm_per_hr?: number | null;
  precipitation_accum_mm?: number | null;
  unit: string;
  unit_scaled: string;
  scale_factor: number;
  is_valid_measurement: boolean;
  is_missing_no_data: boolean;
  observation_start?: string | null;
  observation_end?: string | null;
  observation_time?: string | null;
  granule_id?: string | null;
  product: string;
  version: string;
  provider: string;
  source: string;
  mode: string;
  coverage: boolean;
  status: string;
  message: string;
  provenance?: Record<string, any> | null;
  zero_fabrication_rule: string;
}

export interface NASAIMERGObservationSummary {
  granule_id: string;
  product: string;
  version: string;
  filename: string;
  file_path: string;
  sha256: string;
  observation_start: string;
  observation_end: string;
  download_time?: string | null;
  bounds: {
    min_latitude: number;
    max_latitude: number;
    min_longitude: number;
    max_longitude: number;
  };
  resolution_deg: number;
  crs: string;
  rows: number;
  cols: number;
  precipitation_variable: string;
  precipitation_units: string;
  scale_factor: number;
  valid_pixel_count: number;
  missing_pixel_count: number;
  min_precipitation_raw: number;
  max_precipitation_raw: number;
  mean_precipitation_raw: number;
  max_precipitation_mm_per_hr: number;
  provider: string;
  source: string;
  mode: string;
  status: string;
}


