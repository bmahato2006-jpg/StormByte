/**
 * TypeScript definitions for Phase 4: Satellite AI Analysis & Computer Vision.
 */

export type AnalysisMode = "DEMO" | "RESEARCH" | "LIVE";
export type ConfidenceTier = "HIGH" | "MEDIUM" | "LOW" | "UNAVAILABLE";

export interface CenterDetectionResult {
  pixel_x: number;
  pixel_y: number;
  estimated_radius_pixels: number;
  confidence: ConfidenceTier;
  confidence_score: number;
  method: string;
  georeferenced: boolean;
  latitude: number | null;
  longitude: number | null;
  status_message: string;
}

export interface ClassificationResult {
  category: string;
  confidence: ConfidenceTier;
  is_experimental: boolean;
  criteria_summary: string;
  disclaimer: string;
}

export interface IntensityResult {
  available: boolean;
  wind_speed_knots: number | null;
  central_pressure_hpa: number | null;
  category: string | null;
  uncertainty_range: string | null;
  status_message: string;
}

export interface ExplainabilityResult {
  explanation_type: string;
  explanation_text: string;
  key_visual_regions: Array<Record<string, any>>;
  overlay_image_url: string | null;
  overlay_base64: string | null;
  model_version: string;
}

export interface PreprocessingMetadata {
  original_dimensions: { width: number; height: number };
  processed_dimensions: { width: number; height: number };
  channels: number;
  quality_score: number;
  sharpness: number;
  dynamic_range: number;
  operations_applied: string[];
}

export interface SatelliteAnalysisResponse {
  analysis_id: string;
  source: string;
  source_type: string;
  timestamp: string;
  processing_timestamp: string;
  image_hash: string;
  analysis_mode: AnalysisMode;
  model_name: string;
  model_version: string;
  provenance_id: string;
  preprocessing: PreprocessingMetadata;
  detection_detected: boolean;
  detection_confidence: ConfidenceTier;
  candidate_bounding_box: {
    x_min: number;
    y_min: number;
    x_max: number;
    y_max: number;
    width?: number;
    height?: number;
  } | null;
  center: CenterDetectionResult;
  classification: ClassificationResult;
  intensity: IntensityResult;
  explainability: ExplainabilityResult;
  scientific_disclaimer: string;
}

export interface ModelInfo {
  name: string;
  version: string;
  framework: string;
  input_requirements: string;
  output_schema: string;
  training_status: "ACTIVE" | "NOT_TRAINED";
  validation_status: string;
  description: string;
}

export interface ModelRegistryResponse {
  active_detector: string;
  models: ModelInfo[];
  notice: string;
}
