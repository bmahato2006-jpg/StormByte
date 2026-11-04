import { Cyclone, CycloneSummary, CycloneTrackResponse } from "@/types/cyclone";
import { PredictionResponse, OceanRiskResponse, OceanRiskStatusResponse } from "@/types/prediction";
import { RiskAssessment, AffectedDistrict } from "@/types/risk";

import { EnvironmentalConditions } from "@/types/environment";
import { DigitalTwinResponse } from "@/types/digital_twin";
import { HealthStatus, MLSystemStatus } from "@/types/health";
import { MosdacTelemetryStatus, ImergTelemetryStatus, EnvironmentTelemetryStatus } from "@/types/data_intelligence";

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  (typeof window !== "undefined" ? "/api" : "http://127.0.0.1:8000/api");

async function fetchJson<T>(endpoint: string): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, {
      headers: {
        "Accept": "application/json",
      },
      cache: "no-store",
    });
    if (!res.ok) {
      throw new Error(`API error ${res.status}: ${res.statusText} on ${url}`);
    }
    return (await res.json()) as T;
  } catch (err) {
    console.warn(`[CYCLONE-X Client] Fetch failed on ${url}:`, err);
    throw err;
  }
}

async function postJson<T>(endpoint: string, body?: any): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Accept": "application/json",
        "Content-Type": "application/json",
      },
      body: body ? JSON.stringify(body) : undefined,
      cache: "no-store",
    });
    if (!res.ok) {
      throw new Error(`API error ${res.status}: ${res.statusText} on ${url}`);
    }
    return (await res.json()) as T;
  } catch (err) {
    console.warn(`[CYCLONE-X Client] POST failed on ${url}:`, err);
    throw err;
  }
}

export async function fetchHealth(): Promise<HealthStatus> {
  return fetchJson<HealthStatus>("/health");
}

export async function fetchCyclones(): Promise<CycloneSummary[]> {
  return fetchJson<CycloneSummary[]>("/cyclones");
}

export async function fetchCyclone(id: string): Promise<Cyclone> {
  return fetchJson<Cyclone>(`/cyclones/${id}`);
}

export async function fetchTrack(id: string): Promise<CycloneTrackResponse> {
  return fetchJson<CycloneTrackResponse>(`/cyclones/${id}/track`);
}

export async function fetchPrediction(id: string): Promise<PredictionResponse> {
  return fetchJson<PredictionResponse>(`/cyclones/${id}/prediction`);
}

export async function fetchRisk(id: string): Promise<RiskAssessment> {
  return fetchJson<RiskAssessment>(`/cyclones/${id}/risk`);
}

export async function fetchAffectedDistricts(id: string): Promise<AffectedDistrict[]> {
  return fetchJson<AffectedDistrict[]>(`/cyclones/${id}/affected-districts`);
}

export async function syncLiveSatellite(): Promise<{
  status: string;
  message: string;
  timestamp: string;
  cyclone: Cyclone;
  prediction: PredictionResponse;
  risk: RiskAssessment;
}> {
  return postJson("/cyclones/live/sync-satellite");
}

export async function fetchEnvironment(id: string): Promise<EnvironmentalConditions> {
  return fetchJson<EnvironmentalConditions>(`/cyclones/${id}/environment`);
}

export async function fetchDigitalTwin(id: string): Promise<DigitalTwinResponse> {
  return fetchJson<DigitalTwinResponse>(`/cyclones/${id}/digital-twin`);
}

export async function fetchMLStatus(): Promise<MLSystemStatus> {
  return fetchJson<MLSystemStatus>("/ml/status");
}

// ---------------------------------------------------------------------------
// Phase 3: Scientific Data Intelligence & Provenance
// ---------------------------------------------------------------------------
import {
  DataStatusResponse,
  DataSourceInfo,
  DataProvenance,
  CycloneObservation,
  UploadSatelliteResponse,
  ImportHistoricalReport,
} from "@/types/data_intelligence";

export async function fetchDataStatus(): Promise<DataStatusResponse> {
  return fetchJson<DataStatusResponse>("/data/status");
}

export async function fetchDataSources(): Promise<DataSourceInfo[]> {
  return fetchJson<DataSourceInfo[]>("/data/sources");
}

export async function fetchProvenance(limit: number = 50): Promise<DataProvenance[]> {
  return fetchJson<DataProvenance[]>(`/provenance?limit=${limit}`);
}

export async function fetchObservations(params?: {
  stormId?: string;
  dataMode?: string;
  source?: string;
  limit?: number;
}): Promise<CycloneObservation[]> {
  const q = new URLSearchParams();
  if (params?.stormId) q.append("storm_id", params.stormId);
  if (params?.dataMode) q.append("data_mode", params.dataMode);
  if (params?.source) q.append("source", params.source);
  if (params?.limit) q.append("limit", params.limit.toString());
  const queryStr = q.toString() ? `?${q.toString()}` : "";
  return fetchJson<CycloneObservation[]>(`/observations${queryStr}`);
}

export async function fetchCycloneObservations(stormId: string): Promise<CycloneObservation[]> {
  return fetchJson<CycloneObservation[]>(`/cyclones/${stormId}/observations`);
}

export async function uploadSatelliteImage(formData: FormData): Promise<UploadSatelliteResponse> {
  const url = `${BASE_URL}/data/upload`;
  const res = await fetch(url, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || `Upload failed with status ${res.status}`);
  }
  return (await res.json()) as UploadSatelliteResponse;
}

export async function importHistoricalData(
  file?: File,
  jsonText?: string
): Promise<ImportHistoricalReport> {
  const url = `${BASE_URL}/data/import/historical`;

  if (file) {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(url, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Import failed (${res.status})`);
    }
    return (await res.json()) as ImportHistoricalReport;
  } else if (jsonText) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ format: "json", content: jsonText }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Import failed (${res.status})`);
    }
    return (await res.json()) as ImportHistoricalReport;
  }
  throw new Error("Must provide file or JSON string");
}

// ---------------------------------------------------------------------------
// Phase 4: Satellite AI Image Analysis & Model Registry
// ---------------------------------------------------------------------------
import {
  SatelliteAnalysisResponse,
  ExplainabilityResult,
  ModelRegistryResponse,
} from "@/types/analysis";

export async function analyzeSatelliteImage(
  formData: FormData
): Promise<SatelliteAnalysisResponse> {
  const url = `${BASE_URL}/analysis/satellite`;
  const res = await fetch(url, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || `Analysis failed with status ${res.status}`);
  }
  return (await res.json()) as SatelliteAnalysisResponse;
}

export async function fetchAnalysis(analysisId: string): Promise<SatelliteAnalysisResponse> {
  return fetchJson<SatelliteAnalysisResponse>(`/analysis/${analysisId}`);
}

export async function fetchAnalysisExplanation(analysisId: string): Promise<ExplainabilityResult> {
  return fetchJson<ExplainabilityResult>(`/analysis/${analysisId}/explanation`);
}

export async function fetchModelRegistry(): Promise<ModelRegistryResponse> {
  return fetchJson<ModelRegistryResponse>("/analysis/models");
}

export function getAnalysisOverlayUrl(analysisId: string): string {
  return `${BASE_URL}/analysis/${analysisId}/overlay`;
}

// ---------------------------------------------------------------------------
// Phase 5: Trajectory Forecasting, Physics Vortex & Uncertainty
// ---------------------------------------------------------------------------
import {
  HollandVortexProfile,
  TrajectoryForecastRequest,
} from "@/types/prediction";

export async function executeTrajectoryForecast(
  request: TrajectoryForecastRequest
): Promise<PredictionResponse> {
  const url = `${BASE_URL}/prediction/forecast`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Forecast execution failed (${res.status})`);
  }
  return (await res.json()) as PredictionResponse;
}

export async function fetchCycloneVortexProfile(
  stormId: string
): Promise<HollandVortexProfile> {
  return fetchJson<HollandVortexProfile>(`/prediction/vortex/${stormId}`);
}

export async function fetchPredictionModels(): Promise<any> {
  return fetchJson<any>("/prediction/models");
}

// ---------------------------------------------------------------------------
// Phase 6: AI Impact & Disaster Risk Intelligence
// ---------------------------------------------------------------------------
import {
  CompositeRiskAssessment,
  RiskMapGeoJSON,
  ScenarioRequest,
} from "@/types/impact";

export async function fetchImpact(stormId: string): Promise<CompositeRiskAssessment> {
  return fetchJson<CompositeRiskAssessment>(`/impact/${stormId}`);
}

export async function fetchRiskMapGeoJSON(stormId: string): Promise<RiskMapGeoJSON> {
  return fetchJson<RiskMapGeoJSON>(`/impact/${stormId}/risk-map`);
}

export async function fetchExposure(stormId: string): Promise<any> {
  return fetchJson<any>(`/impact/${stormId}/exposure`);
}

export async function fetchImpactProvenance(stormId: string): Promise<DataProvenance> {
  return fetchJson<DataProvenance>(`/impact/${stormId}/provenance`);
}

export async function runScenarioImpact(
  request: ScenarioRequest
): Promise<CompositeRiskAssessment> {
  const url = `${BASE_URL}/impact/scenario`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Impact scenario evaluation failed (${res.status})`);
  }
  return (await res.json()) as CompositeRiskAssessment;
}

export async function fetchImpactModels(): Promise<any> {
  return fetchJson<any>("/impact/models");
}

// ---------------------------------------------------------------------------
// Phase 8: Real ML Trajectory Forecasting (NOAA IBTrACS Trained LSTM)
// ---------------------------------------------------------------------------
import {
  MLTrajectoryModelInfo,
  MLTrajectoryPredictResponse,
  MLTrajectoryEvaluationResponse,
  MLTrajectoryModelsResponse,
} from "@/types/ml_trajectory";

export async function fetchMLTrajectoryModel(): Promise<MLTrajectoryModelInfo> {
  return fetchJson<MLTrajectoryModelInfo>("/ml/trajectory/model");
}

export async function fetchMLTrajectoryModels(): Promise<MLTrajectoryModelsResponse> {
  return fetchJson<MLTrajectoryModelsResponse>("/ml/trajectory/models");
}

export async function predictMLTrajectory(
  trackHistory: any[],
  forecastHorizons: number[] = [6, 12, 24],
  modelType: "transformer" | "lstm" = "transformer"
): Promise<MLTrajectoryPredictResponse> {
  const url = `${BASE_URL}/ml/trajectory/predict`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      track_history: trackHistory,
      forecast_horizons: forecastHorizons,
      model_type: modelType,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `ML Trajectory prediction failed (${res.status})`);
  }
  return (await res.json()) as MLTrajectoryPredictResponse;
}

export async function fetchMLTrajectoryMetrics(): Promise<MLTrajectoryEvaluationResponse> {
  return fetchJson<MLTrajectoryEvaluationResponse>("/ml/trajectory/metrics");
}

export async function evaluateMLTrajectory(
  forceRerun: boolean = false
): Promise<MLTrajectoryEvaluationResponse> {
  return fetchJson<MLTrajectoryEvaluationResponse>(
    `/ml/trajectory/evaluate?force_rerun=${forceRerun}`
  );
}

// ---------------------------------------------------------------------------
// Phase 10: Scientific End-to-End AI Pipeline Integration
// ---------------------------------------------------------------------------
import {
  PipelineAnalyzeResponse,
  PipelineStatusResponse,
} from "@/types/pipeline";

export async function fetchPipelineStatus(): Promise<PipelineStatusResponse> {
  return fetchJson<PipelineStatusResponse>("/ml/pipeline/status");
}

export async function analyzePipeline(formData: FormData): Promise<PipelineAnalyzeResponse> {
  const url = `${BASE_URL}/ml/pipeline/analyze`;
  const res = await fetch(url, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Pipeline analysis failed (${res.status})`);
  }
  return (await res.json()) as PipelineAnalyzeResponse;
}

export async function uploadSatelliteRaster(file: File): Promise<any> {
  const url = `${BASE_URL}/satellite/upload`;
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(url, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Satellite upload failed (${res.status})`);
  }
  return await res.json();
}

export async function analyzeSatelliteRaster(formData: FormData): Promise<PipelineAnalyzeResponse> {
  const url = `${BASE_URL}/ml/pipeline/analyze-satellite`;
  const res = await fetch(url, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Scientific satellite analysis failed (${res.status})`);
  }
  return (await res.json()) as PipelineAnalyzeResponse;
}

// ---------------------------------------------------------------------------
// Kaggle INSAT-3D Cyclone Detection & Model Comparison APIs
// ---------------------------------------------------------------------------
import {
  KaggleDatasetStatistics,
  DatasetSamplesResponse,
  ModelComparisonResult,
  DetectionEvaluationPayload,
} from "@/types/kaggle_detection";

export async function fetchDetectionDatasetStatistics(): Promise<KaggleDatasetStatistics> {
  return fetchJson<KaggleDatasetStatistics>("/ml/detection/dataset/statistics");
}

export async function fetchDetectionDatasetSamples(params?: {
  split?: string;
  label?: number;
  limit?: number;
  offset?: number;
}): Promise<DatasetSamplesResponse> {
  const q = new URLSearchParams();
  if (params?.split) q.append("split", params.split);
  if (params?.label !== undefined) q.append("label", params.label.toString());
  if (params?.limit) q.append("limit", params.limit.toString());
  if (params?.offset !== undefined) q.append("offset", params.offset.toString());
  const queryStr = q.toString() ? `?${q.toString()}` : "";
  return fetchJson<DatasetSamplesResponse>(`/ml/detection/dataset${queryStr}`);
}

export async function fetchDetectionComparison(): Promise<ModelComparisonResult> {
  return fetchJson<ModelComparisonResult>("/ml/detection/comparison");
}

export async function fetchDetectionMetrics(
  modelVersion: string = "v2"
): Promise<DetectionEvaluationPayload> {
  return fetchJson<DetectionEvaluationPayload>(
    `/ml/detection/metrics?model_version=${modelVersion}`
  );
}

export async function prepareDetectionDataset(): Promise<{
  status: string;
  message: string;
  provenance: any;
}> {
  const url = `${BASE_URL}/ml/detection/dataset/prepare`;
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Dataset preparation failed (${res.status})`);
  }
  return await res.json();
}

// ==============================================================================
// Historical Satellite API (Authentic INSAT-3D Dataset Analysis)
// ==============================================================================

export interface HistoricalImageSummary {
  image_id: string;
  filename: string;
  split: string;
  label: string;
  label_source: string;
  source: string;
  dataset: string;
  satellite: string;
  sensor: string;
  channel: string;
  acquisition_time: string | null;
  storm_name: string | null;
  dataset_reference_wind_knots: number | null;
  intensity_category: string | null;
  convective_cloud_fraction: number | null;
  has_georeference: boolean;
  latitude: number | null;
  longitude: number | null;
  preview_url: string;
}

export interface HistoricalImagesResponse {
  total: number;
  split_filter: string | null;
  label_filter: string | null;
  limit: number;
  images: HistoricalImageSummary[];
  disclaimer: string;
}

export interface HistoricalAnalysisResponse {
  image_id: string;
  data_mode: string;
  source: string;
  dataset: string;
  detection: {
    detected: boolean;
    label: string;
    confidence: number;
    class_probabilities: Record<string, number>;
    structural_type: string;
    center_localization: any;
    cloud_fraction?: number;
  };
  intensity: {
    status: string;
    reason: string;
    estimated_wind_speed_knots: number | null;
    estimated_central_pressure_hpa: number | null;
    dvorak_t_number: number | null;
    model_name: string;
    model_status: string;
    dataset_reference_wind_knots?: number | null;
    dataset_intensity_category?: string | null;
    dataset_note?: string;
  };
  trajectory: {
    status: string;
    message: string;
    available_fixes: number;
    required_fixes: number;
    forecast?: {
      model_name: string;
      model_version: string;
      origin_fix: any;
      predictions: Array<{
        forecast_hour: number;
        forecast_lead_hours: number;
        timestamp: string;
        latitude: number;
        longitude: number;
        uncertainty_km: number;
        uncertainty_radius_km_68: number;
        uncertainty_radius_km_95: number;
      }>;
      uncertainty_method: string;
      forecast_type: string;
      is_experimental: boolean;
    } | null;
  };
  risk: {
    risk_score: number;
    risk_level: string;
    risk_factors: Record<string, any>;
    risk_confidence: number;
    data_completeness: number;
    available_factors_count: number;
    total_factors_count: number;
    explanation: string;
    disclaimer: string;
  };
  provenance: {
    source: string;
    dataset: string;
    image_filename: string;
    image_id: string;
    split: string;
    timestamp: string | null;
    detection_model: string;
    detection_version: string;
    intensity_model: string;
    trajectory_model: string;
    risk_engine: string;
    processing_version: string;
    data_mode: string;
    generated_at: string;
    sha256: string;
  };
  disclaimer: string;
}

export async function fetchHistoricalImages(params?: {
  split?: string;
  label?: string;
  limit?: number;
}): Promise<HistoricalImagesResponse> {
  const q = new URLSearchParams();
  if (params?.split) q.append("split", params.split);
  if (params?.label) q.append("label", params.label);
  if (params?.limit) q.append("limit", params.limit.toString());
  const queryStr = q.toString() ? `?${q.toString()}` : "";
  const endpoint = `/satellite/historical/images${queryStr}`;
  const url = `${BASE_URL}${endpoint}`;
  console.log(`[Historical API] Requesting URL: ${url}`);
  try {
    const res = await fetch(url, {
      headers: {
        "Accept": "application/json",
      },
      cache: "no-store",
    });
    console.log(`[Historical API] HTTP Status: ${res.status} ${res.statusText}`);
    if (!res.ok) {
      throw new Error(`API error ${res.status}: ${res.statusText} on ${url}`);
    }
    const data = (await res.json()) as HistoricalImagesResponse;
    console.log(`[Historical API] Parsed JSON keys: ${Object.keys(data || {}).join(", ")}`);
    console.log(`[Historical API] images.length: ${data?.images?.length ?? 0}, total: ${data?.total ?? 0}`);
    return data;
  } catch (err) {
    console.error(`[Historical API] Fetch failed on ${url}:`, err);
    throw err;
  }
}

export async function fetchHistoricalImageDetails(
  imageId: string
): Promise<HistoricalImageSummary> {
  return fetchJson<HistoricalImageSummary>(`/satellite/historical/images/${imageId}`);
}

export function getHistoricalImagePreviewUrl(imageId: string): string {
  return `${BASE_URL}/satellite/historical/images/${imageId}/preview`;
}

export async function analyzeHistoricalImage(
  imageId: string,
  trackHistory?: any[],
  environmentalObs?: any
): Promise<HistoricalAnalysisResponse> {
  const url = `${BASE_URL}/satellite/historical/analyze`;
  console.log(`[Historical API] Analyzing image: ${imageId} at ${url}`, {
    trackPointsCount: trackHistory?.length ?? 0,
  });
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
      },
      body: JSON.stringify({
        image_id: imageId,
        track_history: trackHistory || null,
        environmental_obs: environmentalObs || null,
      }),
    });
    console.log(`[Historical API] Analysis HTTP Status: ${res.status} ${res.statusText}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Historical analysis failed (${res.status})`);
    }
    const data = (await res.json()) as HistoricalAnalysisResponse;
    console.log(`[Historical API] Analysis response keys: ${Object.keys(data || {}).join(", ")}`);
    console.log(`[Historical API] Detection: ${data?.detection?.label} (confidence: ${data?.detection?.confidence}), Trajectory: ${data?.trajectory?.status}, Risk: ${data?.risk?.risk_score}`);
    return data;
  } catch (err) {
    console.error(`[Historical API] Analysis failed:`, err);
    throw err;
  }
}

// ---------------------------------------------------------------------------
// PDF-Derived Historical Satellite Archive Client API
// ---------------------------------------------------------------------------

export interface PDFSatelliteSample {
  sample_id: string;
  pdf_file: string;
  page_number: number;
  embedded_image_index: number;
  width: number;
  height: number;
  format: string;
  sha256: string;
  provider: string | null;
  satellite: string | null;
  product: string;
  timestamp: string | null;
  storm_name: string | null;
  latitude: number | null;
  longitude: number | null;
  wind_speed_kt: number | null;
  intensity_category: string | null;
  cyclone_center_visible: boolean;
  metadata_source: string;
  metadata_confidence: string;
  label_status: string;
  preview_url: string;
}

export interface PDFSatelliteListResponse {
  total: number;
  limit: number;
  offset: number;
  product_filter: string | null;
  center_marker_filter: boolean | null;
  samples: PDFSatelliteSample[];
  disclaimer: string;
}

export interface PDFSatelliteStatsResponse {
  total_extracted_samples: number;
  source_pdf: string;
  labeled_samples: number;
  unlabeled_samples: number;
  verified_wind_speed_labels: number;
  verified_coordinate_labels: number;
  verified_storm_name_labels: number;
  center_crosshair_marker_count: number;
  product_distribution: Record<string, number>;
  duplicates: {
    exact_sha256_duplicates: number;
    perceptual_dhash_duplicates: number;
    pairs: Array<{
      pair_id: string;
      sample_id_1: string;
      sample_id_2: string;
      page_1: number;
      page_2: number;
      duplicate_type: string;
      hamming_distance: number;
      similarity: number;
      action_recommendation: string;
      notes: string;
    }>;
  };
  ml_safety_rules: {
    supervised_intensity_training_permitted: boolean;
    supervised_intensity_reason: string;
    supervised_track_training_permitted: boolean;
    supervised_track_reason: string;
    cyclone_detection_evaluation_permitted: boolean;
    unsupervised_pretraining_permitted: boolean;
    zero_fabrication_compliance: string;
  };
  disclaimer: string;
}

export interface PDFAnalysisResponse {
  sample_id: string;
  archive_source: string;
  pdf_file: string;
  page_number: number;
  product: string;
  cyclone_center_visible: boolean;
  sha256: string;
  analysis_timestamp: string;
  cyclone_detection: {
    detected: boolean;
    label: string;
    confidence: number;
    model: string;
    class_probabilities: Record<string, number>;
    center_localization: Record<string, any>;
    visual_center_marker_present: boolean;
  };
  intensity_estimation: {
    status: string;
    reason: string;
    estimated_wind_speed_knots: number | null;
    estimated_central_pressure_hpa: number | null;
    dvorak_t_number: string | null;
    model_name: string;
    model_status: string;
    dataset_reference_wind_knots: number | null;
    dataset_intensity_category: string | null;
    dataset_note: string;
  };
  trajectory_prediction: {
    status: string;
    forecast: any | null;
    required_history_length: number;
    provided_history_length: number;
    note: string;
  };
  risk_assessment: {
    overall_risk_score: number;
    risk_level: string;
    components: Record<string, any>;
    summary: string;
    data_completeness: string;
  };
  provenance: {
    image_sha256: string;
    pipeline_version: string;
    label_status: string;
    metadata_confidence: string;
    dataset_provenance: string;
  };
  scientific_disclaimer: string;
}

export async function fetchPDFSatelliteSamples(params?: {
  product?: string;
  centerMarkerOnly?: boolean;
  limit?: number;
  offset?: number;
}): Promise<PDFSatelliteListResponse> {
  const q = new URLSearchParams();
  if (params?.product) q.append("product", params.product);
  if (params?.centerMarkerOnly !== undefined) q.append("center_marker_only", String(params.centerMarkerOnly));
  if (params?.limit) q.append("limit", params.limit.toString());
  if (params?.offset) q.append("offset", params.offset.toString());
  const queryStr = q.toString() ? `?${q.toString()}` : "";
  return fetchJson<PDFSatelliteListResponse>(`/satellite/historical/pdf${queryStr}`);
}

export async function fetchPDFSatelliteStats(): Promise<PDFSatelliteStatsResponse> {
  return fetchJson<PDFSatelliteStatsResponse>("/satellite/historical/pdf/stats");
}

export async function fetchPDFSatelliteSampleDetails(sampleId: string): Promise<PDFSatelliteSample> {
  return fetchJson<PDFSatelliteSample>(`/satellite/historical/pdf/${sampleId}`);
}

export function getPDFSamplePreviewUrl(sampleId: string): string {
  return `${BASE_URL}/satellite/historical/pdf/${sampleId}/preview`;
}

export async function analyzePDFSatelliteSample(
  sampleId: string,
  trackHistory?: any[],
  environmentalObs?: any
): Promise<PDFAnalysisResponse> {
  const url = `${BASE_URL}/satellite/historical/pdf/analyze`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json",
    },
    body: JSON.stringify({
      sample_id: sampleId,
      track_history: trackHistory || null,
      environmental_obs: environmentalObs || null,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `PDF analysis failed (${res.status})`);
  }
  return (await res.json()) as PDFAnalysisResponse;
}

// ==============================================================================
// Best Track Historical Database API (IMD RSMC New Delhi + NOAA IBTrACS)
// ==============================================================================

export interface BestTrackStorm {
  storm_id: string;
  storm_name: string;
  year: number;
  basin: string;
  source: string;
  start_time: string;
  end_time: string;
  point_count: number;
  max_wind_kt: number | null;
  min_pressure_hpa: number | null;
  cyclone_category: string | null;
}

export interface BestTrackFix {
  id: string;
  storm_id: string;
  storm_name: string;
  basin: string;
  subbasin: string | null;
  source: string;
  source_agency: string;
  source_record_id: string | null;
  timestamp_utc: string;
  latitude: number;
  longitude: number;
  max_wind_kt: number | null;
  central_pressure_hpa: number | null;
  cyclone_category: string | null;
  ci_number: number | null;
  pressure_drop_hpa: number | null;
  grade: string | null;
  is_best_track: boolean;
  source_file: string;
  source_url: string | null;
  retrieved_at_utc: string;
  data_quality: string;
  validation_status: string;
  cross_source_difference: boolean;
}

export interface BestTrackTrackResponse {
  storm: BestTrackStorm;
  fixes_count: number;
  fixes: BestTrackFix[];
  geojson: {
    type: string;
    properties: Record<string, any>;
    geometry: {
      type: string;
      coordinates: number[][];
    };
  };
}

export interface BestTrackListResponse {
  total: number;
  limit: number;
  offset: number;
  storms: BestTrackStorm[];
}

export interface BestTrackStatsResponse {
  total_storms: number;
  total_fixes: number;
  year_range: { min_year: number; max_year: number };
  source_counts: Record<string, number>;
  basin_counts: Record<string, number>;
  named_storms_count: number;
}

export interface TrackAssociationResponse {
  status: "VERIFIED_MATCH" | "POSSIBLE_MATCH" | "UNVERIFIED" | "NO_MATCH";
  satellite_timestamp_utc: string | null;
  satellite_coordinates: { latitude: number | null; longitude: number | null } | null;
  time_difference_hours: number | null;
  spatial_distance_km: number | null;
  nearest_fix: BestTrackFix | null;
  candidate_fixes: Array<{
    fix_id: string;
    timestamp: string;
    latitude: number;
    longitude: number;
    time_diff_hours: number;
    dist_km: number;
  }>;
  match_method: string;
  details: string;
}

export interface ForecastEvaluationResponse {
  storm_id: string;
  storm_name: string;
  evaluation_time_utc: string;
  history_fixes_count: number;
  future_fixes_count: number;
  model_name: string;
  horizons_evaluated: number[];
  metrics: Record<string, {
    mae_km?: number | null;
    rmse_km?: number | null;
    target_time?: string;
    actual_time?: string;
    actual_lat?: number;
    actual_lon?: number;
    note?: string;
  }>;
  overall_mae_km: number | null;
  overall_rmse_km: number | null;
  input_history: BestTrackFix[];
  predicted_positions: Array<{
    lead_hours: number;
    timestamp: string;
    latitude: number;
    longitude: number;
    uncertainty_km: number;
  }>;
  actual_observed_positions: Array<{
    lead_hours: number;
    timestamp: string;
    latitude: number;
    longitude: number;
    actual_wind_kt?: number | null;
  }>;
  disclaimer: string;
}

export async function fetchBestTrackStorms(params?: {
  source?: string;
  year?: number;
  basin?: string;
  search?: string;
  limit?: number;
  offset?: number;
}): Promise<BestTrackListResponse> {
  const q = new URLSearchParams();
  if (params?.source) q.append("source", params.source);
  if (params?.year) q.append("year", params.year.toString());
  if (params?.basin) q.append("basin", params.basin);
  if (params?.search) q.append("search", params.search);
  if (params?.limit) q.append("limit", params.limit.toString());
  if (params?.offset) q.append("offset", params.offset.toString());
  const queryStr = q.toString() ? `?${q.toString()}` : "";
  return fetchJson<BestTrackListResponse>(`/best-track/storms${queryStr}`);
}

export async function fetchBestTrackStorm(stormId: string): Promise<BestTrackStorm> {
  return fetchJson<BestTrackStorm>(`/best-track/storms/${encodeURIComponent(stormId)}`);
}

export async function fetchBestTrack(stormId: string): Promise<BestTrackTrackResponse> {
  return fetchJson<BestTrackTrackResponse>(`/best-track/storms/${encodeURIComponent(stormId)}/track`);
}

export async function fetchBestTrackStats(): Promise<BestTrackStatsResponse> {
  return fetchJson<BestTrackStatsResponse>("/best-track/stats");
}

export async function matchImageToTrack(payload: {
  storm_id: string;
  image_timestamp_utc?: string;
  image_latitude?: number;
  image_longitude?: number;
}): Promise<TrackAssociationResponse> {
  const url = `${BASE_URL}/best-track/match-image`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Match image failed (${res.status})`);
  }
  return await res.json();
}

export async function evaluateBestTrackForecast(payload: {
  storm_id: string;
  lead_steps?: number;
}): Promise<ForecastEvaluationResponse> {
  const url = `${BASE_URL}/best-track/evaluate-forecast`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Forecast evaluation failed (${res.status})`);
  }
  return await res.json();
}

export async function analyzeOceanRisk(payload: {
  latitude: number;
  longitude: number;
  radius_km?: number;
}): Promise<OceanRiskResponse> {
  const url = `${BASE_URL}/prediction/ocean-risk`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Ocean risk evaluation failed (${res.status})`);
  }
  return await res.json();
}

export async function fetchOceanRiskStatus(): Promise<OceanRiskStatusResponse> {
  return fetchJson<OceanRiskStatusResponse>("/prediction/ocean-risk/status");
}

export async function fetchPerformanceSummary(): Promise<any> {
  return fetchJson<any>("/ml/performance/summary");
}

export async function fetchMosdacTelemetryStatus(): Promise<MosdacTelemetryStatus> {
  return fetchJson<MosdacTelemetryStatus>("/v2/telemetry/mosdac/status");
}

export async function triggerMosdacPoll(): Promise<MosdacTelemetryStatus> {
  const url = `${BASE_URL}/v2/telemetry/mosdac/poll`;
  const res = await fetch(url, {
    method: "POST",
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(`Failed to trigger MOSDAC poll: ${res.status}`);
  }
  return await res.json();
}







export async function fetchImergTelemetryStatus(): Promise<ImergTelemetryStatus> {
  return fetchJson<ImergTelemetryStatus>("/v2/telemetry/imerg/status");
}

export async function triggerImergPoll(): Promise<ImergTelemetryStatus> {
  const url = `${BASE_URL}/v2/telemetry/imerg/poll`;
  const res = await fetch(url, {
    method: "POST",
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(`Failed to trigger IMERG poll: ${res.status}`);
  }
  return await res.json();
}

export async function fetchEnvironmentTelemetryStatus(): Promise<EnvironmentTelemetryStatus> {
  return fetchJson<EnvironmentTelemetryStatus>("/v2/telemetry/environment/status");
}

export async function triggerEnvironmentPoll(): Promise<EnvironmentTelemetryStatus> {
  const url = `${BASE_URL}/v2/telemetry/environment/poll`;
  const res = await fetch(url, {
    method: "POST",
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(`Failed to trigger Environment poll: ${res.status}`);
  }
  return await res.json();
}

