export type TrajectoryModelType = "transformer" | "lstm";

export interface MLTrajectoryPoint {
  forecast_lead_hours: number;
  forecast_time_label: string;
  valid_time: string;
  latitude: number;
  longitude: number;
  uncertainty_radius_km_68: number;
  uncertainty_radius_km_95: number;
  model_name: string;
  is_experimental: boolean;
}

export interface MLTrajectoryPredictResponse {
  model_name: string;
  model_version: string;
  architecture: string;
  status: string;
  is_experimental: boolean;
  origin_fix: {
    latitude: number;
    longitude: number;
    timestamp: string;
  };
  predictions: MLTrajectoryPoint[];
  uncertainty_method: string;
  checkpoint_metadata: Record<string, any>;
  disclaimer: string;
}

export interface HorizonComparisonMetric {
  horizon: string;
  ml_mae_km: number;
  ml_rmse_km: number;
  ml_median_km: number;
  ml_p90_km: number;
  physics_mae_km: number;
  physics_rmse_km: number;
  physics_median_km: number;
  physics_p90_km: number;
  difference_km: number;
  relative_diff_pct: number;
  better_model: string;
  transformer_mae_km?: number;
  lstm_mae_km?: number;
  transformer_rmse_km?: number;
  lstm_rmse_km?: number;
}

export interface StormEvaluationBreakdown {
  storm_id: string;
  storm_name: string;
  test_samples: number;
  ml_mae_km: number;
  physics_mae_km: number;
  by_horizon: Record<string, { ml_mae_km?: number; physics_mae_km: number; transformer_mae_km?: number; lstm_mae_km?: number }>;
  transformer_mae_km?: number;
  lstm_mae_km?: number;
}

export interface MLTrajectoryEvaluationResponse {
  evaluation_id: string;
  evaluated_at: string;
  test_dataset: {
    manifest_path: string;
    manifest_sha256: string;
    num_test_sequences: number;
    num_test_storms: number;
    horizons_evaluated: number[];
  };
  models_evaluated: Array<{
    name: string;
    type: string;
    version?: string;
    checkpoint?: string;
    checkpoint_epoch?: number;
    training_provenance?: Record<string, any>;
  }>;
  comparison_table: HorizonComparisonMetric[];
  ml_model_metrics: {
    model_name: string;
    overall: Record<string, number>;
    by_horizon: Record<string, Record<string, number>>;
  };
  transformer_model_metrics?: {
    model_name: string;
    overall: Record<string, number>;
    by_horizon: Record<string, Record<string, number>>;
  };
  lstm_model_metrics?: {
    model_name: string;
    overall: Record<string, number>;
    by_horizon: Record<string, Record<string, number>>;
  };
  physics_model_metrics: {
    model_name: string;
    overall: Record<string, number>;
    by_horizon: Record<string, Record<string, number>>;
  };
  per_storm_breakdown: StormEvaluationBreakdown[];
  disclaimer: string;
}

export interface MLTrajectoryModelInfo {
  model_name: string;
  version: string;
  architecture: string;
  framework: string;
  is_trained: boolean;
  status: string;
  is_experimental: boolean;
  input_features: string[];
  input_dim: number;
  d_model?: number;
  hidden_dim?: number;
  num_layers: number;
  output_dim: number;
  forecast_horizons_hours: number[];
  checkpoint_path: string;
  scaler_path: string;
  checkpoint_metadata: Record<string, any>;
  manifests: {
    train: string;
    validation: string;
    test: string;
  };
  disclaimer: string;
}

export interface MLTrajectoryModelsResponse {
  models: Array<{
    model_id: string;
    name: string;
    architecture: string;
    framework: string;
    is_trained: boolean;
    is_primary: boolean;
    status: string;
    hyperparameters: Record<string, any>;
    checkpoint_path: string;
    metadata_path?: string;
  }>;
  active_primary_model: string;
  input_features: string[];
  forecast_horizons_hours: number[];
  scaler_path: string;
}
