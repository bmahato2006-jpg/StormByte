export interface KaggleDatasetStatistics {
  dataset_name: string;
  kaggle_slug: string;
  total_samples: number;
  cyclone_samples: number;
  no_cyclone_samples: number;
  splits: {
    train_samples: number;
    val_samples: number;
    test_samples: number;
  };
  intensity_distribution_knots: {
    min: number;
    max: number;
  };
  image_specs: {
    dimensions: string;
    format: string;
    channels: number;
  };
  sources: Array<{
    name: string;
    samples: number;
    type: string;
  }>;
  integrity_notice: string;
}

export interface DatasetSampleRecord {
  image_id: string;
  file_path: string;
  label: number;
  label_name: string;
  intensity_knots: number | string | null;
  source_dataset: string;
  split?: string;
}

export interface DatasetSamplesResponse {
  total: number;
  offset: number;
  limit: number;
  samples: DatasetSampleRecord[];
}

export interface ConfusionMatrix {
  true_positives: number;
  true_negatives: number;
  false_positives: number;
  false_negatives: number;
}

export interface DetectionModelMetrics {
  accuracy: number;
  accuracy_percentage: number;
  precision: number;
  recall: number;
  specificity: number;
  f1_score: number;
  roc_auc?: number | null;
  false_positive_rate: number;
  false_negative_rate: number;
}

export interface DetectionEvaluationPayload {
  evaluation_id: string;
  evaluated_at: string;
  model_info: {
    name: string;
    checkpoint: string;
    trained_epoch?: number;
    train_loss?: number;
    best_val_loss?: number;
    training_provenance?: any;
  };
  test_dataset: {
    manifest_path: string;
    manifest_sha256: string;
    total_test_samples: number;
    cyclone_samples: number;
    no_cyclone_samples: number;
  };
  metrics: DetectionModelMetrics;
  confusion_matrix: ConfusionMatrix;
  sample_details?: any[];
  disclaimer: string;
}

export interface ModelComparisonResult {
  v1: DetectionEvaluationPayload;
  v2: DetectionEvaluationPayload;
  delta: {
    accuracy_delta: number;
    precision_delta: number;
    recall_delta: number;
    f1_delta: number;
    specificity_delta: number;
    test_sample_delta: number;
  };
  report_markdown: string;
  comparison_summary: string;
}

