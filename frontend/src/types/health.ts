export interface HealthStatus {
  status: string;
  service: string;
  version: string;
}

export interface MLModelDetail {
  available: boolean;
  status: string;
  confidence: string;
}

export interface MLSystemStatus {
  status: string;
  phase: string;
  models: Record<string, MLModelDetail>;
  accuracyNotice: string;
}

