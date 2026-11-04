export interface DataProvenance {
  source: string;
  timestamp: string;
  processingVersion: string;
  modelVersion: string;
  dataType: string; // e.g. "DEMO DATA" | "REAL DATA" | "AI_PREDICTION"
  disclaimer: string;
  parentHash?: string;
  dataHash?: string;
}

