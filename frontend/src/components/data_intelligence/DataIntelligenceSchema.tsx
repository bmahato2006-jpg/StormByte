"use client";

// Re-export DataIntelligenceView as DataIntelligenceSchema for full API and component compatibility
export { DataIntelligenceView as DataIntelligenceSchema, DataIntelligenceView } from "./DataIntelligenceView";
export type { DataIntelligenceTabId } from "./DataIntelligenceView";
export default function DataIntelligenceSchemaDefault() {
  const { DataIntelligenceView } = require("./DataIntelligenceView");
  return <DataIntelligenceView />;
}
