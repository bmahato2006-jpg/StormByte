from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class DataProvenance(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source: str = Field(..., description="Origin of the data stream or sensor array")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of ingestion or observation")
    processingVersion: str = Field(..., description="Software pipeline version applied", validation_alias="processing_version")
    modelVersion: str = Field(..., description="Model version used for inference, or status", validation_alias="model_version")
    dataType: str = Field(..., description="Classification: DEMO DATA, REAL DATA, or AI_PREDICTION", validation_alias="data_type")
    disclaimer: str = Field(..., description="Mandatory non-replacement advisory")
    dataHash: Optional[str] = Field(None, description="SHA-256 hash checksum of the payload", validation_alias="data_hash")
    parentHash: Optional[str] = Field(None, description="Parent observation dataHash for provenance lineage", validation_alias="parent_hash")

