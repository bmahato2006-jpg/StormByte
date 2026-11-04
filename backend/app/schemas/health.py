from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    service: str = Field(..., json_schema_extra={"example": "cyclone-x"})
    version: str = Field(..., json_schema_extra={"example": "0.1.0"})

