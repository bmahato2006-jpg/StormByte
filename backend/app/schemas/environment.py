from pydantic import BaseModel, Field
from typing import Optional
from backend.app.schemas.provenance import DataProvenance

class EnvironmentalConditions(BaseModel):
    cycloneId: str
    seaSurfaceTemperature: Optional[float] = Field(None, description="Sea Surface Temperature in Celsius (°C)")
    verticalWindShear: Optional[float] = Field(None, description="850-200 hPa Deep Layer Vertical Wind Shear in knots")
    meanSeaLevelPressure: Optional[float] = Field(None, description="Peripheral environmental sea level pressure in hPa")
    relativeHumidityMidTroposphere: Optional[float] = Field(None, description="700-500 hPa Relative Humidity (%)")
    outflowTemperature: Optional[float] = Field(None, description="200 hPa Cloud-top outflow temperature in Celsius (°C)")
    oceanHeatContent: Optional[float] = Field(None, description="Tropical Cyclone Heat Potential in kJ/cm²")
    timestamp: str
    provenance: DataProvenance

