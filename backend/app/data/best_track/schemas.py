"""Pydantic schemas for Best Track data structures."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BestTrackFix(BaseModel):
    """Canonical representation of a single historical cyclone position fix."""
    id: str = Field(..., description="Unique record identifier: {source}-{storm_id}-{timestamp}")
    storm_id: str = Field(..., description="Canonical storm identifier, e.g., 2020-AMPHAN")
    storm_name: str = Field(..., description="Uppercase storm name, e.g., AMPHAN")
    basin: str = Field("NI", description="Ocean basin: NI (North Indian Ocean), BOB, or AS")
    subbasin: Optional[str] = Field(None, description="Sub-basin code: BB (Bay of Bengal) or AS (Arabian Sea)")
    source: str = Field(..., description="Data source: IMD_RSMC_NEW_DELHI or NOAA_IBTRACS")
    source_agency: str = Field(..., description="Reporting agency: IMD, WMO, USA/JTWC, etc.")
    source_record_id: Optional[str] = Field(None, description="Original record ID or row index")
    timestamp_utc: datetime = Field(..., description="Timezone-aware UTC timestamp of observation")
    latitude: float = Field(..., description="Latitude in decimal degrees [-90.0, 90.0]")
    longitude: float = Field(..., description="Longitude in decimal degrees [-180.0, 180.0]")
    max_wind_kt: Optional[float] = Field(None, description="Maximum sustained wind in knots (null if unavailable)")
    central_pressure_hpa: Optional[float] = Field(None, description="Estimated central pressure in hPa (null if unavailable)")
    cyclone_category: Optional[str] = Field(None, description="Canonical category: Depression, CS, VSCS, SuCS, etc.")
    ci_number: Optional[float] = Field(None, description="Dvorak Current Intensity / T-number")
    pressure_drop_hpa: Optional[float] = Field(None, description="Central pressure drop in hPa")
    grade: Optional[str] = Field(None, description="Original source classification grade")
    is_best_track: bool = Field(True, description="Flag indicating post-season verified best track")
    source_file: str = Field(..., description="Local or remote source filename")
    source_url: Optional[str] = Field(None, description="Origin URL of source data")
    retrieved_at_utc: datetime = Field(..., description="UTC timestamp when data was ingested")
    data_quality: str = Field("VERIFIED_OFFICIAL", description="Data quality tier")
    validation_status: str = Field("PASSED", description="Validation status")
    cross_source_difference: bool = Field(False, description="True if cross-source observation differs significantly")


class StormIndexRecord(BaseModel):
    """Summary record for a single historical tropical cyclone."""
    storm_id: str
    storm_name: str
    year: int
    basin: str
    source: str
    start_time: str
    end_time: str
    point_count: int
    max_wind_kt: Optional[float] = None
    min_pressure_hpa: Optional[float] = None
    cyclone_category: Optional[str] = None


class TrackAssociationResult(BaseModel):
    """Association assessment between a satellite image and a cyclone track fix."""
    status: str = Field(..., description="VERIFIED_MATCH, POSSIBLE_MATCH, UNVERIFIED, or NO_MATCH")
    satellite_timestamp_utc: Optional[str] = None
    satellite_coordinates: Optional[Dict[str, Optional[float]]] = None
    time_difference_hours: Optional[float] = None
    spatial_distance_km: Optional[float] = None
    nearest_fix: Optional[Dict[str, Any]] = None
    candidate_fixes: List[Dict[str, Any]] = Field(default_factory=list)
    match_method: str = "SPATIOTEMPORAL_PROXIMITY"
    details: str


class ForecastEvaluationResult(BaseModel):
    """Offline historical backtest evaluation result comparing LSTM vs real fixes."""
    storm_id: str
    storm_name: str
    evaluation_time_utc: str
    history_fixes_count: int
    future_fixes_count: int
    model_name: str = "trajectory_lstm_v1"
    horizons_evaluated: List[int] = Field(default_factory=lambda: [6, 12, 24])
    metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    overall_mae_km: Optional[float] = None
    overall_rmse_km: Optional[float] = None
    input_history: List[Dict[str, Any]] = Field(default_factory=list)
    predicted_positions: List[Dict[str, Any]] = Field(default_factory=list)
    actual_observed_positions: List[Dict[str, Any]] = Field(default_factory=list)
    disclaimer: str = (
        "HISTORICAL BACKTEST EVALUATION: Experimental ML model benchmarked against verified "
        "historical best-track fixes. Zero future observations were exposed during inference."
    )


class BestTrackStats(BaseModel):
    """Aggregate statistics for the historical best track catalog."""
    total_storms: int
    total_fixes: int
    year_range: Dict[str, int]
    source_counts: Dict[str, int]
    basin_counts: Dict[str, int]
    named_storms_count: int
