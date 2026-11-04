from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, Float, DateTime, Boolean, JSON, Integer, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Storm(Base):
    __tablename__ = "storms"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    code = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(128), nullable=False)
    basin = Column(String(64), nullable=False, default="North Indian Ocean")
    status = Column(String(32), nullable=False, default="ACTIVE")
    genesis_time = Column(DateTime(timezone=True), nullable=True)
    dissipation_time = Column(DateTime(timezone=True), nullable=True)
    metadata_json = Column(JSON, default=dict)

    observations = relationship("Observation", back_populates="storm", cascade="all, delete-orphan")
    forecast_runs = relationship("ForecastRun", back_populates="storm", cascade="all, delete-orphan")
    risk_records = relationship("RiskRecord", back_populates="storm", cascade="all, delete-orphan")


class Observation(Base):
    __tablename__ = "observations"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    storm_id = Column(String(64), ForeignKey("storms.id"), nullable=False, index=True)
    observation_time = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Geodetic coordinates (Spatial index ready via PostGIS POINT(lon, lat))
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    
    central_pressure_hpa = Column(Float, nullable=False)
    max_sustained_wind_knots = Column(Float, nullable=False)
    intensity_category = Column(String(64), nullable=False)
    radius_max_wind_km = Column(Float, nullable=True)
    
    movement_direction = Column(String(16), nullable=True)
    movement_speed_kmh = Column(Float, nullable=True)
    
    source_type = Column(String(32), nullable=False, default="DEMO")
    data_source = Column(String(128), nullable=False)
    is_synthetic = Column(Boolean, default=False)
    
    storm = relationship("Storm", back_populates="observations")


class ForecastRun(Base):
    __tablename__ = "forecast_runs"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    storm_id = Column(String(64), ForeignKey("storms.id"), nullable=False, index=True)
    run_time = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    model_name = Column(String(128), nullable=False)
    model_version = Column(String(64), nullable=False)
    run_mode = Column(String(32), default="DEMO")
    
    forecast_points = relationship("ForecastPoint", back_populates="run", cascade="all, delete-orphan")
    storm = relationship("Storm", back_populates="forecast_runs")


class ForecastPoint(Base):
    __tablename__ = "forecast_points"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String(64), ForeignKey("forecast_runs.id"), nullable=False, index=True)
    forecast_lead_hours = Column(Integer, nullable=False)  # 6, 12, 24
    valid_time = Column(DateTime(timezone=True), nullable=False)
    
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    predicted_wind_knots = Column(Float, nullable=True)
    predicted_pressure_hpa = Column(Float, nullable=True)
    
    # Uncertainty envelope parameters
    semi_major_km = Column(Float, nullable=False)
    semi_minor_km = Column(Float, nullable=False)
    ellipse_azimuth_deg = Column(Float, default=0.0)
    
    run = relationship("ForecastRun", back_populates="forecast_points")


class RiskRecord(Base):
    __tablename__ = "risk_records"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    storm_id = Column(String(64), ForeignKey("storms.id"), nullable=False, index=True)
    assessment_time = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    risk_level = Column(String(32), nullable=False)
    wind_risk = Column(String(32), nullable=False)
    rainfall_risk = Column(String(32), nullable=False)
    coastal_risk = Column(String(32), nullable=False)
    exposure_risk = Column(String(32), nullable=False)
    surge_estimate_m = Column(Float, nullable=True)
    population_at_risk = Column(Integer, nullable=True)
    
    storm = relationship("Storm", back_populates="risk_records")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    storm_id = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    data_source = Column(String(128), nullable=False)
    data_type = Column(String(64), nullable=False)
    is_synthetic = Column(Boolean, default=False)
    operator_or_system = Column(String(64), default="system")
    details = Column(JSON, default=dict)

