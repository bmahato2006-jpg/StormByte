from datetime import datetime, timezone
import uuid
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class UserRecord(Base):
    __tablename__ = "users"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(128), unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    role = Column(String(32), nullable=False, default="viewer", index=True)
    full_name = Column(String(128), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)


class Storm(Base):
    __tablename__ = "storms"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    code = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(128), nullable=False, index=True)
    basin = Column(String(64), nullable=False, default="North Indian Ocean", index=True)
    status = Column(String(32), nullable=False, default="ACTIVE", index=True)
    genesis_time = Column(DateTime(timezone=True), nullable=True)
    dissipation_time = Column(DateTime(timezone=True), nullable=True)
    metadata_json = Column(JSON, default=dict)

    observations = relationship("Observation", back_populates="storm", cascade="all, delete-orphan", lazy="selectin")
    forecast_runs = relationship("ForecastRun", back_populates="storm", cascade="all, delete-orphan", lazy="selectin")
    risk_records = relationship("RiskRecord", back_populates="storm", cascade="all, delete-orphan", lazy="selectin")

    __table_args__ = (
        Index("idx_storm_basin_status", "basin", "status"),
        Index("idx_storm_code_status", "code", "status"),
    )


class Observation(Base):
    __tablename__ = "observations"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    storm_id = Column(String(64), ForeignKey("storms.id", ondelete="CASCADE"), nullable=False, index=True)
    observation_time = Column(DateTime(timezone=True), nullable=False, index=True)

    # Coordinates with geodetic index
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)

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

    __table_args__ = (
        Index("idx_obs_storm_time", "storm_id", "observation_time"),
        Index("idx_obs_geo_time", "latitude", "longitude", "observation_time"),
    )


class ForecastRun(Base):
    __tablename__ = "forecast_runs"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    storm_id = Column(String(64), ForeignKey("storms.id", ondelete="CASCADE"), nullable=False, index=True)
    run_time = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    model_name = Column(String(128), nullable=False)
    model_version = Column(String(64), nullable=False)
    run_mode = Column(String(32), default="DEMO")

    forecast_points = relationship("ForecastPoint", back_populates="run", cascade="all, delete-orphan", lazy="selectin")
    storm = relationship("Storm", back_populates="forecast_runs")

    __table_args__ = (
        Index("idx_forecast_storm_runtime", "storm_id", "run_time"),
    )


class ForecastPoint(Base):
    __tablename__ = "forecast_points"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String(64), ForeignKey("forecast_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    forecast_lead_hours = Column(Integer, nullable=False)
    valid_time = Column(DateTime(timezone=True), nullable=False, index=True)

    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    predicted_wind_knots = Column(Float, nullable=True)
    predicted_pressure_hpa = Column(Float, nullable=True)

    # Uncertainty envelope
    semi_major_km = Column(Float, nullable=False)
    semi_minor_km = Column(Float, nullable=False)
    ellipse_azimuth_deg = Column(Float, default=0.0)

    run = relationship("ForecastRun", back_populates="forecast_points")

    __table_args__ = (
        Index("idx_fpoint_run_lead", "run_id", "forecast_lead_hours"),
    )


class RiskRecord(Base):
    __tablename__ = "risk_records"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    storm_id = Column(String(64), ForeignKey("storms.id", ondelete="CASCADE"), nullable=False, index=True)
    assessment_time = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    risk_level = Column(String(32), nullable=False, index=True)
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
    user_id = Column(String(64), nullable=True, index=True)
    storm_id = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    data_source = Column(String(128), nullable=False)
    data_type = Column(String(64), nullable=False)
    is_synthetic = Column(Boolean, default=False)
    operator_or_system = Column(String(64), default="system")
    details = Column(JSON, default=dict)
