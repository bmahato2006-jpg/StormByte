"""SQLAlchemy Database Models for Cyclone-X Scientific Data Intelligence.

Prepared for PostgreSQL and PostGIS extensions, fully operational with SQLite
or in-memory mock persistence for zero-crash fallback.
"""

from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, Float, DateTime, Text
from backend.app.models.storm import Base


class CycloneObservationModel(Base):
    """Normalized storage model for tropical cyclone center and intensity observations."""
    __tablename__ = "cyclone_observations"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    storm_id = Column(String(64), nullable=False, index=True)
    storm_name = Column(String(128), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    wind_speed = Column(Float, nullable=False)
    central_pressure = Column(Float, nullable=False)
    classification = Column(String(64), nullable=False)
    source = Column(String(64), nullable=False)
    source_reference = Column(String(256), nullable=True)
    data_mode = Column(String(32), nullable=False, default="DEMO")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class SatelliteObservationModel(Base):
    """Normalized storage model for satellite scans and granule metadata."""
    __tablename__ = "satellite_observations"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    satellite = Column(String(64), nullable=False)
    sensor = Column(String(64), nullable=False)
    product = Column(String(64), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    file_path = Column(String(512), nullable=False)
    source_reference = Column(String(256), nullable=True)
    latitude_min = Column(Float, nullable=False)
    latitude_max = Column(Float, nullable=False)
    longitude_min = Column(Float, nullable=False)
    longitude_max = Column(Float, nullable=False)
    resolution = Column(String(32), nullable=True, default="4km")
    processing_status = Column(String(32), nullable=False, default="RAW")
    data_mode = Column(String(32), nullable=False, default="DEMO")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class EnvironmentalObservationModel(Base):
    """Normalized storage model for ambient oceanic and meteorological conditions."""
    __tablename__ = "environmental_observations"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    sea_surface_temperature = Column(Float, nullable=True)
    rainfall = Column(Float, nullable=True)
    wind_speed = Column(Float, nullable=True)
    wind_direction = Column(Float, nullable=True)
    pressure = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    source = Column(String(64), nullable=False)
    data_mode = Column(String(32), nullable=False, default="DEMO")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class DataProvenanceModel(Base):
    """Audit table recording provenance metadata for ingested datasets."""
    __tablename__ = "data_provenance"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    source = Column(String(64), nullable=False)
    provider = Column(String(128), nullable=False)
    product = Column(String(128), nullable=False)
    dataset_version = Column(String(32), nullable=False, default="1.0")
    acquisition_time = Column(DateTime(timezone=True), nullable=True)
    processing_time = Column(DateTime(timezone=True), nullable=True)
    source_reference = Column(String(512), nullable=True)
    license_notes = Column(Text, nullable=True)
    data_mode = Column(String(32), nullable=False)
    status = Column(String(32), nullable=False)

