-- CYCLONE-X PostGIS Initialization Script
-- Enables PostGIS spatial extensions and creates indexed spatial tables

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Storms Table
CREATE TABLE IF NOT EXISTS storms (
    id VARCHAR(64) PRIMARY KEY,
    code VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(128) NOT NULL,
    basin VARCHAR(64) NOT NULL DEFAULT 'North Indian Ocean',
    status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
    genesis_time TIMESTAMP WITH TIME ZONE,
    dissipation_time TIMESTAMP WITH TIME ZONE,
    metadata_json JSONB DEFAULT '{}'::jsonb
);

-- Observations with PostGIS GEOMETRY POINT
CREATE TABLE IF NOT EXISTS observations (
    id VARCHAR(64) PRIMARY KEY,
    storm_id VARCHAR(64) REFERENCES storms(id) ON DELETE CASCADE,
    observation_time TIMESTAMP WITH TIME ZONE NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    geom GEOMETRY(Point, 4326),
    central_pressure_hpa DOUBLE PRECISION NOT NULL,
    max_sustained_wind_knots DOUBLE PRECISION NOT NULL,
    intensity_category VARCHAR(64) NOT NULL,
    radius_max_wind_km DOUBLE PRECISION,
    movement_direction VARCHAR(16),
    movement_speed_kmh DOUBLE PRECISION,
    source_type VARCHAR(32) NOT NULL DEFAULT 'DEMO',
    data_source VARCHAR(128) NOT NULL,
    is_synthetic BOOLEAN DEFAULT FALSE
);

-- Auto-populate geom column from lat/lon via trigger or insert
CREATE INDEX IF NOT EXISTS idx_observations_geom ON observations USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_observations_time ON observations(observation_time);

-- Forecast Runs and Points with Uncertainty Polygons
CREATE TABLE IF NOT EXISTS forecast_runs (
    id VARCHAR(64) PRIMARY KEY,
    storm_id VARCHAR(64) REFERENCES storms(id) ON DELETE CASCADE,
    run_time TIMESTAMP WITH TIME ZONE NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    run_mode VARCHAR(32) DEFAULT 'DEMO'
);

CREATE TABLE IF NOT EXISTS forecast_points (
    id VARCHAR(64) PRIMARY KEY,
    run_id VARCHAR(64) REFERENCES forecast_runs(id) ON DELETE CASCADE,
    forecast_lead_hours INT NOT NULL,
    valid_time TIMESTAMP WITH TIME ZONE NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    geom GEOMETRY(Point, 4326),
    cone_polygon GEOMETRY(Polygon, 4326),
    predicted_wind_knots DOUBLE PRECISION,
    predicted_pressure_hpa DOUBLE PRECISION,
    semi_major_km DOUBLE PRECISION NOT NULL,
    semi_minor_km DOUBLE PRECISION NOT NULL,
    ellipse_azimuth_deg DOUBLE PRECISION DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_forecast_points_geom ON forecast_points USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_forecast_cone ON forecast_points USING GIST(cone_polygon);

-- Risk Assessments
CREATE TABLE IF NOT EXISTS risk_records (
    id VARCHAR(64) PRIMARY KEY,
    storm_id VARCHAR(64) REFERENCES storms(id) ON DELETE CASCADE,
    assessment_time TIMESTAMP WITH TIME ZONE NOT NULL,
    risk_level VARCHAR(32) NOT NULL,
    wind_risk VARCHAR(32) NOT NULL,
    rainfall_risk VARCHAR(32) NOT NULL,
    coastal_risk VARCHAR(32) NOT NULL,
    exposure_risk VARCHAR(32) NOT NULL,
    surge_estimate_m DOUBLE PRECISION,
    population_at_risk INT
);

-- Audit & Provenance Logs
CREATE TABLE IF NOT EXISTS audit_logs (
    id VARCHAR(64) PRIMARY KEY,
    storm_id VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    data_source VARCHAR(128) NOT NULL,
    data_type VARCHAR(64) NOT NULL,
    is_synthetic BOOLEAN DEFAULT FALSE,
    operator_or_system VARCHAR(64) DEFAULT 'system',
    details JSONB DEFAULT '{}'::jsonb
);

