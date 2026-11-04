# CYCLONE-X: Scientific Data API Schemas & Reference

## Base URL
`http://127.0.0.1:8000/api`

---

## 1. Data Intelligence & Status Endpoints

### `GET /data/status`
Returns database health, active data mode, observation counters, and external adapter availability.

**Response `200 OK`:**
```json
{
  "active_data_mode": "DEMO",
  "database_status": "STANDALONE_IN_MEMORY_FALLBACK (POSTGRESQL OFFLINE)",
  "is_database_connected": false,
  "total_cyclone_observations": 11,
  "total_satellite_observations": 1,
  "total_environmental_observations": 1,
  "total_provenance_records": 4,
  "sources": [ ... ],
  "scientific_disclaimer": "EXPERIMENTAL RESEARCH SYSTEM - NOT AN OFFICIAL WEATHER FORECAST."
}
```

### `GET /data/sources`
Returns catalog of meteorological data sources and their credential status.

**Response `200 OK`:**
```json
[
  {
    "name": "MOSDAC (INSAT-3D/3DR/3DS)",
    "provider": "Space Applications Centre (ISRO)",
    "description": "Meteorological Satellite Imager & Sounder feeds...",
    "supported_products": ["L1B_TIR1", "L1B_WV", "L2B_HEM"],
    "status": "NOT_CONFIGURED",
    "data_mode": "DEMO",
    "configured": false,
    "requires_auth": true,
    "last_checked": "2026-09-10T21:24:01Z",
    "notes": "Requires valid ISRO MOSDAC Open Data portal credentials."
  }
]
```

### `GET /provenance`
Returns chronologically descending audit records of scientific datasets.

**Parameters:**
- `limit`: integer (default: 50, max: 200)
- `mode`: optional filter (`DEMO`, `HISTORICAL`, `USER_UPLOAD`, `LIVE_NRT`)

**Response `200 OK`:**
```json
[
  {
    "id": "prov-amphan-2020-baseline",
    "source": "IMD_BEST_TRACK",
    "provider": "India Meteorological Department (RSMC New Delhi)",
    "product": "Super Cyclonic Storm AMPHAN (BOB 01/2020) Best Track Archive",
    "dataset_version": "IMD-2020.1",
    "acquisition_time": "2020-05-16T00:00:00Z",
    "processing_time": "2026-09-10T21:20:00Z",
    "source_reference": "IMD Cyclone e-Atlas / RSMC Bulletin Archives",
    "license_notes": "Public Meteorological Data for Research & Education",
    "data_mode": "DEMO",
    "status": "VERIFIED_ARCHIVE"
  }
]
```

---

## 2. Observation Query Endpoints

### `GET /observations`
Query normalized cyclone center observations.

**Parameters:**
- `storm_id`: optional string filter (e.g. `BOB-01-2020`)
- `data_mode`: optional enum filter (`DEMO`, `HISTORICAL`, etc.)
- `source`: optional string filter
- `limit`: integer (default: 100)
- `offset`: integer (default: 0)

**Response `200 OK`:**
```json
[
  {
    "id": "obs-19f8a2c1",
    "storm_id": "BOB-01-2020",
    "storm_name": "AMPHAN",
    "timestamp": "2020-05-18T12:00:00Z",
    "latitude": 13.9,
    "longitude": 86.5,
    "wind_speed": 140.0,
    "central_pressure": 907.0,
    "classification": "Super Cyclonic Storm",
    "source": "IMD_BEST_TRACK_ARCHIVE",
    "source_reference": "IMD Cyclone e-Atlas (Demo Curated)",
    "data_mode": "DEMO",
    "created_at": "2026-09-10T21:24:00Z"
  }
]
```

### `GET /cyclones/{storm_id}/observations`
Retrieves chronologically sorted track observations for the specified cyclone ID.

---

## 3. Data Ingestion Endpoints

### `POST /data/upload`
Uploads raw satellite image granule (PNG, JPG, TIFF) up to 15MB.

**Form Data:**
- `file`: binary raster data (required)
- `satellite_name`: optional string (e.g. `INSAT-3D`)
- `sensor_type`: optional string (e.g. `IMAGER`)
- `product_type`: optional string (e.g. `TIR1`)

**Response `200 OK`:**
```json
{
  "id": "sat-upload-a4f1082c9e71",
  "filename": "sat-upload-a4f1082c9e71.png",
  "file_path": "/uploads/sat-upload-a4f1082c9e71.png",
  "file_type": "PNG",
  "file_size_bytes": 1048576,
  "dimensions": {
    "width": 1024,
    "height": 1024
  },
  "timestamp": "2026-09-10T21:30:00Z",
  "data_mode": "USER_UPLOAD",
  "message": "Satellite imagery uploaded and cataloged as USER UPLOADED DATA."
}
```

### `POST /data/import/historical`
Ingests historical cyclone tracks from CSV or JSON files with physical value validation.

**Content-Type:** `multipart/form-data` with `file` OR `application/json` with `{ "format": "json", "records": [ ... ] }`.

**Response `200 OK`:**
```json
{
  "records_received": 14,
  "records_imported": 14,
  "records_rejected": 0,
  "validation_errors": [],
  "imported_storm_id": "BOB-02-2021",
  "data_mode": "HISTORICAL",
  "timestamp": "2026-09-10T21:32:00Z"
}
```

