# CYCLONE-X: Scientific Data Pipeline & Architecture

## 1. End-to-End Pipeline Overview

```
[EXTERNAL SOURCES]               [SECURITY & VALIDATION]              [NORMALIZATION]
MOSDAC / ISRO  ---------\       /--> Extension & Magic Bytes ---> SatelliteObservation
IMD Best Track ----------> [INGESTION] -> Coordinate Bounds [-90,90] -> CycloneObservation
NASA GPM IMERG --------/   [ADAPTERS]  -> Barometric Physical Limits -> EnvironmentalObservation
User Uploads   ---------/       \--> Safe UUID File Isolation  ---> DataProvenance Audit
                                                    |
                                                    v
                                    [DATA SERVICE STORAGE LAYER]
                                    |-- PostgreSQL / PostGIS (Primary)
                                    \-- In-Memory Mock Store (Zero-Crash Fallback)
```

---

## 2. Ingestion Adapters

Adapters inherit from `DataSourceAdapter` (`backend.app.data.adapters.base.DataSourceAdapter`):
- `get_metadata() -> DataSourceInfo`: Reports provider details, supported channels, and live telemetry.
- `status() -> DataSourceStatus`: Honest check of environment variables and connectivity (`ACTIVE`, `NOT_CONFIGURED`, `AVAILABLE`, `UNAVAILABLE`).
- `validate(raw_data: Any) -> bool`: Checks incoming records for physical and schema consistency.
- `normalize(raw_data: Any) -> List[Any]`: Converts raw payload into typed Pydantic models.
- `fetch(**kwargs)`: Protected by zero-fabrication guard; raises `RuntimeError` if invoked when unconfigured rather than returning fake numbers.

---

## 3. Validation & Sanitization Engine

### Coordinate Validation
- Latitude must be between **-90.0°** and **+90.0°**.
- Longitude must be between **-180.0°** and **+180.0°**.
- Values outside these ranges are rejected and recorded in `validation_errors`.

### Meteorological Limits
- Sustained wind speed: **0.0 to 350.0 knots**.
- Central barometric pressure: **800.0 to 1050.0 hPa** (historical record minimum for North Indian Ocean is ~898 hPa; 800 hPa allows global extremes).
- Sea Surface Temperature: **-5.0°C to 45.0°C**.
- Rainfall rate: **0.0 to 500.0 mm/hr**.

### Upload Security Safeguards
- **Allowed Extensions**: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`.
- **Magic Bytes Validation**: Enforces PNG signature (`\x89PNG\r\n\x1a\n`), JPEG signature (`\xff\xd8\xff`), or TIFF byte orders (`II*` / `MM*`).
- **Path Traversal Prevention**: File is renamed to `sat-upload-{uuid12}.{ext}` and stored in `backend/uploads/` outside source code trees.
- **Size Limitation**: Hard ceiling of 15 MB per upload.
- **Execution Prevention**: Upload directories are mounted as non-executable static file stores.

---

## 4. Provenance & Auditing Ledger

The `ProvenanceTracker` (`backend.app.data.provenance.tracker.ProvenanceTracker`) maintains an immutable in-memory and database-backed ledger:
- Every dataset ingestion generates an audit entry with:
  - `source`: e.g. `MOSDAC_ISRO`, `IMD_BEST_TRACK`, `USER_UPLOAD`
  - `provider`: Publishing meteorological agency or user
  - `product`: Granule or batch identifier
  - `dataset_version`: Data catalog release version
  - `acquisition_time`: Physical acquisition UTC
  - `processing_time`: System ingestion UTC
  - `license_notes`: Scientific research usage license
  - `data_mode`: Provenance classification tag
  - `status`: Verification result

---

## 5. Storage & Database Fallback Architecture

1. **PostgreSQL + PostGIS**: Configured via `DATABASE_URL` for geospatial queries (`ST_DWithin`, spatial indexes on cyclone centers).
2. **Graceful Zero-Crash Fallback**: If PostgreSQL is offline or unconfigured, the application automatically initializes its in-memory repository with benchmark demo scenarios (Amphan 2020, Biparjoy 2023).
3. **Honest Reporting**: `/api/data/status` and the UI top status bar accurately report:
   - `STANDALONE_IN_MEMORY_FALLBACK (POSTGRESQL OFFLINE)`
   - `is_database_connected: false`
   - The platform never fabricates database connectivity.

