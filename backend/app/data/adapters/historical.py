"""Historical Cyclone Data Adapter.

Parses and validates CSV and JSON historical cyclone track files (such as IMD Best Track,
IBTrACS, JTWC historical archives).

Provides rigorous scientific validation:
- Latitude [-90.0, 90.0]
- Longitude [-180.0, 180.0]
- Valid ISO / standard timestamps
- Numerical wind speed in physical range [0.0, 350.0] knots
- Atmospheric pressure in physical range [800.0, 1050.0] hPa
- Required storm identification
"""

import csv
import io
import json
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Tuple, Union
from backend.app.data.adapters.base import DataSourceAdapter
from backend.app.data.schemas.observations import (
    DataSourceInfo,
    DataSourceStatus,
    DataMode,
    CycloneObservation,
    ImportHistoricalReport,
)


class HistoricalTrackAdapter(DataSourceAdapter):
    """Adapter for ingesting and validating historical cyclone tracks from CSV/JSON."""

    def __init__(self):
        super().__init__(name="HISTORICAL_TRACK_ARCHIVE", provider="RSMC / IBTrACS / Local Archives", requires_auth=False)
        self.supported_formats = ["CSV", "JSON"]

    def is_configured(self) -> bool:
        return True

    def status(self) -> DataSourceStatus:
        return DataSourceStatus.AVAILABLE

    def get_metadata(self) -> DataSourceInfo:
        return DataSourceInfo(
            name="Historical Best Track Importer",
            provider="WMO / RSMC / Local Datasets",
            description="High-fidelity parser for historical cyclone tracks with coordinate and physical value verification.",
            supported_products=["CSV_TRACK", "JSON_TRACK", "IBTrACS_SUBSET"],
            status=DataSourceStatus.AVAILABLE,
            data_mode=DataMode.HISTORICAL,
            configured=True,
            requires_auth=False,
            last_checked=datetime.now(timezone.utc),
            notes="Ready for historical archive imports. Always tagged as HISTORICAL DATA.",
        )

    def validate(self, raw_data: Any) -> bool:
        """Validate an individual record dictionary."""
        if not isinstance(raw_data, dict):
            return False
        return "storm_id" in raw_data and "latitude" in raw_data and "longitude" in raw_data

    def _parse_timestamp(self, ts_raw: Any) -> datetime:
        """Attempt multiple common datetime formats."""
        if isinstance(ts_raw, datetime):
            return ts_raw if ts_raw.tzinfo else ts_raw.replace(tzinfo=timezone.utc)
        
        ts_str = str(ts_raw).strip()
        # Try standard ISO
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass

        # Try common meteorological track formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y%m%d%H",
            "%Y%m%d%H%M",
            "%d/%m/%Y %H:%M",
            "%d-%m-%Y %H:%M:%S",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(ts_str, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        raise ValueError(f"Invalid timestamp format: '{ts_str}'")

    def validate_and_normalize_record(
        self,
        record: Dict[str, Any],
        row_idx: int
    ) -> Tuple[Optional[CycloneObservation], Optional[str]]:
        """Validate individual record, returning (observation, error_message)."""
        # 1. Storm identification
        storm_id = str(record.get("storm_id", "")).strip()
        if not storm_id:
            return None, f"Row {row_idx}: Missing required field 'storm_id'"

        storm_name = str(record.get("storm_name", storm_id)).strip() or storm_id

        # 2. Coordinates
        try:
            lat = float(record.get("latitude", 0.0))
            if not (-90.0 <= lat <= 90.0):
                return None, f"Row {row_idx}: Latitude {lat} outside valid range [-90, 90]"
        except (ValueError, TypeError):
            return None, f"Row {row_idx}: Invalid latitude '{record.get('latitude')}'"

        try:
            lon = float(record.get("longitude", 0.0))
            if not (-180.0 <= lon <= 180.0):
                return None, f"Row {row_idx}: Longitude {lon} outside valid range [-180, 180]"
        except (ValueError, TypeError):
            return None, f"Row {row_idx}: Invalid longitude '{record.get('longitude')}'"

        # 3. Timestamp
        raw_ts = record.get("timestamp")
        if not raw_ts:
            return None, f"Row {row_idx}: Missing required timestamp"
        try:
            ts = self._parse_timestamp(raw_ts)
        except Exception as e:
            return None, f"Row {row_idx}: {str(e)}"

        # 4. Wind speed
        try:
            wind = float(record.get("wind_speed", 30.0))
            if wind < 0.0 or wind > 350.0:
                return None, f"Row {row_idx}: Wind speed {wind} knots outside physical range [0, 350]"
        except (ValueError, TypeError):
            return None, f"Row {row_idx}: Invalid wind speed '{record.get('wind_speed')}'"

        # 5. Central Pressure
        try:
            pressure = float(record.get("central_pressure", 1000.0))
            if pressure < 800.0 or pressure > 1050.0:
                return None, f"Row {row_idx}: Pressure {pressure} hPa outside physical range [800, 1050]"
        except (ValueError, TypeError):
            return None, f"Row {row_idx}: Invalid pressure '{record.get('central_pressure')}'"

        classification = str(record.get("classification", "Cyclonic Storm"))
        source = str(record.get("source", "HISTORICAL_IMPORT"))
        ref = str(record.get("source_reference", "Historical Best Track Dataset"))

        obs = CycloneObservation(
            id=str(record.get("id", uuid.uuid4())),
            storm_id=storm_id,
            storm_name=storm_name,
            timestamp=ts,
            latitude=lat,
            longitude=lon,
            wind_speed=wind,
            central_pressure=pressure,
            classification=classification,
            source=source,
            source_reference=ref,
            data_mode=DataMode.HISTORICAL,
        )
        return obs, None

    def normalize(self, raw_data: Any, **kwargs) -> List[CycloneObservation]:
        """Normalizes parsed records list."""
        if isinstance(raw_data, list):
            observations = []
            for idx, item in enumerate(raw_data, 1):
                obs, err = self.validate_and_normalize_record(item, idx)
                if obs:
                    observations.append(obs)
            return observations
        return []

    def import_content(
        self,
        content: Union[str, bytes],
        file_format: str
    ) -> Tuple[List[CycloneObservation], ImportHistoricalReport]:
        """Parse raw CSV or JSON text/bytes and return observations plus a detailed report."""
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = content

        errors: List[str] = []
        valid_observations: List[CycloneObservation] = []
        records_received = 0
        storm_ids_seen = set()

        fmt = file_format.strip().lower()

        if fmt == "json":
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    # Handle potential wrapper: {"records": [...]} or {"observations": [...]}
                    data = data.get("records") or data.get("observations") or [data]
                if not isinstance(data, list):
                    return [], ImportHistoricalReport(
                        records_received=0,
                        records_imported=0,
                        records_rejected=1,
                        validation_errors=["JSON must be an array of records or contain a 'records' list."],
                    )
                records_received = len(data)
                for idx, row in enumerate(data, 1):
                    if not isinstance(row, dict):
                        errors.append(f"Record {idx}: Must be a JSON object")
                        continue
                    obs, err = self.validate_and_normalize_record(row, idx)
                    if err:
                        errors.append(err)
                    elif obs:
                        valid_observations.append(obs)
                        storm_ids_seen.add(obs.storm_id)
            except Exception as e:
                return [], ImportHistoricalReport(
                    records_received=0,
                    records_imported=0,
                    records_rejected=1,
                    validation_errors=[f"Failed to parse JSON: {str(e)}"],
                )

        elif fmt == "csv":
            try:
                reader = csv.DictReader(io.StringIO(text))
                if not reader.fieldnames:
                    return [], ImportHistoricalReport(
                        records_received=0,
                        records_imported=0,
                        records_rejected=1,
                        validation_errors=["CSV file is empty or missing header row"],
                    )
                rows = list(reader)
                records_received = len(rows)
                for idx, row in enumerate(rows, 1):
                    obs, err = self.validate_and_normalize_record(row, idx)
                    if err:
                        errors.append(err)
                    elif obs:
                        valid_observations.append(obs)
                        storm_ids_seen.add(obs.storm_id)
            except Exception as e:
                return [], ImportHistoricalReport(
                    records_received=0,
                    records_imported=0,
                    records_rejected=1,
                    validation_errors=[f"Failed to parse CSV: {str(e)}"],
                )
        else:
            return [], ImportHistoricalReport(
                records_received=0,
                records_imported=0,
                records_rejected=1,
                validation_errors=[f"Unsupported file format '{file_format}'. Supported: csv, json"],
            )

        report = ImportHistoricalReport(
            records_received=records_received,
            records_imported=len(valid_observations),
            records_rejected=len(errors),
            validation_errors=errors[:50],  # cap returned errors
            imported_storm_id=next(iter(storm_ids_seen)) if len(storm_ids_seen) == 1 else (
                f"{len(storm_ids_seen)} storms" if storm_ids_seen else None
            ),
            data_mode=DataMode.HISTORICAL,
        )
        return valid_observations, report

