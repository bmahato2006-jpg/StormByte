"""Validation and Quarantine engine for Best Track observations."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from backend.app.data.best_track.schemas import BestTrackFix
from backend.app.data.best_track.source_registry import NIO_LAT_BOUNDS, NIO_LON_BOUNDS


class BestTrackValidator:
    """Performs rigorous meteorological and geospatial validation on candidate track fixes."""

    @staticmethod
    def validate_fix(fix: BestTrackFix) -> Tuple[bool, Optional[str]]:
        """Validates a single fix against scientific and geodetic rules.

        Returns (is_valid, rejection_reason).
        """
        # 1. Check coordinates range
        if not (-90.0 <= fix.latitude <= 90.0):
            return False, f"Latitude {fix.latitude} out of global range [-90, 90]"
        if not (-180.0 <= fix.longitude <= 180.0):
            return False, f"Longitude {fix.longitude} out of global range [-180, 180]"

        # 2. Check North Indian Ocean bounding box
        min_lat, max_lat = NIO_LAT_BOUNDS
        min_lon, max_lon = NIO_LON_BOUNDS
        if not (min_lat <= fix.latitude <= max_lat):
            return False, f"Latitude {fix.latitude} outside North Indian Ocean envelope [{min_lat}, {max_lat}]"
        if not (min_lon <= fix.longitude <= max_lon):
            return False, f"Longitude {fix.longitude} outside North Indian Ocean envelope [{min_lon}, {max_lon}]"

        # 3. Check year plausible
        year = fix.timestamp_utc.year
        if not (1842 <= year <= 2026):
            return False, f"Observation year {year} outside valid historical range [1842, 2026]"

        # 4. Check plausible wind if present
        if fix.max_wind_kt is not None:
            if not (10.0 <= fix.max_wind_kt <= 250.0):
                return False, f"Unphysical wind speed {fix.max_wind_kt} kt (expected 10-250 kt)"

        # 5. Check plausible central pressure if present
        if fix.central_pressure_hpa is not None:
            if not (850.0 <= fix.central_pressure_hpa <= 1040.0):
                return False, f"Unphysical central pressure {fix.central_pressure_hpa} hPa (expected 850-1040 hPa)"

        return True, None

    @classmethod
    def audit_fixes(
        cls, fixes: List[BestTrackFix]
    ) -> Tuple[List[BestTrackFix], List[Dict[str, Any]], Dict[str, Any]]:
        """Audits a collection of fixes, returning (valid_fixes, quarantine_records, report)."""
        valid: List[BestTrackFix] = []
        quarantine: List[Dict[str, Any]] = []

        total_records = len(fixes)
        missing_coords = 0
        missing_wind = 0
        missing_pressure = 0

        for f in fixes:
            if f.max_wind_kt is None:
                missing_wind += 1
            if f.central_pressure_hpa is None:
                missing_pressure += 1

            is_valid, reason = cls.validate_fix(f)
            if is_valid:
                valid.append(f)
            else:
                quarantine.append({
                    "id": f.id,
                    "storm_id": f.storm_id,
                    "storm_name": f.storm_name,
                    "source": f.source,
                    "timestamp": f.timestamp_utc.isoformat(),
                    "latitude": f.latitude,
                    "longitude": f.longitude,
                    "max_wind_kt": f.max_wind_kt,
                    "central_pressure_hpa": f.central_pressure_hpa,
                    "rejection_reason": reason,
                })

        report = {
            "total_records": total_records,
            "valid_records": len(valid),
            "quarantined_records": len(quarantine),
            "missing_wind": missing_wind,
            "missing_pressure": missing_pressure,
            "missing_coordinates": missing_coords,
            "validation_pass_rate_percent": round((len(valid) / total_records * 100), 2) if total_records else 0,
        }

        return valid, quarantine, report
