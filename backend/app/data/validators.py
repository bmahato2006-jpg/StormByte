"""Scientific data validators for Cyclone-X Phase 7.

Ensures strict quality control without silently substituting or fabricating data.
Questionable or impossible physical values are flagged rather than altered.
"""

from datetime import datetime
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ScientificDataValidator:
    """Scientific validation suite for meteorological observations and grids."""

    @staticmethod
    def validate_coordinates(lat: Any, lon: Any) -> Tuple[bool, Optional[str]]:
        """Validate geographic coordinates within physical planetary bounds."""
        try:
            f_lat = float(lat)
            f_lon = float(lon)
        except (ValueError, TypeError):
            return False, f"Coordinates cannot be parsed as floats: lat={lat}, lon={lon}"

        if math.isnan(f_lat) or math.isnan(f_lon):
            return False, "Coordinate values are NaN"

        if not (-90.0 <= f_lat <= 90.0):
            return False, f"Latitude {f_lat} is outside physical bounds [-90, 90]"

        if not (-180.0 <= f_lon <= 180.0):
            return False, f"Longitude {f_lon} is outside physical bounds [-180, 180]"

        return True, None

    @staticmethod
    def validate_wind_speed(wind_kt: Any) -> Tuple[bool, Optional[str]]:
        """Validate wind speed in knots."""
        if wind_kt is None or str(wind_kt).strip() in ("", "None", "null", "NaN"):
            return True, None  # Valid missing value

        try:
            val = float(wind_kt)
        except (ValueError, TypeError):
            return False, f"Wind speed '{wind_kt}' cannot be parsed as float"

        if math.isnan(val):
            return True, None  # Handled as missing

        if val < 0.0:
            return False, f"Wind speed {val} kt cannot be negative"

        if val > 350.0:
            return False, f"Wind speed {val} kt exceeds planetary maximum (~350 kt)"

        return True, None

    @staticmethod
    def validate_central_pressure(pressure_hpa: Any) -> Tuple[bool, Optional[str]]:
        """Validate estimated central pressure in hPa."""
        if pressure_hpa is None or str(pressure_hpa).strip() in ("", "None", "null", "NaN"):
            return True, None

        try:
            val = float(pressure_hpa)
        except (ValueError, TypeError):
            return False, f"Pressure '{pressure_hpa}' cannot be parsed as float"

        if math.isnan(val):
            return True, None

        if not (800.0 <= val <= 1050.0):
            return False, f"Central pressure {val} hPa is outside plausible atmospheric range [800, 1050]"

        return True, None

    @staticmethod
    def validate_timestamp(ts: Any) -> Tuple[bool, Optional[str], Optional[datetime]]:
        """Validate observation timestamp."""
        if isinstance(ts, datetime):
            return True, None, ts

        if not ts or not isinstance(ts, str):
            return False, "Timestamp is missing or invalid type", None

        # Common meteorological ISO/standard formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M",
            "%Y%m%d%H%M",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(ts.strip(), fmt)
                return True, None, dt
            except ValueError:
                continue

        return False, f"Timestamp '{ts}' does not match standard meteorological formats", None

    @staticmethod
    def validate_track_continuity(
        points: List[Dict[str, Any]],
        max_speed_kmh: float = 120.0
    ) -> List[str]:
        """Detect duplicate timestamps and physically impossible storm displacements."""
        issues: List[str] = []
        seen_timestamps = set()

        # Sort points by timestamp if possible
        valid_points = []
        for p in points:
            ts = p.get("timestamp")
            if ts:
                valid_points.append(p)
                if ts in seen_timestamps:
                    issues.append(f"Duplicate timestamp detected: {ts}")
                seen_timestamps.add(ts)

        # Check movement velocities between consecutive points
        for i in range(1, len(valid_points)):
            p0 = valid_points[i - 1]
            p1 = valid_points[i]

            t0 = p0.get("timestamp")
            t1 = p1.get("timestamp")
            if not isinstance(t0, datetime) or not isinstance(t1, datetime):
                continue

            dt_hours = (t1 - t0).total_seconds() / 3600.0
            if dt_hours <= 0:
                issues.append(f"Non-forward time sequence between {t0} and {t1}")
                continue

            lat0, lon0 = p0.get("latitude"), p0.get("longitude")
            lat1, lon1 = p1.get("latitude"), p1.get("longitude")
            if lat0 is None or lon0 is None or lat1 is None or lon1 is None:
                continue

            # Haversine distance
            dist_km = ScientificDataValidator._haversine_km(lat0, lon0, lat1, lon1)
            speed_kmh = dist_km / dt_hours
            if speed_kmh > max_speed_kmh:
                issues.append(
                    f"Physically implausible displacement speed: {speed_kmh:.1f} km/h "
                    f"between {t0} and {t1} (threshold: {max_speed_kmh} km/h)"
                )

        return issues

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return r * c

    @staticmethod
    def validate_satellite_image_file(file_path: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Validate satellite raster image readability and dimensions."""
        p = Path(file_path)
        if not p.exists():
            return False, f"File does not exist: {file_path}", {}

        stats: Dict[str, Any] = {
            "size_bytes": p.stat().st_size,
            "format": p.suffix.upper().replace(".", "")
        }

        # Check file extension
        valid_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".h5", ".nc"]
        if p.suffix.lower() not in valid_exts:
            return False, f"Unsupported satellite raster extension: {p.suffix}", stats

        # If PNG or JPG, verify readable via PIL if installed
        try:
            from PIL import Image
            with Image.open(p) as img:
                stats["dimensions"] = {"width": img.width, "height": img.height}
                stats["mode"] = img.mode
                if img.width < 32 or img.height < 32:
                    return False, f"Image dimensions ({img.width}x{img.height}) below minimum 32x32", stats
        except Exception as e:
            return False, f"Failed to decode image raster: {str(e)}", stats

        return True, None, stats

