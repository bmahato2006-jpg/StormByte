"""Canonical Normalizer for Best Track Observations."""

from datetime import datetime, timezone
import math
from typing import Any, Dict, Optional
from backend.app.data.best_track.schemas import BestTrackFix


class BestTrackNormalizer:
    """Standardizes heterogeneous best-track fixes into the canonical BestTrackFix schema."""

    GRADE_MAPPING = {
        "D": "Depression",
        "DD": "Deep Depression",
        "CS": "Cyclonic Storm",
        "SCS": "Severe Cyclonic Storm",
        "VSCS": "Very Severe Cyclonic Storm",
        "ESCS": "Extremely Severe Cyclonic Storm",
        "SUCS": "Super Cyclonic Storm",
        "SUPER CYCLONE": "Super Cyclonic Storm",
        "SUPER CYCLONIC STORM": "Super Cyclonic Storm",
        "CYCLONIC STORM": "Cyclonic Storm",
        "DEPRESSION": "Depression",
        "DEEP DEPRESSION": "Deep Depression",
    }

    @staticmethod
    def normalize_longitude(lon: float) -> float:
        """Normalizes longitude to the [-180.0, 180.0] range."""
        while lon > 180.0:
            lon -= 360.0
        while lon < -180.0:
            lon += 360.0
        return round(lon, 4)

    @staticmethod
    def normalize_latitude(lat: float) -> float:
        """Normalizes latitude to [-90.0, 90.0] range."""
        return round(max(-90.0, min(90.0, lat)), 4)

    @staticmethod
    def normalize_timestamp(dt: Any) -> datetime:
        """Converts date/datetime/string into timezone-aware UTC datetime."""
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        if isinstance(dt, str):
            clean = dt.strip().replace("Z", "+00:00")
            parsed = datetime.fromisoformat(clean)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        raise ValueError(f"Cannot parse timestamp from {dt!r}")

    @classmethod
    def categorize_by_wind(cls, wind_kt: Optional[float], original_grade: Optional[str] = None) -> Optional[str]:
        """Categorizes storm intensity using standard IMD/RSMC classifications."""
        if original_grade:
            cleaned = str(original_grade).strip().upper()
            if cleaned in cls.GRADE_MAPPING:
                return cls.GRADE_MAPPING[cleaned]

        if wind_kt is None or math.isnan(wind_kt):
            return None

        w = float(wind_kt)
        if w < 17:
            return "Low Pressure Area"
        elif w <= 27:
            return "Well Marked Low"
        elif w <= 33:
            return "Depression"
        elif w <= 47:
            return "Deep Depression" if w <= 33 else "Cyclonic Storm"
        elif w <= 63:
            return "Severe Cyclonic Storm"
        elif w <= 89:
            return "Very Severe Cyclonic Storm"
        elif w <= 119:
            return "Extremely Severe Cyclonic Storm"
        else:
            return "Super Cyclonic Storm"

    @classmethod
    def normalize_record(
        cls,
        raw_fix: Dict[str, Any],
        source: str,
        source_file: str,
        source_url: Optional[str] = None,
        retrieved_at_utc: Optional[datetime] = None,
    ) -> Optional[BestTrackFix]:
        """Maps a raw observation row to a validated canonical BestTrackFix."""
        try:
            ts = cls.normalize_timestamp(raw_fix["timestamp"])
            lat = cls.normalize_latitude(float(raw_fix["latitude"]))
            lon = cls.normalize_longitude(float(raw_fix["longitude"]))
        except Exception:
            return None

        # Clean optional floats without zero-fabrication
        wind = raw_fix.get("max_wind_kt")
        wind_val: Optional[float] = None
        if wind is not None and str(wind).strip() not in ("", "nan", "NaN", "None", "-999", "-9999"):
            try:
                fv = float(wind)
                if not math.isnan(fv) and fv >= 0:
                    wind_val = round(fv, 1)
            except (ValueError, TypeError):
                pass

        pres = raw_fix.get("central_pressure_hpa")
        pres_val: Optional[float] = None
        if pres is not None and str(pres).strip() not in ("", "nan", "NaN", "None", "-999", "-9999"):
            try:
                fp = float(pres)
                if not math.isnan(fp) and fp >= 0:
                    pres_val = round(fp, 1)
            except (ValueError, TypeError):
                pass

        ci = raw_fix.get("ci_number")
        ci_val: Optional[float] = None
        if ci is not None and str(ci).strip() not in ("", "nan", "NaN", "None"):
            try:
                fci = float(ci)
                if not math.isnan(fci):
                    ci_val = round(fci, 1)
            except (ValueError, TypeError):
                pass

        dp = raw_fix.get("pressure_drop_hpa")
        dp_val: Optional[float] = None
        if dp is not None and str(dp).strip() not in ("", "nan", "NaN", "None"):
            try:
                fdp = float(dp)
                if not math.isnan(fdp):
                    dp_val = round(fdp, 1)
            except (ValueError, TypeError):
                pass

        grade = str(raw_fix.get("grade") or "").strip() or None
        category = cls.categorize_by_wind(wind_val, original_grade=grade)

        storm_name = str(raw_fix.get("storm_name") or "UNNAMED").strip().upper()
        storm_id = str(raw_fix.get("storm_id") or f"{ts.year}-{storm_name}").strip()
        basin = str(raw_fix.get("basin") or "NI").strip().upper()
        subbasin = str(raw_fix.get("subbasin") or "").strip().upper() or None

        rec_id = f"{source}-{storm_id}-{ts.strftime('%Y%m%d%H%M')}"
        now_utc = retrieved_at_utc or datetime.now(timezone.utc)

        return BestTrackFix(
            id=rec_id,
            storm_id=storm_id,
            storm_name=storm_name,
            basin=basin,
            subbasin=subbasin,
            source=source,
            source_agency=raw_fix.get("source_agency") or ("IMD" if "IMD" in source else "NOAA"),
            source_record_id=str(raw_fix.get("source_record_id") or ""),
            timestamp_utc=ts,
            latitude=lat,
            longitude=lon,
            max_wind_kt=wind_val,
            central_pressure_hpa=pres_val,
            cyclone_category=category,
            ci_number=ci_val,
            pressure_drop_hpa=dp_val,
            grade=grade,
            is_best_track=True,
            source_file=source_file,
            source_url=source_url,
            retrieved_at_utc=now_utc,
            data_quality="VERIFIED_OFFICIAL",
            validation_status="PASSED",
            cross_source_difference=False,
        )
