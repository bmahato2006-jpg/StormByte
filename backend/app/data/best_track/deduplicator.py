"""Deduplication and Cross-Source Pairing Engine for Best Track Observations."""

import math
from typing import Dict, List, Set, Tuple
from backend.app.data.best_track.schemas import BestTrackFix
from backend.app.data.best_track.source_registry import EARTH_RADIUS_KM


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


class BestTrackDeduplicator:
    """Performs deterministic deduplication and cross-source discrepancy analysis."""

    @staticmethod
    def deduplicate(fixes: List[BestTrackFix]) -> Tuple[List[BestTrackFix], int]:
        """Eliminates exact duplicate fixes within each source by (source, storm_id, timestamp_utc)."""
        seen_keys: Set[str] = set()
        unique_fixes: List[BestTrackFix] = []
        duplicate_count = 0

        for f in fixes:
            key = f"{f.source}:{f.storm_id}:{f.timestamp_utc.isoformat()}"
            if key in seen_keys:
                duplicate_count += 1
                continue
            seen_keys.add(key)
            unique_fixes.append(f)

        return unique_fixes, duplicate_count

    @staticmethod
    def flag_cross_source_differences(
        fixes: List[BestTrackFix], max_dist_km: float = 30.0, max_time_hours: float = 1.0
    ) -> List[BestTrackFix]:
        """Identifies paired IMD and IBTrACS observations and flags significant differences."""
        # Index by rounded 3-hour timestamp bucket
        by_time: Dict[str, List[BestTrackFix]] = {}
        for f in fixes:
            bucket = f.timestamp_utc.strftime("%Y-%m-%d-%H")
            by_time.setdefault(bucket, []).append(f)

        for bucket, bucket_fixes in by_time.items():
            if len(bucket_fixes) < 2:
                continue
            imd_fixes = [x for x in bucket_fixes if "IMD" in x.source]
            ibtracs_fixes = [x for x in bucket_fixes if "IBTRACS" in x.source]

            for imd_f in imd_fixes:
                for ib_f in ibtracs_fixes:
                    # Check storm name match
                    if imd_f.storm_name == ib_f.storm_name and imd_f.storm_name != "UNNAMED":
                        d_km = haversine(imd_f.latitude, imd_f.longitude, ib_f.latitude, ib_f.longitude)
                        # Significant difference if distance > 25 km or wind difference > 10 kt
                        wind_diff = 0.0
                        if imd_f.max_wind_kt is not None and ib_f.max_wind_kt is not None:
                            wind_diff = abs(imd_f.max_wind_kt - ib_f.max_wind_kt)

                        if d_km > 25.0 or wind_diff > 10.0:
                            imd_f.cross_source_difference = True
                            ib_f.cross_source_difference = True

        return fixes
