"""Dataset Merger and Manifest Generator for Best Track Data."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List
from backend.app.data.best_track.schemas import BestTrackFix, StormIndexRecord
from backend.app.data.best_track.source_registry import (
    CANONICAL_CSV,
    CANONICAL_JSON,
    IMD_MANIFEST_CSV,
    QUARANTINE_CSV,
    STORM_INDEX_CSV,
    VALIDATION_REPORT_JSON,
)


class BestTrackMerger:
    """Merges validated IMD and IBTrACS observations, builds the storm index, and exports artifacts."""

    @staticmethod
    def build_storm_index(fixes: List[BestTrackFix]) -> List[StormIndexRecord]:
        """Aggregates individual fixes into summary storm index records."""
        grouped: Dict[str, List[BestTrackFix]] = {}
        for f in fixes:
            grouped.setdefault(f.storm_id, []).append(f)

        index_records: List[StormIndexRecord] = []
        for storm_id, s_fixes in grouped.items():
            s_fixes.sort(key=lambda x: x.timestamp_utc)
            first_f = s_fixes[0]
            last_f = s_fixes[-1]

            winds = [x.max_wind_kt for x in s_fixes if x.max_wind_kt is not None]
            pressures = [x.central_pressure_hpa for x in s_fixes if x.central_pressure_hpa is not None]

            max_w = max(winds) if winds else None
            min_p = min(pressures) if pressures else None

            # Determine highest classification attained
            category = last_f.cyclone_category or first_f.cyclone_category
            for x in s_fixes:
                if x.cyclone_category in ("Super Cyclonic Storm", "Extremely Severe Cyclonic Storm", "Very Severe Cyclonic Storm"):
                    category = x.cyclone_category
                    break

            index_records.append(
                StormIndexRecord(
                    storm_id=storm_id,
                    storm_name=first_f.storm_name,
                    year=first_f.timestamp_utc.year,
                    basin=first_f.basin,
                    source=first_f.source,
                    start_time=first_f.timestamp_utc.isoformat(),
                    end_time=last_f.timestamp_utc.isoformat(),
                    point_count=len(s_fixes),
                    max_wind_kt=max_w,
                    min_pressure_hpa=min_p,
                    cyclone_category=category,
                )
            )

        # Sort by year desc, then storm_id
        index_records.sort(key=lambda x: (x.year, x.start_time), reverse=True)
        return index_records

    @classmethod
    def export_all(
        cls,
        fixes: List[BestTrackFix],
        quarantine: List[Dict[str, Any]],
        validation_report: Dict[str, Any],
        imd_manifest_rows: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Serializes canonical CSV, JSON, index, quarantine, and manifests to disk."""
        # 1. Export Storm Index CSV
        index_records = cls.build_storm_index(fixes)
        with open(STORM_INDEX_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "storm_id", "storm_name", "year", "basin", "source",
                "start_time", "end_time", "point_count", "max_wind_kt",
                "min_pressure_hpa", "cyclone_category"
            ])
            for r in index_records:
                writer.writerow([
                    r.storm_id, r.storm_name, r.year, r.basin, r.source,
                    r.start_time, r.end_time, r.point_count,
                    r.max_wind_kt if r.max_wind_kt is not None else "",
                    r.min_pressure_hpa if r.min_pressure_hpa is not None else "",
                    r.cyclone_category or "",
                ])

        # 2. Export Master Canonical CSV
        with open(CANONICAL_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "storm_id", "storm_name", "basin", "subbasin", "source",
                "source_agency", "source_record_id", "timestamp_utc", "latitude",
                "longitude", "max_wind_kt", "central_pressure_hpa", "cyclone_category",
                "ci_number", "pressure_drop_hpa", "grade", "is_best_track",
                "source_file", "retrieved_at_utc", "cross_source_difference"
            ])
            for x in fixes:
                writer.writerow([
                    x.id, x.storm_id, x.storm_name, x.basin, x.subbasin or "",
                    x.source, x.source_agency, x.source_record_id or "",
                    x.timestamp_utc.isoformat(), x.latitude, x.longitude,
                    x.max_wind_kt if x.max_wind_kt is not None else "",
                    x.central_pressure_hpa if x.central_pressure_hpa is not None else "",
                    x.cyclone_category or "",
                    x.ci_number if x.ci_number is not None else "",
                    x.pressure_drop_hpa if x.pressure_drop_hpa is not None else "",
                    x.grade or "",
                    1 if x.is_best_track else 0,
                    x.source_file,
                    x.retrieved_at_utc.isoformat(),
                    1 if x.cross_source_difference else 0,
                ])

        # 3. Export Master Canonical JSON
        json_data = [x.model_dump(mode="json") for x in fixes]
        with open(CANONICAL_JSON, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        # 4. Export Quarantine CSV
        with open(QUARANTINE_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "storm_id", "storm_name", "source", "timestamp",
                "latitude", "longitude", "max_wind_kt", "central_pressure_hpa", "rejection_reason"
            ])
            for q in quarantine:
                writer.writerow([
                    q.get("id"), q.get("storm_id"), q.get("storm_name"), q.get("source"),
                    q.get("timestamp"), q.get("latitude"), q.get("longitude"),
                    q.get("max_wind_kt") if q.get("max_wind_kt") is not None else "",
                    q.get("central_pressure_hpa") if q.get("central_pressure_hpa") is not None else "",
                    q.get("rejection_reason"),
                ])

        # 5. Export Validation Report JSON
        with open(VALIDATION_REPORT_JSON, "w", encoding="utf-8") as f:
            json.dump(validation_report, f, indent=2)

        # 6. Export IMD Manifest CSV
        with open(IMD_MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "year", "source", "original_filename", "local_path",
                "source_reference", "sha256", "download_timestamp",
                "parse_status", "record_count"
            ])
            for m in imd_manifest_rows:
                writer.writerow([
                    m.get("year"), m.get("source"), m.get("original_filename"),
                    m.get("local_path"), m.get("source_reference"),
                    m.get("sha256"), m.get("download_timestamp"),
                    m.get("parse_status"), m.get("record_count"),
                ])

        return {
            "canonical_csv": str(CANONICAL_CSV),
            "canonical_json": str(CANONICAL_JSON),
            "storm_index_csv": str(STORM_INDEX_CSV),
            "quarantine_csv": str(QUARANTINE_CSV),
            "validation_report_json": str(VALIDATION_REPORT_JSON),
            "imd_manifest_csv": str(IMD_MANIFEST_CSV),
        }
