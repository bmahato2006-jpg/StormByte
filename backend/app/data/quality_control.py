"""Quality control engine for Cyclone-X Phase 7.

Conducts automated scientific validation of raw and normalized datasets,
generating an auditable DataQualityReport with PASS / WARNING / FAIL verdicts.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.app.data.schemas import DataQualityReport, QualityStatus
from backend.app.data.validators import ScientificDataValidator


class ScientificQualityControl:
    """Evaluates scientific integrity and flags data anomalies."""

    @staticmethod
    def audit_track_dataset(dataset_id: str, dataset_name: str, records: List[Dict[str, Any]]) -> DataQualityReport:
        """Run comprehensive quality control over a set of track observations."""
        report = DataQualityReport(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            total_records=len(records),
            variables=["latitude", "longitude", "max_wind_kt", "central_pressure_hpa", "timestamp"],
            units={
                "latitude": "deg_north",
                "longitude": "deg_east",
                "max_wind_kt": "knots",
                "central_pressure_hpa": "hPa",
                "timestamp": "ISO-8601 UTC"
            }
        )

        if not records:
            report.quality_status = QualityStatus.FAIL
            report.issues.append("Dataset contains 0 records.")
            return report

        valid_count = 0
        invalid_count = 0
        missing_counts: Dict[str, int] = {
            "latitude": 0,
            "longitude": 0,
            "max_wind_kt": 0,
            "central_pressure_hpa": 0,
            "timestamp": 0
        }

        timestamps: List[datetime] = []
        lats: List[float] = []
        lons: List[float] = []
        seen_timestamps_per_storm: Dict[str, set] = {}

        for r in records:
            is_record_valid = True
            storm_id = r.get("storm_id") or r.get("SID") or "DEFAULT"

            # Check coords
            lat = r.get("latitude") if "latitude" in r else r.get("LAT")
            lon = r.get("longitude") if "longitude" in r else r.get("LON")
            if lat is None:
                missing_counts["latitude"] += 1
                is_record_valid = False
            if lon is None:
                missing_counts["longitude"] += 1
                is_record_valid = False

            if lat is not None and lon is not None:
                valid_coords, coord_err = ScientificDataValidator.validate_coordinates(lat, lon)
                if not valid_coords:
                    is_record_valid = False
                    report.issues.append(f"Coordinate error: {coord_err}")
                else:
                    lats.append(float(lat))
                    lons.append(float(lon))

            # Check timestamp
            ts = r.get("timestamp") or r.get("ISO_TIME")
            if ts is None:
                missing_counts["timestamp"] += 1
                is_record_valid = False
            else:
                valid_ts, ts_err, dt = ScientificDataValidator.validate_timestamp(ts)
                if not valid_ts or not dt:
                    is_record_valid = False
                    report.issues.append(f"Timestamp error: {ts_err}")
                else:
                    timestamps.append(dt)
                    if storm_id not in seen_timestamps_per_storm:
                        seen_timestamps_per_storm[storm_id] = set()
                    if dt in seen_timestamps_per_storm[storm_id]:
                        report.duplicate_timestamps += 1
                    seen_timestamps_per_storm[storm_id].add(dt)

            # Check wind
            wind = r.get("max_wind_kt") if "max_wind_kt" in r else r.get("WMO_WIND")
            if wind is None or str(wind).strip() in ("", "None", "null", "NaN"):
                missing_counts["max_wind_kt"] += 1
            else:
                valid_w, w_err = ScientificDataValidator.validate_wind_speed(wind)
                if not valid_w:
                    is_record_valid = False
                    report.issues.append(f"Wind speed error: {w_err}")

            # Check pressure
            pres = r.get("central_pressure_hpa") if "central_pressure_hpa" in r else r.get("WMO_PRES")
            if pres is None or str(pres).strip() in ("", "None", "null", "NaN"):
                missing_counts["central_pressure_hpa"] += 1
            else:
                valid_p, p_err = ScientificDataValidator.validate_central_pressure(pres)
                if not valid_p:
                    is_record_valid = False
                    report.issues.append(f"Pressure error: {p_err}")

            if is_record_valid:
                valid_count += 1
            else:
                invalid_count += 1

        report.valid_records = valid_count
        report.invalid_records = invalid_count
        report.missing_values = missing_counts

        # Spatial and temporal summary
        if timestamps:
            min_ts = min(timestamps).strftime("%Y-%m-%d")
            max_ts = max(timestamps).strftime("%Y-%m-%d")
            report.time_range = f"{min_ts} to {max_ts}"

        if lats and lons:
            report.spatial_coverage = (
                f"Lat [{min(lats):.2f}, {max(lats):.2f}], Lon [{min(lons):.2f}, {max(lons):.2f}]"
            )

        # Overall Status
        if invalid_count == 0 and report.duplicate_timestamps == 0:
            report.quality_status = QualityStatus.PASS
        elif invalid_count < len(records) * 0.1:  # Less than 10% issues
            report.quality_status = QualityStatus.WARNING
            report.issues.append("Dataset passed with minor warnings or missing meteorological fields.")
        else:
            report.quality_status = QualityStatus.FAIL
            report.issues.append("Dataset failed QC: excessive invalid records or malformed data.")

        return report

    @staticmethod
    def audit_raster_dataset(file_path: str, dataset_id: str, dataset_name: str) -> DataQualityReport:
        """Audit a geospatial raster (e.g. WorldPop GeoTIFF or INSAT satellite imagery)."""
        from PIL import Image
        from backend.app.data.provenance import compute_file_sha256

        p = Path(file_path)
        report = DataQualityReport(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            total_records=1,
            variables=["dimensions", "bands", "spatial_resolution", "integrity_checksum"],
            units={"dimensions": "pixels (width x height)", "resolution": "arc-seconds / km"}
        )

        if not p.exists():
            report.quality_status = QualityStatus.FAIL
            report.invalid_records = 1
            report.issues.append(f"Raster file does not exist on disk: {file_path}")
            return report

        if p.stat().st_size == 0:
            report.quality_status = QualityStatus.FAIL
            report.invalid_records = 1
            report.issues.append(f"Raster file is 0 bytes (empty/corrupt): {file_path}")
            return report

        try:
            with Image.open(p) as im:
                report.valid_records = 1
                report.quality_status = QualityStatus.PASS
                report.spatial_coverage = f"Raster dims: {im.width}x{im.height} ({im.mode} mode, {len(im.getbands())} band(s))"
        except Exception as e:
            report.quality_status = QualityStatus.FAIL
            report.invalid_records = 1
            report.issues.append(f"Raster failed format verification: {str(e)}")

        return report

    @staticmethod
    def audit_satellite_dataset(dataset_id: str, dataset_name: str, granules: List[Dict[str, Any]]) -> DataQualityReport:
        """Run quality control over a catalog of satellite granules."""
        report = DataQualityReport(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            total_records=len(granules),
            variables=["granule_id", "satellite", "sensor", "channel", "acquisition_time"],
            units={"acquisition_time": "ISO-8601 UTC"}
        )

        if not granules:
            report.quality_status = QualityStatus.WARNING
            report.issues.append("Catalog contains 0 granules registered.")
            return report

        valid = 0
        invalid = 0
        missing = {"file_path": 0, "acquisition_time": 0}

        for g in granules:
            fpath = g.get("file_path")
            if not fpath:
                missing["file_path"] += 1
            elif not Path(fpath).exists():
                report.issues.append(f"Referenced raster file not on disk: {fpath}")

            ts = g.get("acquisition_time")
            if not ts:
                missing["acquisition_time"] += 1
                invalid += 1
            else:
                valid += 1

        report.valid_records = valid
        report.invalid_records = invalid
        report.missing_values = missing
        report.quality_status = QualityStatus.PASS if invalid == 0 else QualityStatus.WARNING
        return report

    @classmethod
    def generate_all_qc_reports(cls, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Execute automated QC across all available datasets and write JSON reports."""
        import json
        from backend.app.data.adapters.ibtracs_adapter import IbtracsAdapter
        from backend.app.data.adapters.imd_adapter import ImdAdapter
        from backend.app.data.adapters.mosdac_adapter import MosdacAdapter
        from backend.app.data.adapters.exposure_adapter import ExposureAdapter
        from backend.app.data.paths import DATA_ROOT

        out_dir = Path(output_dir) if output_dir else DATA_ROOT / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)

        summary: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reports": {}
        }

        # 1. IBTrACS Full QC
        ibtracs_adapter = IbtracsAdapter.load_full_dataset()
        ibtracs_report = ibtracs_adapter.run_quality_control()
        summary["reports"]["ibtracs_nio"] = ibtracs_report.model_dump(mode="json")
        with open(out_dir / "ibtracs_qc_report.json", "w", encoding="utf-8") as f:
            json.dump(summary["reports"]["ibtracs_nio"], f, indent=2)

        # 2. IMD QC
        imd_adapter = ImdAdapter()
        imd_report = imd_adapter.run_quality_control()
        summary["reports"]["imd_best_track"] = imd_report.model_dump(mode="json")
        with open(out_dir / "imd_qc_report.json", "w", encoding="utf-8") as f:
            json.dump(summary["reports"]["imd_best_track"], f, indent=2)

        # 3. MOSDAC Satellite QC
        mosdac_adapter = MosdacAdapter()
        mosdac_report = mosdac_adapter.run_quality_control()
        summary["reports"]["mosdac_insat"] = mosdac_report.model_dump(mode="json")
        with open(out_dir / "satellite_qc_report.json", "w", encoding="utf-8") as f:
            json.dump(summary["reports"]["mosdac_insat"], f, indent=2)

        # 4. Exposure & Population Raster QC
        exposure_adapter = ExposureAdapter()
        exposure_report = exposure_adapter.run_quality_control()
        summary["reports"]["worldpop_density"] = exposure_report.model_dump(mode="json")
        with open(out_dir / "exposure_qc_report.json", "w", encoding="utf-8") as f:
            json.dump(summary["reports"]["worldpop_density"], f, indent=2)

        # Master summary
        with open(out_dir / "master_qc_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

