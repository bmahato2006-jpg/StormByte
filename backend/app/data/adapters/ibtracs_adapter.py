"""NOAA IBTrACS v04 Scientific Dataset Adapter for Cyclone-X Phase 7.

Ingests, validates, and normalizes global tropical cyclone best-track records
(specifically for the North Indian Ocean basin / Bay of Bengal & Arabian Sea).
"""

import csv
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.app.core.config import settings
from backend.app.data.normalizer import ScientificDataNormalizer
from backend.app.data.paths import DATA_ROOT
from backend.app.data.provenance import compute_file_sha256, PipelineProvenanceTracker
from backend.app.data.quality_control import ScientificQualityControl
from backend.app.data.schemas import (
    DataQualityReport,
    NormalizedTrackRecord,
    ScientificDataMode,
)


class IbtracsAdapter:
    """Adapter for reading, parsing, validating, and downloading NOAA IBTrACS CSV tracks."""

    DEFAULT_DOWNLOAD_URL = (
        "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.NI.list.v04r00.csv"
    )

    def __init__(self, raw_csv_path: Optional[str] = None, use_full: bool = False):
        if raw_csv_path:
            self.raw_path = Path(raw_csv_path)
        elif use_full:
            full_path = DATA_ROOT / "raw" / "ibtracs" / "ibtracs_nio_full.csv"
            subset_path = DATA_ROOT / "raw" / "ibtracs" / "ibtracs_nio_subset.csv"
            self.raw_path = full_path if full_path.exists() else subset_path
        else:
            subset_path = DATA_ROOT / "raw" / "ibtracs" / "ibtracs_nio_subset.csv"
            full_path = DATA_ROOT / "raw" / "ibtracs" / "ibtracs_nio_full.csv"
            self.raw_path = subset_path if subset_path.exists() else full_path

    @classmethod
    def load_full_dataset(cls) -> "IbtracsAdapter":
        """Factory method returning an adapter configured for the full NOAA NI archive."""
        return cls(use_full=True)

    @property
    def is_full_dataset(self) -> bool:
        """Returns True if pointing to the complete NOAA IBTrACS NI archive."""
        return "full" in self.raw_path.name.lower() or self.raw_path.stat().st_size > 1_000_000

    def is_available(self) -> bool:
        """Check if the raw IBTrACS CSV exists on disk."""
        return self.raw_path.exists() and self.raw_path.is_file()

    def download(self, force: bool = False, url: Optional[str] = None) -> str:
        """Download authentic NOAA NCEI IBTrACS NI archive from NOAA server."""
        target_path = DATA_ROOT / "raw" / "ibtracs" / "ibtracs_nio_full.csv"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists() and not force:
            self.raw_path = target_path
            return str(target_path)

        download_url = url or getattr(settings, "IBTRACS_SOURCE_URL", self.DEFAULT_DOWNLOAD_URL)
        req = urllib.request.Request(
            download_url,
            headers={"User-Agent": "Cyclone-X-Scientific-Client/1.0 (IMD-MoES Research)"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()

        with open(target_path, "wb") as f:
            f.write(data)

        self.raw_path = target_path
        return str(target_path)

    def parse_raw_records(self) -> List[Dict[str, Any]]:
        """Parse raw CSV rows into dictionaries, cleanly filtering NOAA units header row."""
        if not self.is_available():
            return []

        records: List[Dict[str, Any]] = []
        with open(self.raw_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # NOAA NCEI IBTrACS row 2 contains units ("degrees_north", "degrees_east", "kts", etc.)
                lat_val = row.get("LAT", "")
                if lat_val == "degrees_north" or lat_val == "degrees_east":
                    continue
                # Also ensure not an empty units row without SID
                if not row.get("SID") and not row.get("ISO_TIME"):
                    continue
                records.append(dict(row))
        return records

    def normalize(self, data_mode: ScientificDataMode = ScientificDataMode.REAL_RESEARCH) -> List[NormalizedTrackRecord]:
        """Convert raw rows to normalized Cyclone-X track records."""
        raw_rows = self.parse_raw_records()
        normalized: List[NormalizedTrackRecord] = []
        for r in raw_rows:
            norm_rec = ScientificDataNormalizer.normalize_ibtracs_row(r, data_mode=data_mode)
            if norm_rec:
                normalized.append(norm_rec)
        return normalized

    def run_quality_control(self) -> DataQualityReport:
        """Run automated quality control on raw IBTrACS data."""
        raw_rows = self.parse_raw_records()
        dataset_name = (
            "NOAA IBTrACS v04 (North Indian Ocean Full Archive)"
            if self.is_full_dataset
            else "NOAA IBTrACS v04 (North Indian Ocean Benchmark Subset)"
        )
        return ScientificQualityControl.audit_track_dataset(
            dataset_id="ibtracs_nio",
            dataset_name=dataset_name,
            records=raw_rows
        )

    def export_processed(self, output_dir: Optional[str] = None) -> str:
        """Export normalized records to data/processed/tracks/ibtracs_normalized.json."""
        out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[4] / "data" / "processed" / "tracks"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "ibtracs_normalized.json"

        norm_records = self.normalize()
        data = [r.model_dump(mode="json") for r in norm_records]
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return str(out_file)
