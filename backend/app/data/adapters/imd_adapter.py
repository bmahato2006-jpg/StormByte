"""IMD Historical Cyclone Best-Track Dataset Adapter for Cyclone-X Phase 7.

Ingests authoritative historical bulletin and e-Atlas archives from RSMC New Delhi
(India Meteorological Department).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.app.data.normalizer import ScientificDataNormalizer
from backend.app.data.paths import DATA_ROOT
from backend.app.data.quality_control import ScientificQualityControl
from backend.app.data.schemas import (
    DataQualityReport,
    NormalizedTrackRecord,
    ScientificDataMode,
)


class ImdAdapter:
    """Adapter for reading, validating, and normalizing official IMD historical records."""

    def __init__(self, raw_json_path: Optional[str] = None):
        if raw_json_path:
            self.raw_path = Path(raw_json_path)
        else:
            self.raw_path = DATA_ROOT / "raw" / "imd" / "imd_cyclone_atlas_sample.json"

    def is_available(self) -> bool:
        """Check if raw IMD JSON records exist on disk."""
        return self.raw_path.exists() and self.raw_path.is_file()

    def parse_raw_records(self) -> List[Dict[str, Any]]:
        """Parse raw IMD JSON array into dictionaries."""
        if not self.is_available():
            return []

        try:
            with open(self.raw_path, mode="r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []

    def normalize(self, data_mode: ScientificDataMode = ScientificDataMode.REAL_RESEARCH) -> List[NormalizedTrackRecord]:
        """Convert raw IMD entries to normalized Cyclone-X track records."""
        raw_rows = self.parse_raw_records()
        normalized: List[NormalizedTrackRecord] = []
        for r in raw_rows:
            norm_rec = ScientificDataNormalizer.normalize_imd_record(r, data_mode=data_mode)
            if norm_rec:
                normalized.append(norm_rec)
        return normalized

    def run_quality_control(self) -> DataQualityReport:
        """Run automated quality control on raw IMD data."""
        raw_rows = self.parse_raw_records()
        return ScientificQualityControl.audit_track_dataset(
            dataset_id="imd_best_track",
            dataset_name="IMD Historical Cyclone Best-Track Bulletins",
            records=raw_rows
        )

    def export_processed(self, output_dir: Optional[str] = None) -> str:
        """Export normalized records to data/processed/tracks/imd_normalized.json."""
        out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[4] / "data" / "processed" / "tracks"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "imd_normalized.json"

        norm_records = self.normalize()
        data = [r.model_dump(mode="json") for r in norm_records]
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return str(out_file)
