"""Satellite Processor for Cyclone-X Phase 7.

Validates satellite image rasters, assesses image quality and sharpness,
and serializes normalized metadata records without modifying raw files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.app.data.adapters.mosdac_adapter import MosdacAdapter
from backend.app.data.paths import DATA_ROOT
from backend.app.data.schemas import NormalizedSatelliteRecord, ScientificDataMode
from backend.app.data.validators import ScientificDataValidator


class SatelliteProcessor:
    """Processes, assesses, and indexes satellite granules."""

    def __init__(self, data_root: Optional[str] = None):
        self.data_root = Path(data_root) if data_root else DATA_ROOT
        self.processed_dir = self.data_root / "processed" / "satellite"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process_all_granules(self) -> Dict[str, Any]:
        """Process MOSDAC catalog and export normalized metadata."""
        mosdac = MosdacAdapter()
        norm_records = mosdac.normalize(data_mode=ScientificDataMode.REAL_RESEARCH)

        # Enhance each record with physical file verification if file exists
        for rec in norm_records:
            if rec.file_path:
                full_path = Path(rec.file_path)
                if not full_path.is_absolute():
                    full_path = self.data_root.parent / rec.file_path
                if full_path.exists():
                    valid, _, stats = ScientificDataValidator.validate_satellite_image_file(str(full_path))
                    rec.provenance["file_stats"] = stats

        out_file = self.processed_dir / "insat_normalized.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump([r.model_dump(mode="json") for r in norm_records], f, indent=2)

        return {
            "satellite_records": len(norm_records),
            "exported_file": str(out_file)
        }
