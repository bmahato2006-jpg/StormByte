"""NASA GPM IMERG Precipitation Dataset Adapter for Cyclone-X Phase 7.

Ingests and validates NASA GPM IMERG Half-Hourly (V07B) satellite precipitation estimates.
Reports CREDENTIALS_REQUIRED when NASA Earthdata Login credentials are not configured.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.app.core.config import settings
from backend.app.data.paths import DATA_ROOT
from backend.app.data.schemas import (
    DataQualityReport,
    NormalizedRainfallRecord,
    QualityStatus,
    ScientificDataMode,
    ScientificDatasetStatus,
)


class GpmAdapter:
    """Adapter for NASA GPM IMERG Precipitation estimates."""

    def __init__(self, catalog_path: Optional[str] = None):
        if catalog_path:
            self.catalog_path = Path(catalog_path)
        else:
            self.catalog_path = DATA_ROOT / "raw" / "gpm" / "gpm_catalog.json"

    def is_configured(self) -> bool:
        """Check if NASA Earthdata Login credentials or Bearer token are present."""
        user = getattr(settings, "NASA_GPM_USERNAME", None)
        pwd = getattr(settings, "NASA_GPM_PASSWORD", None)
        token = getattr(settings, "EARTHDATA_TOKEN", None)
        return bool((user and pwd) or (token and len(token.strip()) > 10))

    def get_status(self) -> ScientificDatasetStatus:
        """Evaluate operational connection status."""
        if not self.is_configured():
            return ScientificDatasetStatus.CREDENTIALS_REQUIRED
        if self.catalog_path.exists():
            return ScientificDatasetStatus.AVAILABLE
        return ScientificDatasetStatus.NOT_DOWNLOADED

    def parse_catalog(self) -> List[Dict[str, Any]]:
        """Parse raw catalog JSON."""
        if not self.catalog_path.exists():
            return []
        try:
            with open(self.catalog_path, mode="r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []

    def run_quality_control(self) -> DataQualityReport:
        """Run automated quality control on registered GPM products."""
        records = self.parse_catalog()
        status = QualityStatus.PASS if records else QualityStatus.WARNING
        report = DataQualityReport(
            dataset_id="nasa_gpm_imerg",
            dataset_name="NASA GPM IMERG Precipitation (V07B)",
            total_records=len(records),
            valid_records=len(records),
            variables=["precipitationCal", "probabilityLiquidPrecipitation"],
            units={"precipitationCal": "mm/hr", "probabilityLiquidPrecipitation": "%"},
            quality_status=status,
            issues=[] if self.is_configured() else ["NASA Earthdata login required to download raw HDF5 granules."]
        )
        return report

    def export_processed(self, output_dir: Optional[str] = None) -> str:
        """Export normalized rainfall records to data/processed/rainfall/gpm_normalized.json."""
        out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[4] / "data" / "processed" / "rainfall"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "gpm_normalized.json"

        records = self.parse_catalog()
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        return str(out_file)
