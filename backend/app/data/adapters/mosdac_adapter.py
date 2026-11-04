"""MOSDAC / ISRO Satellite Dataset Adapter for Cyclone-X Phase 7.

Ingests and validates INSAT-3D, INSAT-3DR, and INSAT-3DS meteorological imagery.
Reports CREDENTIALS_REQUIRED when ISRO MOSDAC API credentials are not configured.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.app.core.config import settings
from backend.app.data.normalizer import ScientificDataNormalizer
from backend.app.data.paths import DATA_ROOT
from backend.app.data.quality_control import ScientificQualityControl
from backend.app.data.schemas import (
    DataQualityReport,
    NormalizedSatelliteRecord,
    ScientificDataMode,
    ScientificDatasetStatus,
)


class MosdacAdapter:
    """Adapter for reading and processing MOSDAC / INSAT satellite payloads."""

    def __init__(self, catalog_path: Optional[str] = None):
        if catalog_path:
            self.catalog_path = Path(catalog_path)
        else:
            self.catalog_path = DATA_ROOT / "raw" / "mosdac" / "insat_catalog.json"

    def is_configured(self) -> bool:
        """Check if MOSDAC SSO credentials are provided in settings."""
        has_sso = bool(getattr(settings, "MOSDAC_USERNAME", None) and getattr(settings, "MOSDAC_PASSWORD", None))
        has_key = bool(getattr(settings, "MOSDAC_API_KEY", None))
        return has_sso or has_key

    def get_status(self) -> ScientificDatasetStatus:
        """Evaluate operational connection status."""
        if self.catalog_path.exists():
            return ScientificDatasetStatus.AVAILABLE
        if not self.is_configured():
            return ScientificDatasetStatus.CREDENTIALS_REQUIRED
        return ScientificDatasetStatus.NOT_DOWNLOADED

    def parse_catalog(self) -> List[Dict[str, Any]]:
        """Parse raw catalog JSON array."""
        if not self.catalog_path.exists():
            return []

        try:
            with open(self.catalog_path, mode="r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []

    def normalize(self, data_mode: ScientificDataMode = ScientificDataMode.REAL_RESEARCH) -> List[NormalizedSatelliteRecord]:
        """Convert raw catalog entries to normalized Cyclone-X satellite records."""
        granules = self.parse_catalog()
        normalized: List[NormalizedSatelliteRecord] = []
        for g in granules:
            norm_rec = ScientificDataNormalizer.normalize_mosdac_granule(g, data_mode=data_mode)
            if norm_rec:
                normalized.append(norm_rec)
        return normalized

    def run_quality_control(self) -> DataQualityReport:
        """Run automated quality control on registered satellite granules."""
        granules = self.parse_catalog()
        return ScientificQualityControl.audit_satellite_dataset(
            dataset_id="mosdac_insat",
            dataset_name="MOSDAC / ISRO INSAT Satellite Payloads",
            granules=granules
        )

    def export_processed(self, output_dir: Optional[str] = None) -> str:
        """Export normalized satellite metadata to data/processed/satellite/insat_normalized.json."""
        out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[4] / "data" / "processed" / "satellite"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "insat_normalized.json"

        norm_records = self.normalize()
        data = [r.model_dump(mode="json") for r in norm_records]
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return str(out_file)
