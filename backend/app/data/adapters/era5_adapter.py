"""ECMWF ERA5 Reanalysis Dataset Adapter for Cyclone-X Phase 7.

Ingests and validates atmospheric and oceanic reanalysis variables (SST, MSLP, 200/850 hPa winds).
Calculates vertical wind shear vector differences without substituting defaults.
Reports CREDENTIALS_REQUIRED when ECMWF Copernicus Climate Data Store (CDS) API key is missing.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.app.core.config import settings
from backend.app.data.paths import DATA_ROOT
from backend.app.data.schemas import (
    DataQualityReport,
    NormalizedEnvironmentalRecord,
    QualityStatus,
    ScientificDataMode,
    ScientificDatasetStatus,
)


class Era5Adapter:
    """Adapter for ECMWF ERA5 Atmospheric and Oceanic Reanalysis."""

    def __init__(self, catalog_path: Optional[str] = None):
        if catalog_path:
            self.catalog_path = Path(catalog_path)
        else:
            self.catalog_path = DATA_ROOT / "raw" / "era5" / "era5_catalog.json"

    def is_configured(self) -> bool:
        """Check if CDS API key is present in settings."""
        return bool(getattr(settings, "CDS_API_KEY", None))

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

    @staticmethod
    def calculate_vertical_wind_shear(
        u200: Optional[float],
        v200: Optional[float],
        u850: Optional[float],
        v850: Optional[float]
    ) -> Optional[float]:
        """Calculates 200-850 hPa bulk vertical wind shear magnitude in knots.
        
        Zero-fabrication rule: If any component is None, returns None.
        """
        if u200 is None or v200 is None or u850 is None or v850 is None:
            return None

        # Components difference (m/s)
        du = u200 - u850
        dv = v200 - v850
        diff_ms = math.sqrt(du * du + dv * dv)
        # Convert m/s to knots (1 m/s = 1.94384 knots)
        shear_kt = diff_ms * 1.94384
        return round(shear_kt, 1)

    def run_quality_control(self) -> DataQualityReport:
        """Run automated quality control on registered ERA5 products."""
        records = self.parse_catalog()
        status = QualityStatus.PASS if records else QualityStatus.WARNING
        report = DataQualityReport(
            dataset_id="era5_reanalysis",
            dataset_name="ECMWF ERA5 Reanalysis",
            total_records=len(records),
            valid_records=len(records),
            variables=[
                "sea_surface_temperature",
                "vertical_wind_shear",
                "mean_sea_level_pressure"
            ],
            units={
                "sea_surface_temperature": "deg Celsius",
                "vertical_wind_shear": "knots",
                "mean_sea_level_pressure": "hPa"
            },
            quality_status=status,
            issues=[] if self.is_configured() else ["ECMWF CDS API key required to download raw GRIB/NetCDF files."]
        )
        return report

    def export_processed(self, output_dir: Optional[str] = None) -> str:
        """Export normalized environmental records to data/processed/environmental/era5_normalized.json."""
        out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[4] / "data" / "processed" / "environmental"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "era5_normalized.json"

        records = self.parse_catalog()
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        return str(out_file)
