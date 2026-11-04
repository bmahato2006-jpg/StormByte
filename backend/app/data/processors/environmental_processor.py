from pathlib import Path
from typing import Any, Dict, Optional
from backend.app.data.adapters.era5_adapter import Era5Adapter
from backend.app.data.adapters.gpm_adapter import GpmAdapter
from backend.app.data.adapters.exposure_adapter import ExposureAdapter
from backend.app.data.paths import DATA_ROOT


class EnvironmentalProcessor:
    """Processes atmospheric, oceanic, precipitation, and demographic exposure records."""

    def __init__(self, data_root: Optional[str] = None):
        self.data_root = Path(data_root) if data_root else DATA_ROOT

    def process_all(self) -> Dict[str, Any]:
        """Export processed environmental, precipitation, and exposure catalogs."""
        results: Dict[str, Any] = {}

        era5 = Era5Adapter()
        results["era5_exported"] = era5.export_processed(str(self.data_root / "processed" / "environmental"))

        gpm = GpmAdapter()
        results["gpm_exported"] = gpm.export_processed(str(self.data_root / "processed" / "rainfall"))

        return results
