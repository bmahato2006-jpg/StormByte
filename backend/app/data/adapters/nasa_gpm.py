"""NASA GPM / IMERG Precipitation Data Source Adapter.

Handles:
- NASA Global Precipitation Measurement (GPM)
- Integrated Multi-satellitE Retrievals for GPM (IMERG) Early / Late / Final runs
- Gridded precipitation fluxes (mm/hr) at 0.1° x 0.1° (~10km) resolution

Follows Cyclone-X Scientific Integrity Rules:
- Requires valid NASA Earthdata Login credentials.
- Unconfigured state is exposed truthfully.
"""

import os
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional
from backend.app.data.adapters.base import DataSourceAdapter
from backend.app.data.schemas.observations import (
    DataSourceInfo,
    DataSourceStatus,
    DataMode,
    EnvironmentalObservation,
)


class NasaGpmAdapter(DataSourceAdapter):
    """Adapter for NASA GPM / IMERG precipitation retrievals."""

    def __init__(self):
        super().__init__(name="NASA_GPM_IMERG", provider="NASA / JAXA", requires_auth=True)
        self.username = os.getenv("NASA_GPM_USERNAME")
        self.password = os.getenv("NASA_GPM_PASSWORD")
        self.api_url = os.getenv("NASA_GPM_URL", "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3")
        self.supported_products = [
            "GPM_3IMERGHH_Early",
            "GPM_3IMERGHH_Late",
            "GPM_3IMERGHH_Final",
        ]

    def is_configured(self) -> bool:
        return bool(self.username and self.username.strip() and self.password and self.password.strip())

    def status(self) -> DataSourceStatus:
        if not self.is_configured():
            return DataSourceStatus.NOT_CONFIGURED
        return DataSourceStatus.ACTIVE

    def get_metadata(self) -> DataSourceInfo:
        configured = self.is_configured()
        return DataSourceInfo(
            name="NASA GPM / IMERG",
            provider="NASA Goddard Space Flight Center & JAXA",
            description=(
                "Global Precipitation Measurement calibrated multi-satellite precipitation estimates "
                "providing half-hourly rainfall rates at 0.1° spatial resolution."
            ),
            supported_products=self.supported_products,
            status=self.status(),
            data_mode=DataMode.LIVE_NRT if configured else DataMode.DEMO,
            configured=configured,
            requires_auth=True,
            last_checked=datetime.now(timezone.utc),
            notes=(
                "Requires NASA Earthdata Login account for GES DISC data access. "
                "Running in safe mode with zero fabrication."
                if not configured else "Earthdata credentials present."
            ),
        )

    def validate(self, raw_data: Any) -> bool:
        if not isinstance(raw_data, dict):
            return False
        required = ["timestamp", "latitude", "longitude", "rainfall"]
        return all(r in raw_data for r in required)

    def normalize(self, raw_data: Dict[str, Any], **kwargs) -> List[EnvironmentalObservation]:
        """Normalize IMERG precipitation data point into an EnvironmentalObservation."""
        if not self.validate(raw_data):
            raise ValueError("Invalid NASA GPM / IMERG rainfall record")

        obs = EnvironmentalObservation(
            id=str(raw_data.get("id", uuid.uuid4())),
            timestamp=datetime.fromisoformat(str(raw_data["timestamp"])),
            latitude=float(raw_data["latitude"]),
            longitude=float(raw_data["longitude"]),
            rainfall=float(raw_data["rainfall"]),
            source=f"NASA_GPM_{raw_data.get('product', 'IMERG_V07')}",
            data_mode=DataMode.LIVE_NRT if self.is_configured() else DataMode.DEMO,
        )
        return [obs]

