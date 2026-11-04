"""India Meteorological Department (IMD) Data Source Adapter.

Handles:
- Historical cyclone best-track archives (RSMC New Delhi)
- Official cyclone bulletins and RSMC warning formats

Follows Cyclone-X Scientific Integrity Rules:
- Never fabricates IMD warnings or official forecasts.
- Historical IMD information is strictly demarcated and NEVER displayed as a current warning.
- Experimental predictions remain strictly decoupled from official IMD bulletins.
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
    CycloneObservation,
)


class ImdAdapter(DataSourceAdapter):
    """Adapter for India Meteorological Department cyclone records and bulletins."""

    def __init__(self):
        super().__init__(name="IMD_RSMC", provider="India Meteorological Department", requires_auth=True)
        self.api_url = os.getenv("IMD_API_URL")
        self.api_key = os.getenv("IMD_API_KEY")
        self.supported_products = [
            "BEST_TRACK_ARCHIVE",
            "RSMC_TROPICAL_WEATHER_OUTLOOK",
            "SPECIAL_TROPICAL_CYCLONE_BULLETIN",
        ]

    def is_configured(self) -> bool:
        """Evaluates whether live IMD API / GTS bulletin feed credentials exist."""
        return bool(self.api_key and self.api_key.strip() and self.api_url and self.api_url.strip())

    def status(self) -> DataSourceStatus:
        if not self.is_configured():
            return DataSourceStatus.NOT_CONFIGURED
        return DataSourceStatus.ACTIVE

    def get_metadata(self) -> DataSourceInfo:
        configured = self.is_configured()
        return DataSourceInfo(
            name="IMD / RSMC New Delhi",
            provider="India Meteorological Department (Ministry of Earth Sciences)",
            description=(
                "Official Regional Specialized Meteorological Centre (RSMC) best tracks, "
                "synoptic charts, and cyclone warning bulletins for the North Indian Ocean."
            ),
            supported_products=self.supported_products,
            status=self.status(),
            data_mode=DataMode.HISTORICAL if not configured else DataMode.OFFICIAL_INFO,
            configured=configured,
            requires_auth=True,
            last_checked=datetime.now(timezone.utc),
            notes=(
                "Historical best tracks are active for validation. Real-time official bulletins "
                "require GTS or authorized IMD API credentials. Historical data is NEVER labeled as a current warning."
            ),
        )

    def validate(self, raw_data: Any) -> bool:
        if not isinstance(raw_data, dict):
            return False
        required = ["storm_id", "storm_name", "timestamp", "latitude", "longitude", "wind_speed", "central_pressure"]
        return all(r in raw_data for r in required)

    def normalize(self, raw_data: Dict[str, Any], is_historical: bool = True, **kwargs) -> List[CycloneObservation]:
        """Normalize IMD track records. Ensures data_mode is strictly HISTORICAL or OFFICIAL_INFO."""
        if not self.validate(raw_data):
            raise ValueError("Invalid IMD observation record format")

        mode = DataMode.HISTORICAL if is_historical else (
            DataMode.OFFICIAL_INFO if self.is_configured() else DataMode.DEMO
        )

        obs = CycloneObservation(
            id=str(raw_data.get("id", uuid.uuid4())),
            storm_id=str(raw_data["storm_id"]),
            storm_name=str(raw_data["storm_name"]),
            timestamp=datetime.fromisoformat(str(raw_data["timestamp"])),
            latitude=float(raw_data["latitude"]),
            longitude=float(raw_data["longitude"]),
            wind_speed=float(raw_data["wind_speed"]),
            central_pressure=float(raw_data["central_pressure"]),
            classification=str(raw_data.get("classification", "Cyclonic Storm")),
            source="IMD_RSMC_NEW_DELHI",
            source_reference=raw_data.get("source_reference", "IMD Best Track Archive"),
            data_mode=mode,
        )
        return [obs]

