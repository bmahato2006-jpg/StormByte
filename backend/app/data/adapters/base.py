"""Abstract Base Data Source Adapter for Cyclone-X.

Defines the contract for external scientific data ingestion, verification,
and normalization.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from backend.app.data.schemas.observations import (
    DataSourceInfo,
    DataSourceStatus,
    DataMode,
)


class DataSourceAdapter(ABC):
    """Abstract interface for all scientific data source adapters."""

    def __init__(self, name: str, provider: str, requires_auth: bool = True):
        self.name = name
        self.provider = provider
        self.requires_auth = requires_auth

    @abstractmethod
    def get_metadata(self) -> DataSourceInfo:
        """Return operational metadata and current configuration status."""
        pass

    @abstractmethod
    def status(self) -> DataSourceStatus:
        """Evaluate provider connectivity or credential presence."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if necessary environment credentials or paths are present."""
        pass

    @abstractmethod
    def validate(self, raw_data: Any) -> bool:
        """Verify scientific integrity and structure of incoming data."""
        pass

    @abstractmethod
    def normalize(self, raw_data: Any, **kwargs) -> List[Any]:
        """Convert provider-specific payloads into Cyclone-X normalized schemas."""
        pass

    def fetch(self, **kwargs) -> Optional[Any]:
        """Fetch records from provider if authenticated.
        
        Zero-fabrication rule: If not configured, raise RuntimeError rather
        than generating fake live data.
        """
        if not self.is_configured():
            raise RuntimeError(
                f"Data source '{self.name}' ({self.provider}) is not configured with valid credentials. "
                "Per Cyclone-X Scientific Integrity Rules, live data cannot be simulated."
            )
        return None

