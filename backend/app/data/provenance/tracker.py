"""Scientific Data Provenance Tracker.

Maintains an immutable, auditable log of all scientific data sources,
ingestion runs, user uploads, and demo scenarios.

Follows Cyclone-X Scientific Integrity Rules:
- Real-time/live data is never simulated.
- Every record has explicit provenance, data mode, and acquisition timestamps.
"""

from datetime import datetime, timezone
import uuid
from typing import Dict, List, Optional
from backend.app.data.schemas.observations import DataMode, DataProvenance


class ProvenanceTracker:
    """Manages provenance audit logs for all scientific datasets."""

    def __init__(self):
        self._records: List[DataProvenance] = []
        self._initialize_baseline_provenance()

    def _initialize_baseline_provenance(self):
        """Seed initial provenance records for verified demo and baseline datasets."""
        now = datetime.now(timezone.utc)
        self._records = [
            DataProvenance(
                id="prov-amphan-2020-baseline",
                source="IMD_BEST_TRACK",
                provider="India Meteorological Department (RSMC New Delhi)",
                product="Super Cyclonic Storm AMPHAN (BOB 01/2020) Best Track Archive",
                dataset_version="IMD-2020.1",
                acquisition_time=datetime(2020, 5, 16, 0, 0, tzinfo=timezone.utc),
                processing_time=now,
                source_reference="IMD Cyclone e-Atlas / RSMC Bulletin Archives",
                license_notes="Public Meteorological Data for Research & Education",
                data_mode=DataMode.DEMO,
                status="VERIFIED_ARCHIVE",
            ),
            DataProvenance(
                id="prov-biparjoy-2023-baseline",
                source="IMD_BEST_TRACK",
                provider="India Meteorological Department (RSMC New Delhi)",
                product="Extremely Severe Cyclonic Storm BIPARJOY (ARB 01/2023) Best Track Archive",
                dataset_version="IMD-2023.2",
                acquisition_time=datetime(2023, 6, 6, 6, 0, tzinfo=timezone.utc),
                processing_time=now,
                source_reference="RSMC Report No. 03(02)/2023-CW",
                license_notes="Public Meteorological Data for Research & Education",
                data_mode=DataMode.DEMO,
                status="VERIFIED_ARCHIVE",
            ),
            DataProvenance(
                id="prov-mosdac-insat-registry",
                source="MOSDAC_ISRO",
                provider="Space Applications Centre, ISRO",
                product="INSAT-3D/3DR Meteorological Satellite Payloads",
                dataset_version="MOSDAC-V3",
                acquisition_time=None,
                processing_time=now,
                source_reference="https://mosdac.gov.in",
                license_notes="ISRO Open Data Policy (Registration Required)",
                data_mode=DataMode.DEMO,
                status="NOT_CONFIGURED",
            ),
            DataProvenance(
                id="prov-nasa-gpm-registry",
                source="NASA_GPM_IMERG",
                provider="NASA / GES DISC",
                product="GPM IMERG Early/Late/Final Half-Hourly Precipitation",
                dataset_version="V07B",
                acquisition_time=None,
                processing_time=now,
                source_reference="DOI: 10.5067/GPM/IMERG/3B-HH/07",
                license_notes="NASA Open Science Data Policy",
                data_mode=DataMode.DEMO,
                status="NOT_CONFIGURED",
            ),
        ]

    def record_ingestion(
        self,
        source: str,
        provider: str,
        product: str,
        data_mode: DataMode,
        status: str,
        source_reference: Optional[str] = None,
        dataset_version: str = "1.0",
        acquisition_time: Optional[datetime] = None,
        license_notes: Optional[str] = "Open Scientific Data"
    ) -> DataProvenance:
        """Append a newly verified ingestion run to the provenance ledger."""
        rec = DataProvenance(
            id=f"prov-{uuid.uuid4().hex[:12]}",
            source=source,
            provider=provider,
            product=product,
            dataset_version=dataset_version,
            acquisition_time=acquisition_time,
            processing_time=datetime.now(timezone.utc),
            source_reference=source_reference,
            license_notes=license_notes,
            data_mode=data_mode,
            status=status,
        )
        self._records.insert(0, rec)  # Newest first
        return rec

    def get_all(self, limit: int = 100) -> List[DataProvenance]:
        """Return all provenance records ordered from newest to oldest."""
        return self._records[:limit]

    def get_by_mode(self, data_mode: DataMode) -> List[DataProvenance]:
        """Filter provenance records by scientific data mode."""
        return [r for r in self._records if r.data_mode == data_mode]

    def count(self) -> int:
        return len(self._records)


provenance_tracker = ProvenanceTracker()

