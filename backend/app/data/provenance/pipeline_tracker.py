"""Scientific data pipeline provenance chaining for Cyclone-X Phase 7.

Maintains an unbroken cryptographic chain of custody:
SOURCE_DATASET -> RAW_FILE -> NORMALIZATION -> QUALITY_CONTROL -> PROCESSED_DATASET -> MODEL_INPUT

Every transformation record carries:
- source
- dataType
- parentHash (SHA-256 of upstream artifact)
- dataHash (SHA-256 of current artifact)
- timestamp
- disclaimer
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional
from backend.app.core.provenance import SCIENTIFIC_DISCLAIMER, create_provenance
from backend.app.data.schemas.datasets import ScientificDataMode


def compute_file_sha256(file_path: str) -> Optional[str]:
    """Compute SHA-256 checksum of a file on disk."""
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_object_sha256(obj: Any) -> str:
    """Compute deterministic SHA-256 hash of a Python dict/object."""
    serialized = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class PipelineProvenanceTracker:
    """Orchestrates multi-stage scientific provenance chaining."""

    @staticmethod
    def create_transformation_provenance(
        stage: str,
        source: str,
        data_type: str,
        parent_hash: Optional[str] = None,
        data_hash: Optional[str] = None,
        data_mode: ScientificDataMode = ScientificDataMode.REAL_RESEARCH,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Creates an auditable provenance block for a pipeline transformation."""
        prov = create_provenance(
            source=source,
            processing_version=f"phase7-{stage.lower()}",
            model_version="Pipeline Preprocessing (No Model Inference)",
            data_type=data_type,
            parent_hash=parent_hash,
            data_hash=data_hash,
        )
        prov["stage"] = stage
        prov["dataMode"] = data_mode.value
        prov["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        if extra_metadata:
            prov["metadata"] = extra_metadata
        return prov

    @staticmethod
    def record_dataset_ingestion(
        dataset_id: str,
        source_name: str,
        provider: str,
        product_name: str,
        raw_file_path: Optional[str] = None,
        data_mode: ScientificDataMode = ScientificDataMode.REAL_RESEARCH,
    ) -> Dict[str, Any]:
        """Registers raw dataset ingestion into the persistent audit ledger."""
        raw_hash = compute_file_sha256(raw_file_path) if raw_file_path else None
        prov = PipelineProvenanceTracker.create_transformation_provenance(
            stage="RAW_INGESTION",
            source=source_name,
            data_type="RAW_SCIENTIFIC_DATASET",
            parent_hash=None,
            data_hash=raw_hash,
            data_mode=data_mode,
            extra_metadata={
                "dataset_id": dataset_id,
                "provider": provider,
                "product": product_name,
                "file_path": raw_file_path
            }
        )
        return prov

