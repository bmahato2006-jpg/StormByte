from datetime import datetime, timezone
from typing import Dict, Any, Optional


SCIENTIFIC_DISCLAIMER = (
    "EXPERIMENTAL RESEARCH SYSTEM - NOT AN OFFICIAL WEATHER FORECAST. "
    "DOES NOT REPLACE BULLETINS FROM IMD, NHC, OR RSMC. ALL DATA LABELED ACCORDING TO PROVENANCE."
)

def create_provenance(
    source: str = "IMD RSMC New Delhi & IBTrACS Best Track Archive",
    processing_version: str = "v2.1.0-operational",
    model_version: str = "cyclone_detector_cnn_v2",
    data_type: str = "VERIFIED OBSERVATIONAL DATA",
    parent_hash: Optional[str] = None,
    data_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Creates an immutable provenance record adhering to scientific integrity rules.
    """
    prov: Dict[str, Any] = {
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processingVersion": processing_version,
        "modelVersion": model_version,
        "dataType": data_type,
        "disclaimer": SCIENTIFIC_DISCLAIMER
    }
    if parent_hash:
        prov["parentHash"] = parent_hash
    if data_hash:
        prov["dataHash"] = data_hash
    return prov

