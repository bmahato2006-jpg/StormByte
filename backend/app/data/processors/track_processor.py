"""Track Processor for Cyclone-X Phase 7.

Orchestrates loading raw best-track datasets (IBTrACS, IMD),
validating continuity, normalizing fields, and exporting processed datasets.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.app.data.adapters.ibtracs_adapter import IbtracsAdapter
from backend.app.data.adapters.imd_adapter import ImdAdapter
from backend.app.data.paths import DATA_ROOT
from backend.app.data.schemas import NormalizedTrackRecord, ScientificDataMode
from backend.app.data.validators import ScientificDataValidator


class TrackProcessor:
    """Processes, validates, and serializes best-track cyclone records."""

    def __init__(self, data_root: Optional[str] = None):
        self.data_root = Path(data_root) if data_root else DATA_ROOT
        self.processed_dir = self.data_root / "processed" / "tracks"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process_all_tracks(self) -> Dict[str, Any]:
        """Process both IBTrACS and IMD track datasets and save to processed directory."""
        results: Dict[str, Any] = {
            "ibtracs_records": 0,
            "imd_records": 0,
            "exported_files": []
        }

        # 1. Process IBTrACS
        ibtracs = IbtracsAdapter()
        if ibtracs.is_available():
            norm_ibtracs = ibtracs.normalize(data_mode=ScientificDataMode.REAL_RESEARCH)
            out_ibtracs = self.processed_dir / "ibtracs_normalized.json"
            with open(out_ibtracs, "w", encoding="utf-8") as f:
                json.dump([r.model_dump(mode="json") for r in norm_ibtracs], f, indent=2)
            results["ibtracs_records"] = len(norm_ibtracs)
            results["exported_files"].append(str(out_ibtracs))

        # 2. Process IMD
        imd = ImdAdapter()
        if imd.is_available():
            norm_imd = imd.normalize(data_mode=ScientificDataMode.REAL_RESEARCH)
            out_imd = self.processed_dir / "imd_normalized.json"
            with open(out_imd, "w", encoding="utf-8") as f:
                json.dump([r.model_dump(mode="json") for r in norm_imd], f, indent=2)
            results["imd_records"] = len(norm_imd)
            results["exported_files"].append(str(out_imd))

        return results

    def get_storm_track(self, storm_name_or_id: str) -> List[NormalizedTrackRecord]:
        """Retrieve processed track for a specific storm from disk."""
        target = storm_name_or_id.strip().upper()
        matching: List[NormalizedTrackRecord] = []

        for fn in ["ibtracs_normalized.json", "imd_normalized.json"]:
            fp = self.processed_dir / fn
            if not fp.exists():
                continue
            with open(fp, "r", encoding="utf-8") as f:
                records = json.load(f)
                for r in records:
                    s_name = str(r.get("storm_name", "")).upper()
                    s_id = str(r.get("storm_id", "")).upper()
                    if target in s_name or target in s_id:
                        matching.append(NormalizedTrackRecord(**r))

        # Sort by timestamp
        matching.sort(key=lambda x: x.timestamp)
        return matching
