"""Official NOAA/NCEI IBTrACS North Indian Ocean Ingestion Adapter."""

import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

from backend.app.data.best_track.normalizer import BestTrackNormalizer
from backend.app.data.best_track.schemas import BestTrackFix
from backend.app.data.best_track.source_registry import (
    IBTRACS_NI_DOWNLOAD_URL,
    IBTRACS_NI_FILENAME,
    RAW_IBTRACS_DIR,
)


def compute_sha256(path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class IbtracsBestTrackAdapter:
    """Ingests, parses, and standardizes official NOAA NCEI IBTrACS NI subset CSV files."""

    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = csv_path or (RAW_IBTRACS_DIR / IBTRACS_NI_FILENAME)

    def is_available(self) -> bool:
        """Returns True if the raw IBTrACS CSV archive is present locally."""
        return self.csv_path.exists() and self.csv_path.stat().st_size > 1_000_000

    def download(self, force: bool = False) -> Tuple[Path, str]:
        """Downloads the official IBTrACS NI archive from NOAA NCEI."""
        RAW_IBTRACS_DIR.mkdir(parents=True, exist_ok=True)
        if self.csv_path.exists() and not force:
            return self.csv_path, compute_sha256(self.csv_path)

        req = urllib.request.Request(
            IBTRACS_NI_DOWNLOAD_URL,
            headers={"User-Agent": "Cyclone-X-Research-Client/1.0 (NOAA-IBTrACS-Ingest)"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            content = resp.read()

        with open(self.csv_path, "wb") as f:
            f.write(content)

        return self.csv_path, compute_sha256(self.csv_path)

    def parse_all(self, target_years: Optional[List[int]] = None) -> List[BestTrackFix]:
        """Parses the NOAA IBTrACS NI dataset into canonical BestTrackFix records."""
        if not self.is_available():
            self.download()

        fixes: List[BestTrackFix] = []

        with open(self.csv_path, mode="r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header1 = next(reader, None)
            header2 = next(reader, None)  # Units row ("degrees_north", "kts", etc.)

            if not header1:
                return []

            # Column indices
            col_map = {name: idx for idx, name in enumerate(header1)}

            sid_idx = col_map.get("SID")
            name_idx = col_map.get("NAME")
            season_idx = col_map.get("SEASON")
            time_idx = col_map.get("ISO_TIME")
            lat_idx = col_map.get("LAT")
            lon_idx = col_map.get("LON")
            wmo_wind_idx = col_map.get("WMO_WIND")
            wmo_pres_idx = col_map.get("WMO_PRES")
            wmo_agency_idx = col_map.get("WMO_AGENCY")
            usa_wind_idx = col_map.get("USA_WIND")
            usa_pres_idx = col_map.get("USA_PRES")
            usa_agency_idx = col_map.get("USA_AGENCY")
            subbasin_idx = col_map.get("SUBBASIN")

            for row_idx, row in enumerate(reader):
                if not row or len(row) < 10:
                    continue

                # Filter unit rows or empty headers
                lat_str = row[lat_idx] if lat_idx is not None else ""
                if lat_str in ("degrees_north", "degrees_east", ""):
                    continue

                season_str = row[season_idx] if season_idx is not None else ""
                try:
                    year = int(season_str)
                except ValueError:
                    continue

                if target_years and year not in target_years:
                    continue

                storm_name = row[name_idx].strip().upper() if name_idx is not None else "UNNAMED"
                if storm_name == "NOT_NAMED":
                    storm_name = "UNNAMED"
                sid = row[sid_idx].strip() if sid_idx is not None else f"IBTRACS-{year}-{row_idx}"
                canonical_storm_id = f"IBTRACS-{year}-{storm_name}" if storm_name != "UNNAMED" else f"IBTRACS-{sid}"

                ts_str = row[time_idx].strip() if time_idx is not None else ""
                try:
                    lat_f = float(lat_str)
                    lon_f = float(row[lon_idx])
                except (ValueError, TypeError):
                    continue

                # Determine wind and pressure: WMO first, then USA/JTWC fallback
                wind_val: Optional[str] = None
                if wmo_wind_idx is not None and row[wmo_wind_idx].strip():
                    wind_val = row[wmo_wind_idx].strip()
                elif usa_wind_idx is not None and row[usa_wind_idx].strip():
                    wind_val = row[usa_wind_idx].strip()

                pres_val: Optional[str] = None
                if wmo_pres_idx is not None and row[wmo_pres_idx].strip():
                    pres_val = row[wmo_pres_idx].strip()
                elif usa_pres_idx is not None and row[usa_pres_idx].strip():
                    pres_val = row[usa_pres_idx].strip()

                agency = "WMO"
                if wmo_agency_idx is not None and row[wmo_agency_idx].strip():
                    agency = row[wmo_agency_idx].strip()
                elif usa_agency_idx is not None and row[usa_agency_idx].strip():
                    agency = row[usa_agency_idx].strip()

                subbasin = row[subbasin_idx].strip() if subbasin_idx is not None else None

                raw_dict = {
                    "storm_id": canonical_storm_id,
                    "storm_name": storm_name,
                    "basin": "NI",
                    "subbasin": subbasin,
                    "timestamp": ts_str,
                    "latitude": lat_f,
                    "longitude": lon_f,
                    "max_wind_kt": wind_val,
                    "central_pressure_hpa": pres_val,
                    "source_agency": agency,
                    "source_record_id": sid,
                }

                fix = BestTrackNormalizer.normalize_record(
                    raw_dict,
                    source="NOAA_IBTRACS",
                    source_file=self.csv_path.name,
                    source_url=IBTRACS_NI_DOWNLOAD_URL,
                )
                if fix:
                    fixes.append(fix)

        return fixes
