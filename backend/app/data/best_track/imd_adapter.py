"""Official IMD RSMC New Delhi Best Track Ingestion Adapter."""

from datetime import datetime, time, timezone
import hashlib
import os
from pathlib import Path
import re
import ssl
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
import urllib.request
import pandas as pd

from backend.app.data.best_track.normalizer import BestTrackNormalizer
from backend.app.data.best_track.schemas import BestTrackFix
from backend.app.data.best_track.source_registry import (
    IMD_EXCEL_DOWNLOAD_URL,
    IMD_EXCEL_FILENAME,
    RAW_IMD_DIR,
)


def compute_sha256(path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ImdBestTrackAdapter:
    """Ingests, parses, and standardizes official IMD RSMC Best Track Excel datasets (1982–2026)."""

    def __init__(self, excel_path: Optional[Path] = None):
        self.excel_path = excel_path or (RAW_IMD_DIR / IMD_EXCEL_FILENAME)

    def is_available(self) -> bool:
        """Returns True if the raw IMD Excel archive is present locally."""
        return self.excel_path.exists() and self.excel_path.stat().st_size > 1_000_000

    def download(self, force: bool = False) -> Tuple[Path, str]:
        """Downloads the official IMD Best Track Excel archive from RSMC New Delhi."""
        RAW_IMD_DIR.mkdir(parents=True, exist_ok=True)
        if self.excel_path.exists() and not force:
            return self.excel_path, compute_sha256(self.excel_path)

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(
            IMD_EXCEL_DOWNLOAD_URL,
            headers={"User-Agent": "Cyclone-X-Research-Client/1.0 (MoES-IMD Scientific Assessment)"},
        )
        with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
            content = resp.read()

        with open(self.excel_path, "wb") as f:
            f.write(content)

        return self.excel_path, compute_sha256(self.excel_path)

    @staticmethod
    def _find_header_row(df_raw: pd.DataFrame) -> int:
        """Dynamically identifies the row index containing the column headers."""
        keywords = ["serial", "latitude", "basin", "date", "name", "grade"]
        for idx in range(min(15, len(df_raw))):
            row_str = " ".join(str(x).lower() for x in df_raw.iloc[idx].values)
            matches = sum(1 for k in keywords if k in row_str)
            if matches >= 2:
                return idx
        return 0

    @staticmethod
    def _parse_time_string(val: Any) -> Tuple[int, int]:
        """Parses UTC time field into (hour, minute). Handles '0300', 300, '03:00', etc."""
        if pd.isna(val):
            return 0, 0
        s = str(val).strip()
        if ":" in s:
            parts = s.split(":")
            return int(parts[0]), int(parts[1])
        # Integer or 4-char string representation like '300' or '0300' or '1200'
        digits = re.sub(r"\D", "", s)
        if not digits:
            return 0, 0
        padded = digits.zfill(4)
        hh = int(padded[:2])
        mm = int(padded[2:4])
        return min(23, max(0, hh)), min(59, max(0, mm))

    @classmethod
    def _parse_date_and_time(cls, date_val: Any, time_val: Any) -> Optional[datetime]:
        """Combines IMD Date and Time cells into a timezone-aware UTC datetime."""
        if pd.isna(date_val):
            return None

        hh, mm = cls._parse_time_string(time_val)

        if isinstance(date_val, datetime):
            return datetime(
                date_val.year, date_val.month, date_val.day, hh, mm, tzinfo=timezone.utc
            )
        elif isinstance(date_val, pd.Timestamp):
            return datetime(
                date_val.year, date_val.month, date_val.day, hh, mm, tzinfo=timezone.utc
            )

        # String parsing
        s = str(date_val).strip().split()[0]
        # Common formats: DD-MM-YYYY, DD/MM/YYYY, YYYY-MM-DD
        for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y"):
            try:
                dt = datetime.strptime(s, fmt)
                return datetime(dt.year, dt.month, dt.day, hh, mm, tzinfo=timezone.utc)
            except ValueError:
                continue

        return None

    def parse_sheet(self, sheet_name: str, xl: pd.ExcelFile) -> List[Dict[str, Any]]:
        """Parses an individual annual sheet from the IMD Best Track workbook."""
        df_raw = xl.parse(sheet_name, header=None)
        if len(df_raw) < 2:
            return []

        header_idx = self._find_header_row(df_raw)
        headers = [str(x).strip() for x in df_raw.iloc[header_idx].values]
        df_data = df_raw.iloc[header_idx + 1:].copy()
        df_data.columns = headers

        # Locate key columns by lower-case substring
        col_map: Dict[str, Optional[str]] = {
            "serial": None, "basin": None, "name": None, "date": None,
            "time": None, "lat": None, "lon": None, "ci": None,
            "pres": None, "wind": None, "dp": None, "grade": None,
        }

        for c in headers:
            cl = c.lower()
            if "serial" in cl and not col_map["serial"]:
                col_map["serial"] = c
            elif "basin" in cl and not col_map["basin"]:
                col_map["basin"] = c
            elif "name" in cl and not col_map["name"]:
                col_map["name"] = c
            elif "date" in cl and not col_map["date"]:
                col_map["date"] = c
            elif "time" in cl and not col_map["time"]:
                col_map["time"] = c
            elif "lat" in cl and not col_map["lat"]:
                col_map["lat"] = c
            elif "lon" in cl and not col_map["lon"]:
                col_map["lon"] = c
            elif ("ci" in cl or "t. no" in cl or "t.no" in cl) and not col_map["ci"]:
                col_map["ci"] = c
            elif ("e.c.p" in cl or "pressure" in cl or "pres" in cl) and not col_map["pres"]:
                col_map["pres"] = c
            elif ("wind" in cl or "surface wind" in cl) and not col_map["wind"]:
                col_map["wind"] = c
            elif ("drop" in cl or "delta p" in cl) and not col_map["dp"]:
                col_map["dp"] = c
            elif "grade" in cl and not col_map["grade"]:
                col_map["grade"] = c

        raw_fixes: List[Dict[str, Any]] = []

        current_serial: str = "1"
        current_basin: str = "NI"
        current_name: str = "UNNAMED"
        current_date_val: Any = None

        for row_idx, row in df_data.iterrows():
            # Update serial if present
            if col_map["serial"] and pd.notna(row.get(col_map["serial"])):
                s_val = str(row[col_map["serial"]]).strip().replace(".0", "")
                if s_val and s_val.isdigit():
                    current_serial = s_val

            # Update basin if present
            if col_map["basin"] and pd.notna(row.get(col_map["basin"])):
                b_val = str(row[col_map["basin"]]).strip().upper()
                if b_val in ("BOB", "AS", "BAY OF BENGAL", "ARABIAN SEA", "LAND"):
                    current_basin = "BOB" if "B" in b_val else ("AS" if "A" in b_val else "NI")

            # Update storm name if present
            if col_map["name"] and pd.notna(row.get(col_map["name"])):
                n_val = str(row[col_map["name"]]).strip().upper()
                # Exclude header artifacts or summary strings
                if n_val and not any(k in n_val for k in ["NAME", "NAN", "NONE", "WEAKENED", "DISSIPATED"]):
                    current_name = n_val

            # Update date if present
            if col_map["date"] and pd.notna(row.get(col_map["date"])):
                current_date_val = row[col_map["date"]]

            # Latitude & Longitude
            lat_raw = row.get(col_map["lat"]) if col_map["lat"] else None
            lon_raw = row.get(col_map["lon"]) if col_map["lon"] else None

            if pd.isna(lat_raw) or pd.isna(lon_raw):
                continue

            try:
                lat_str = re.sub(r"[^0-9.]", "", str(lat_raw))
                lon_str = re.sub(r"[^0-9.]", "", str(lon_raw))
                lat = float(lat_str)
                lon = float(lon_str)
            except (ValueError, TypeError):
                continue

            # Parse Timestamp
            time_raw = row.get(col_map["time"]) if col_map["time"] else None
            ts = self._parse_date_and_time(current_date_val, time_raw)
            if not ts:
                # Fallback to sheet year
                try:
                    yr = int(re.sub(r"\D", "", sheet_name))
                    ts = datetime(yr, 1, 1, 0, 0, tzinfo=timezone.utc)
                except Exception:
                    continue

            # Wind, Pressure, CI, Grade
            wind_raw = row.get(col_map["wind"]) if col_map["wind"] else None
            pres_raw = row.get(col_map["pres"]) if col_map["pres"] else None
            ci_raw = row.get(col_map["ci"]) if col_map["ci"] else None
            dp_raw = row.get(col_map["dp"]) if col_map["dp"] else None
            grade_raw = row.get(col_map["grade"]) if col_map["grade"] else None

            # Construct unique storm ID: e.g. IMD-2020-AMPHAN or IMD-2020-BOB-01
            storm_id = f"IMD-{ts.year}-{current_name}" if current_name != "UNNAMED" else f"IMD-{ts.year}-{current_basin}-{current_serial}"

            raw_fixes.append({
                "storm_id": storm_id,
                "storm_name": current_name,
                "basin": current_basin,
                "timestamp": ts.isoformat(),
                "latitude": lat,
                "longitude": lon,
                "max_wind_kt": wind_raw,
                "central_pressure_hpa": pres_raw,
                "ci_number": ci_raw,
                "pressure_drop_hpa": dp_raw,
                "grade": grade_raw,
                "source": "IMD_RSMC_NEW_DELHI",
                "source_agency": "IMD",
                "source_record_id": f"{sheet_name}:{row_idx}",
                "source_file": self.excel_path.name,
                "year": ts.year,
            })

        return raw_fixes

    def parse_all(self, target_years: Optional[List[int]] = None) -> Tuple[List[BestTrackFix], List[Dict[str, Any]]]:
        """Parses all (or filtered) years from the IMD workbook into canonical BestTrackFix records."""
        if not self.is_available():
            self.download()

        xl = pd.ExcelFile(self.excel_path)
        sheet_names = xl.sheet_names

        all_fixes: List[BestTrackFix] = []
        manifest_rows: List[Dict[str, Any]] = []

        file_hash = compute_sha256(self.excel_path)
        download_time = datetime.now(timezone.utc).isoformat()

        for s in sheet_names:
            digits = re.sub(r"\D", "", s)
            if not digits:
                continue
            year = int(digits)
            if target_years and year not in target_years:
                continue

            try:
                raw_rows = self.parse_sheet(s, xl)
                sheet_fixes: List[BestTrackFix] = []
                for r in raw_rows:
                    f = BestTrackNormalizer.normalize_record(
                        r,
                        source="IMD_RSMC_NEW_DELHI",
                        source_file=self.excel_path.name,
                        source_url=IMD_EXCEL_DOWNLOAD_URL,
                    )
                    if f:
                        sheet_fixes.append(f)

                all_fixes.extend(sheet_fixes)
                manifest_rows.append({
                    "year": year,
                    "source": "IMD_RSMC_NEW_DELHI",
                    "original_filename": self.excel_path.name,
                    "local_path": str(self.excel_path),
                    "source_reference": f"RSMC New Delhi Best Track Archive Sheet {s}",
                    "sha256": file_hash,
                    "download_timestamp": download_time,
                    "parse_status": "SUCCESS",
                    "record_count": len(sheet_fixes),
                })
            except Exception as e:
                manifest_rows.append({
                    "year": year,
                    "source": "IMD_RSMC_NEW_DELHI",
                    "original_filename": self.excel_path.name,
                    "local_path": str(self.excel_path),
                    "source_reference": f"RSMC New Delhi Sheet {s}",
                    "sha256": file_hash,
                    "download_timestamp": download_time,
                    "parse_status": f"ERROR: {str(e)}",
                    "record_count": 0,
                })

        return all_fixes, manifest_rows
