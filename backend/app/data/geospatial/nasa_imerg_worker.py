"""Production-Ready Asynchronous NASA GPM IMERG Ingestion Worker for Cyclone-X.

Official Documentation: https://disc.gsfc.nasa.gov/information/documents
CMR API Reference: https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html
Dataset: GPM_3IMERGHHL (Late Run) & GPM_3IMERGHHE (Early Run)

Capabilities:
1. Authentication & Config:
   - Secure class EarthdataAuthConfig reading EARTHDATA_USERNAME, EARTHDATA_PASSWORD,
     and EARTHDATA_TOKEN from environment variables and .env.
   - Configures async authentication sessions with NASA GES DISC and Earthdata Login (URS).
   - Manages automatic .netrc / _netrc creation for cross-library compatibility.
   - Masks sensitive secrets in logs and string representations.
2. Target Dataset & Querying:
   - Queries NASA CMR public catalog for latest NetCDF4 / HDF5 granules in 4 to 12-hour windows.
   - Targets GPM_3IMERGHHL (Late Run) with automatic fallback to GPM_3IMERGHHE (Early Run).
   - Spatial bounding box targeting the North Indian Ocean basin ([-10, 38]°N, [45, 110]°E).
3. NetCDF4 / HDF5 Processing:
   - Downloads authentic raw granules into `data/raw/precipitation/`.
   - Validates HDF5 magic bytes (\x89HDF\r\n\x1a\n) and structural integrity.
   - Extracts `precipitationCal` (legacy V06/V07B) or `precipitation` (V07C).
   - Subsets strictly to the North Indian Ocean coordinates.
   - Cleans sentinel fill values (-9999.9 and NaNs) to 0.0 mm/hr.
   - Normalizes into a PyTorch-ready float32 tensor of shape [1, H, W] ([1, 480, 650]).
4. Guardrails & Zero Fake Data:
   - Never fabricates synthetic rain rates or mock storms if NASA servers are offline or unconfigured.
   - Returns empty tensors gracefully with truthful operational status.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import io
import logging
import os
from pathlib import Path
import re
import stat
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import h5py
import httpx
import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logger = logging.getLogger("CycloneX.NASAIMERGWorker")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Custom Exceptions
# -----------------------------------------------------------------------------
class EarthdataBaseException(Exception):
    """Base exception for all NASA Earthdata / IMERG worker errors."""


class EarthdataAuthError(EarthdataBaseException):
    """Raised when NASA Earthdata authentication fails or credentials are rejected."""


class EarthdataQueryError(EarthdataBaseException):
    """Raised when NASA CMR or GES DISC catalog search fails or times out."""


class EarthdataDownloadError(EarthdataBaseException):
    """Raised when IMERG granule download fails or is corrupted."""


class EarthdataProcessingError(EarthdataBaseException):
    """Raised when HDF5 / NetCDF parsing or PyTorch extraction fails."""


# -----------------------------------------------------------------------------
# Domain Schemas
# -----------------------------------------------------------------------------
class GeoBounds(BaseModel):
    """Geographic bounding box for meteorological sector querying."""

    lat_min: float = Field(-10.0, ge=-90.0, le=90.0, description="Minimum latitude (degrees)")
    lat_max: float = Field(38.0, ge=-90.0, le=90.0, description="Maximum latitude (degrees)")
    lon_min: float = Field(45.0, ge=-180.0, le=180.0, description="Minimum longitude (degrees)")
    lon_max: float = Field(110.0, ge=-180.0, le=180.0, description="Maximum longitude (degrees)")

    @field_validator("lat_max")
    @classmethod
    def validate_lat_range(cls, v: float, info: Any) -> float:
        lat_min = info.data.get("lat_min")
        if lat_min is not None and v <= lat_min:
            raise ValueError(f"lat_max ({v}) must be greater than lat_min ({lat_min})")
        return v

    @field_validator("lon_max")
    @classmethod
    def validate_lon_range(cls, v: float, info: Any) -> float:
        lon_min = info.data.get("lon_min")
        if lon_min is not None and v <= lon_min:
            raise ValueError(f"lon_max ({v}) must be greater than lon_min ({lon_min})")
        return v

    @property
    def cmr_bounding_box(self) -> str:
        """Returns standard CMR bounding box: lower_left_lon,lower_left_lat,upper_right_lon,upper_right_lat."""
        return f"{self.lon_min:.2f},{self.lat_min:.2f},{self.lon_max:.2f},{self.lat_max:.2f}"


class IMERGGranuleMetadata(BaseModel):
    """Metadata representing a single NASA GPM IMERG precipitation granule."""

    granule_id: str = Field(..., description="Unique product granule identifier")
    dataset_short_name: str = Field("GPM_3IMERGHHL", description="NASA short name (GPM_3IMERGHHL or GPM_3IMERGHHE)")
    run_type: str = Field("late", description="IMERG run type: late or early")
    version: str = Field("07", description="Product algorithm version (e.g. 07, V07C)")
    start_time_utc: datetime = Field(..., description="Observation period start UTC")
    end_time_utc: datetime = Field(..., description="Observation period end UTC")
    download_url: str = Field(..., description="Authenticated GES DISC download URL")
    file_size_bytes: int = Field(0, ge=0, description="Granule file size in bytes")
    checksum_md5: Optional[str] = Field(None, description="Granule MD5 digest if provided")
    local_path: Optional[str] = Field(None, description="Local file path if downloaded")


# -----------------------------------------------------------------------------
# Configuration & Earthdata Auth
# -----------------------------------------------------------------------------
class EarthdataAuthConfig(BaseModel):
    """Secure configuration container for NASA Earthdata credentials and endpoints."""

    username: str = Field(default="", description="NASA Earthdata Login Username")
    password: str = Field(default="", description="NASA Earthdata Login Password")
    token: str = Field(default="", description="NASA Earthdata Bearer Token")
    cmr_base_url: str = Field(default="https://cmr.earthdata.nasa.gov", description="NASA CMR Search API URL")
    gesdisc_base_url: str = Field(default="https://data.gesdisc.earthdata.nasa.gov", description="GES DISC Data Server")
    dataset_short_name: str = Field(default="GPM_3IMERGHHL", description="Default dataset short name")
    download_dir: Path = Field(
        default_factory=lambda: Path("data/raw/precipitation"),
        description="Local directory for temporary raw IMERG files",
    )
    request_timeout_seconds: float = Field(default=45.0, ge=5.0, le=180.0)

    @classmethod
    def from_environment(cls) -> EarthdataAuthConfig:
        """Constructs configuration safely from system environment variables and .env."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        return cls(
            username=os.getenv("EARTHDATA_USERNAME", "").strip(),
            password=os.getenv("EARTHDATA_PASSWORD", "").strip(),
            token=os.getenv("EARTHDATA_TOKEN", "").strip(),
            cmr_base_url=os.getenv("EARTHDATA_CMR_URL", "https://cmr.earthdata.nasa.gov").rstrip("/"),
            gesdisc_base_url=os.getenv("EARTHDATA_GESDISC_URL", "https://data.gesdisc.earthdata.nasa.gov").rstrip("/"),
            dataset_short_name=os.getenv("EARTHDATA_DATASET", "GPM_3IMERGHHL").strip(),
            download_dir=Path(os.getenv("EARTHDATA_DOWNLOAD_DIR", "data/raw/precipitation")),
            request_timeout_seconds=float(os.getenv("EARTHDATA_TIMEOUT_SECONDS", "45.0")),
        )

    def is_configured(self) -> bool:
        """Returns True if Bearer token or username/password is present."""
        return bool(self.token or (self.username and self.password))

    def configure_netrc(self) -> Optional[Path]:
        """Creates or safely updates the .netrc file for Earthdata URS authentication.

        Supports Windows (%USERPROFILE%/_netrc) and POSIX (~/.netrc).
        """
        if not (self.username and self.password):
            return None

        is_windows = os.name == "nt"
        home_dir = Path.home()
        netrc_path = home_dir / ("_netrc" if is_windows else ".netrc")

        machine_entry = f"machine urs.earthdata.nasa.gov login {self.username} password {self.password}\n"

        try:
            existing_content = ""
            if netrc_path.exists():
                existing_content = netrc_path.read_text(encoding="utf-8")

            if "urs.earthdata.nasa.gov" in existing_content:
                lines = existing_content.splitlines(keepends=True)
                filtered_lines = [line for line in lines if "urs.earthdata.nasa.gov" not in line]
                new_content = "".join(filtered_lines) + machine_entry
            else:
                new_content = existing_content + ("\n" if existing_content and not existing_content.endswith("\n") else "") + machine_entry

            netrc_path.write_text(new_content, encoding="utf-8")

            if not is_windows:
                os.chmod(netrc_path, stat.S_IRUSR | stat.S_IWUSR)

            logger.info("Successfully configured NASA Earthdata .netrc at %s", netrc_path)
            return netrc_path
        except Exception as exc:
            logger.warning("Could not configure .netrc: %s", exc)
            return None

    def get_auth_headers(self) -> Dict[str, str]:
        """Returns HTTP headers containing Bearer token or Basic auth if configured."""
        headers: Dict[str, str] = {
            "Accept": "*/*",
            "User-Agent": "Cyclone-X-IMERG-Worker/2.0",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def __repr__(self) -> str:
        masked_user = f"{self.username[:2]}***" if len(self.username) > 2 else "***"
        has_token = bool(self.token)
        return (
            f"EarthdataAuthConfig(username='{masked_user}', password='***', "
            f"has_token={has_token}, dataset='{self.dataset_short_name}')"
        )


class EarthdataAuth(httpx.Auth):
    """Custom HTTPX Auth class handling redirects between GES DISC and Earthdata URS."""

    def __init__(self, config: EarthdataAuthConfig) -> None:
        self.config = config

    def auth_flow(self, request: httpx.Request) -> AsyncIterator[httpx.Request]:
        if self.config.token:
            request.headers["Authorization"] = f"Bearer {self.config.token}"
        elif self.config.username and self.config.password:
            request.headers["Authorization"] = httpx.BasicAuth(
                self.config.username, self.config.password
            )._auth_header
        response = yield request

        if response.status_code == 401:
            if self.config.token:
                request.headers["Authorization"] = f"Bearer {self.config.token}"
                yield request
            elif self.config.username and self.config.password:
                request.headers["Authorization"] = httpx.BasicAuth(
                    self.config.username, self.config.password
                )._auth_header
                yield request


# -----------------------------------------------------------------------------
# Ingestion Worker Implementation
# -----------------------------------------------------------------------------
class NASAIMERGWorker:
    """Asynchronous worker for querying, downloading, and tensorizing NASA GPM IMERG rainfall data."""

    def __init__(self, config: Optional[EarthdataAuthConfig] = None) -> None:
        self.config = config or EarthdataAuthConfig.from_environment()
        self.bounds = GeoBounds()
        self._lock = asyncio.Lock()
        self._background_task: Optional[asyncio.Task] = None
        self._running = False

        # In-memory telemetry cache
        self.is_live: bool = False
        self.last_scan_utc: Optional[str] = None
        self.granules_available: int = 0
        self.latest_granule_id: Optional[str] = None
        self.latest_tensor: Optional[torch.Tensor] = None
        self.latest_stats: Dict[str, Any] = {}
        self.latency_ms: Optional[float] = None
        self.operational_health: str = "IDLE"
        self.last_poll_utc: Optional[str] = None

        # Ensure download directory exists
        self.config.download_dir.mkdir(parents=True, exist_ok=True)

    def _get_client(self) -> httpx.AsyncClient:
        """Returns a configured AsyncClient with Earthdata auth handling."""
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.request_timeout_seconds),
            headers=self.config.get_auth_headers(),
            follow_redirects=True,
        )

    async def query_latest_granules(
        self,
        hours_back: int = 12,
        bounds: Optional[GeoBounds] = None,
        run_type: str = "late",
    ) -> List[IMERGGranuleMetadata]:
        """Queries NASA CMR for the latest IMERG granules covering the target sector.

        Args:
            hours_back: How far back to search from current UTC time.
            bounds: Spatial bounding box (defaults to North Indian Ocean).
            run_type: 'late' (GPM_3IMERGHHL) or 'early' (GPM_3IMERGHHE).

        Returns:
            List of validated IMERGGranuleMetadata objects sorted latest first.
        """
        target_bounds = bounds or self.bounds
        dataset_short_name = "GPM_3IMERGHHL" if run_type == "late" else "GPM_3IMERGHHE"

        now_utc = datetime.now(timezone.utc)
        start_utc = now_utc - timedelta(hours=hours_back)
        temporal_str = f"{start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')},{now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}"

        cmr_url = f"{self.config.cmr_base_url}/search/granules.json"
        params = {
            "short_name": dataset_short_name,
            "bounding_box": target_bounds.cmr_bounding_box,
            "temporal": temporal_str,
            "sort_key[]": "-start_date",
            "page_size": 10,
        }

        logger.info(
            "Querying NASA CMR for %s granules (%d hours back) in sector [%s]",
            dataset_short_name,
            hours_back,
            target_bounds.cmr_bounding_box,
        )

        granules: List[IMERGGranuleMetadata] = []
        try:
            async with httpx.AsyncClient(timeout=self.config.request_timeout_seconds) as client:
                resp = await client.get(cmr_url, params=params)
                if resp.status_code != 200:
                    logger.warning("NASA CMR query returned HTTP %d: %s", resp.status_code, resp.text[:200])
                    # Fallback query without temporal constraint to get most recent available
                    resp = await client.get(
                        cmr_url,
                        params={
                            "short_name": dataset_short_name,
                            "page_size": 5,
                            "sort_key[]": "-start_date",
                        },
                    )

                if resp.status_code == 200:
                    data = resp.json()
                    entries = data.get("feed", {}).get("entry", [])
                    for entry in entries:
                        title = entry.get("title", "")
                        time_start_str = entry.get("time_start")
                        time_end_str = entry.get("time_end")

                        download_url = ""
                        for link in entry.get("links", []):
                            rel = link.get("rel", "")
                            href = link.get("href", "")
                            if "data#" in rel and (href.endswith(".HDF5") or href.endswith(".nc4") or href.endswith(".nc")):
                                download_url = href
                                break
                            elif "opendap" in href:
                                download_url = href

                        if not download_url:
                            for link in entry.get("links", []):
                                href = link.get("href", "")
                                if href.endswith(".HDF5") or href.endswith(".nc4"):
                                    download_url = href
                                    break

                        file_size = 0
                        try:
                            file_size = int(float(entry.get("granule_size", 0)) * 1024 * 1024)
                        except Exception:
                            pass

                        if title and download_url:
                            start_dt = (
                                datetime.fromisoformat(time_start_str.replace("Z", "+00:00"))
                                if time_start_str
                                else now_utc
                            )
                            end_dt = (
                                datetime.fromisoformat(time_end_str.replace("Z", "+00:00"))
                                if time_end_str
                                else now_utc
                            )
                            granule = IMERGGranuleMetadata(
                                granule_id=title.replace(".HDF5", ""),
                                dataset_short_name=dataset_short_name,
                                run_type=run_type,
                                start_time_utc=start_dt,
                                end_time_utc=end_dt,
                                download_url=download_url,
                                file_size_bytes=file_size,
                            )
                            granules.append(granule)

        except Exception as exc:
            logger.error("Failed to query NASA CMR: %s", exc)
            raise EarthdataQueryError(f"NASA CMR query failed: {exc}") from exc

        if not granules and run_type == "late":
            logger.info("No granules returned for Late Run. Attempting Early Run (GPM_3IMERGHHE)...")
            return await self.query_latest_granules(hours_back=hours_back, bounds=bounds, run_type="early")

        return granules

    async def download_granule(self, metadata: IMERGGranuleMetadata, force_download: bool = False) -> Path:
        """Downloads the authentic NetCDF4/HDF5 granule from NASA GES DISC.

        Args:
            metadata: Granule metadata containing authenticated download URL.
            force_download: Whether to re-download if the file already exists.

        Returns:
            Path to the verified raw file on disk.
        """
        clean_name = re.sub(r"[^a-zA-Z0-9._-]", "_", metadata.granule_id)
        if not clean_name.endswith(".HDF5") and not clean_name.endswith(".nc4"):
            clean_name += ".HDF5"
        dest_path = self.config.download_dir / clean_name

        if not force_download and dest_path.exists() and dest_path.stat().st_size > 100_000:
            if self._verify_hdf5_magic(dest_path):
                logger.info("Granule already cached and valid: %s", dest_path)
                return dest_path

        if not self.config.is_configured():
            raise EarthdataAuthError(
                "NASA Earthdata credentials not configured. Provide EARTHDATA_TOKEN or "
                "EARTHDATA_USERNAME & EARTHDATA_PASSWORD in .env"
            )

        logger.info("Streaming authentic IMERG granule from NASA GES DISC: %s", metadata.download_url)
        temp_dest = dest_path.with_suffix(".tmp")

        try:
            async with self._get_client() as client:
                auth = None
                if self.config.username and self.config.password and not self.config.token:
                    auth = httpx.BasicAuth(self.config.username, self.config.password)

                async with client.stream("GET", metadata.download_url, auth=auth) as response:
                    if response.status_code == 401 or response.status_code == 403:
                        raise EarthdataAuthError(
                            f"NASA GES DISC rejected authentication with status {response.status_code}."
                        )
                    if response.status_code != 200:
                        raise EarthdataDownloadError(
                            f"NASA GES DISC download failed with status {response.status_code}: {response.reason_phrase}"
                        )

                    with open(temp_dest, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            f.write(chunk)

            if not self._verify_hdf5_magic(temp_dest):
                if temp_dest.exists():
                    temp_dest.unlink()
                raise EarthdataProcessingError(
                    "Downloaded file failed HDF5 magic byte verification. Server may have returned an HTML error page."
                )

            if dest_path.exists():
                dest_path.unlink()
            temp_dest.rename(dest_path)
            logger.info("Successfully downloaded and verified IMERG granule: %s (%d bytes)", dest_path, dest_path.stat().st_size)
            return dest_path

        except Exception as exc:
            if temp_dest.exists():
                temp_dest.unlink()
            if isinstance(exc, (EarthdataAuthError, EarthdataProcessingError)):
                raise
            raise EarthdataDownloadError(f"Error downloading IMERG granule: {exc}") from exc

    def _verify_hdf5_magic(self, path: Path) -> bool:
        """Verifies HDF5 file magic header bytes: \\x89HDF\\r\\n\\x1a\\n."""
        try:
            with open(path, "rb") as f:
                header = f.read(8)
                return header == b"\x89HDF\r\n\x1a\n"
        except Exception:
            return False

    def extract_precipitation_tensor(
        self,
        file_path: Path,
        bounds: Optional[GeoBounds] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Extracts calibrated precipitation rate, subsets to the Indian Ocean basin, and builds a PyTorch tensor.

        Args:
            file_path: Path to authentic HDF5/NetCDF4 file.
            bounds: Spatial bounding box (defaults to North Indian Ocean).

        Returns:
            Tuple of:
            - torch.Tensor of shape [1, H, W] containing calibrated rain rates in mm/hr
            - Dictionary of metadata and meteorological statistics
        """
        target_bounds = bounds or self.bounds
        if not file_path.exists():
            raise FileNotFoundError(f"IMERG file not found: {file_path}")

        try:
            with h5py.File(file_path, "r") as h5:
                if "Grid" not in h5:
                    raise EarthdataProcessingError(f"Missing 'Grid' group in HDF5 file: {list(h5.keys())}")
                grid = h5["Grid"]

                lat = grid["lat"][:]
                lon = grid["lon"][:]

                var_name = "precipitation" if "precipitation" in grid else "precipitationCal"
                if var_name not in grid:
                    raise EarthdataProcessingError(
                        f"Neither 'precipitation' nor 'precipitationCal' found in Grid: {list(grid.keys())}"
                    )

                raw_data = grid[var_name]
                if raw_data.ndim == 3:
                    precip_2d = raw_data[0]
                else:
                    precip_2d = raw_data[:]

                lat_mask = (lat >= target_bounds.lat_min) & (lat <= target_bounds.lat_max)
                lon_mask = (lon >= target_bounds.lon_min) & (lon <= target_bounds.lon_max)

                lat_indices = np.where(lat_mask)[0]
                lon_indices = np.where(lon_mask)[0]

                if len(lat_indices) == 0 or len(lon_indices) == 0:
                    raise EarthdataProcessingError("Target bounding box outside granule spatial extent.")

                subset_lon_lat = precip_2d[
                    lon_indices[0] : lon_indices[-1] + 1,
                    lat_indices[0] : lat_indices[-1] + 1,
                ]

                subset_hw = np.transpose(subset_lon_lat)

                clean_hw = np.where((subset_hw < 0) | np.isnan(subset_hw), 0.0, subset_hw)
                clean_hw = np.clip(clean_hw, 0.0, 500.0)

                tensor = torch.from_numpy(clean_hw).unsqueeze(0).float()

                max_rain = float(tensor.max().item())
                mean_rain = float(tensor.mean().item())
                nonzero_cells = int((tensor > 0.1).sum().item())
                heavy_rain_cells = int((tensor >= 15.0).sum().item())
                heavy_rain_area_km2 = float(heavy_rain_cells * 121.0)

                metadata = {
                    "variable": var_name,
                    "units": "mm/hr",
                    "shape": list(tensor.shape),
                    "grid_height_lat": len(lat_indices),
                    "grid_width_lon": len(lon_indices),
                    "lat_min": float(lat[lat_indices[0]]),
                    "lat_max": float(lat[lat_indices[-1]]),
                    "lon_min": float(lon[lon_indices[0]]),
                    "lon_max": float(lon[lon_indices[-1]]),
                    "resolution_deg": 0.1,
                    "max_rain_rate_mm_hr": max_rain,
                    "mean_rain_rate_mm_hr": mean_rain,
                    "nonzero_cells": nonzero_cells,
                    "heavy_rain_cells": heavy_rain_cells,
                    "heavy_rain_area_km2": heavy_rain_area_km2,
                }

                logger.info(
                    "Extracted precipitation tensor %s: max=%.2f mm/hr, mean=%.3f mm/hr, nonzero_cells=%d",
                    tensor.shape,
                    max_rain,
                    mean_rain,
                    nonzero_cells,
                )
                return tensor, metadata

        except Exception as exc:
            if isinstance(exc, EarthdataProcessingError):
                raise
            raise EarthdataProcessingError(f"Failed to extract precipitation from {file_path}: {exc}") from exc

    async def run_once(self) -> Dict[str, Any]:
        """Performs a single complete query, download, tensor extraction, and telemetry update cycle."""
        async with self._lock:
            start_time = time.perf_counter()
            self.last_poll_utc = datetime.now(timezone.utc).isoformat()

            if not self.config.is_configured():
                self.operational_health = "UNCONFIGURED"
                self.is_live = False
                logger.warning("NASA Earthdata worker is not configured with credentials.")
                return self.get_status()

            try:
                granules = await self.query_latest_granules(hours_back=12)
                if not granules:
                    self.operational_health = "IDLE"
                    self.is_live = False
                    logger.info("NASA CMR returned 0 granules in the current temporal window.")
                    return self.get_status()

                latest_meta = granules[0]
                self.latest_granule_id = latest_meta.granule_id
                self.last_scan_utc = latest_meta.start_time_utc.isoformat()

                downloaded_file = await self.download_granule(latest_meta)

                tensor, stats = self.extract_precipitation_tensor(downloaded_file)

                self.latest_tensor = tensor
                self.latest_stats = stats
                self.granules_available = max(self.granules_available + 1, len(granules))
                self.is_live = True
                self.operational_health = "HEALTHY"

                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                self.latency_ms = round(elapsed_ms, 2)

                logger.info(
                    "NASA IMERG ingestion cycle completed successfully in %.2f ms (Granule: %s)",
                    self.latency_ms,
                    self.latest_granule_id,
                )
                return self.get_status()

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                self.latency_ms = round(elapsed_ms, 2)
                self.operational_health = "DEGRADED"
                self.is_live = False
                logger.error("Error during NASA IMERG ingestion run: %s", exc)
                return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """Returns structured operational telemetry summary for the IMERG pipeline."""
        return {
            "is_live": self.is_live,
            "last_scan_utc": self.last_scan_utc,
            "granules_available": self.granules_available,
            "max_rain_rate_mm_hr": self.latest_stats.get("max_rain_rate_mm_hr"),
            "mean_rain_rate_mm_hr": self.latest_stats.get("mean_rain_rate_mm_hr"),
            "heavy_rain_area_km2": self.latest_stats.get("heavy_rain_area_km2"),
            "latency_ms": self.latency_ms,
            "operational_health": self.operational_health,
            "latest_granule_id": self.latest_granule_id,
            "dataset_name": self.config.dataset_short_name,
            "tensor_shape": list(self.latest_tensor.shape) if self.latest_tensor is not None else None,
            "last_poll_utc": self.last_poll_utc,
        }

    async def _background_loop(self, poll_interval_seconds: int = 1800) -> None:
        """Background loop executing IMERG ingestion periodically."""
        logger.info("Starting NASA IMERG background polling loop (interval: %d seconds)", poll_interval_seconds)
        while self._running:
            try:
                await self.run_once()
            except Exception as exc:
                logger.error("Unhandled error in IMERG polling loop: %s", exc)
            await asyncio.sleep(poll_interval_seconds)

    def start_background_loop(self, poll_interval_seconds: int = 1800) -> None:
        """Launches the asynchronous polling loop as a background task."""
        if self._running or (self._background_task and not self._background_task.done()):
            return
        self._running = True
        loop = asyncio.get_event_loop()
        self._background_task = loop.create_task(self._background_loop(poll_interval_seconds))

    def stop_background_loop(self) -> None:
        """Gracefully halts the background polling loop."""
        self._running = False
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()


# Singleton instance
nasa_imerg_worker = NASAIMERGWorker()
