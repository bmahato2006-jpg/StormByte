"""Cyclone-X MOSDAC Live Ingestion Worker Entrypoint.

Delegates to backend.app.data.geospatial.mosdac_ingestion_worker.
"""

from backend.app.data.geospatial.mosdac_ingestion_worker import (
    MOSDACAuthConfig,
    MOSDACLockoutManager,
    MOSDACClient,
    MOSDACIngestionWorker,
    GranuleMetadata,
    GeoBounds,
    fetch_live_insat_tiles,
    download_satellite_granule,
    extract_and_normalize_tir1,
    MOSDACBaseException,
    MOSDACAuthError,
    MOSDACAccountLockedError,
    MOSDACQueryError,
    MOSDACDownloadError,
    MOSDACCorruptFileError,
    main,
)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
