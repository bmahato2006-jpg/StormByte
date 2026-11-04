from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "CYCLONE-X"
    VERSION: str = "0.1.0"
    API_PREFIX: str = "/api"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    DEMO_MODE: bool = False
    
    # CORS Origins (Allow frontend production on Vercel & local development)
    CORS_ORIGINS: List[str] = [
        "https://frontend-virid-nu-57pxwx9lvb.vercel.app",
        "https://frontend-erl66fv60-paschim-banga-durgapuja-samannay-samity.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]
    
    # Database Configuration (PostGIS Ready)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/cyclone_x"

    # Phase 7 & 11 Scientific Dataset Configuration
    IBTRACS_SOURCE_URL: str = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.NI.list.v04r00.csv"
    IMD_SOURCE_URL: str = "https://rsmcnewdelhi.imd.gov.in"
    MOSDAC_API_URL: str = "https://mosdac.gov.in"
    MOSDAC_USERNAME: str = ""
    MOSDAC_PASSWORD: str = ""
    MOSDAC_DATASET_ID: str = "3DIMG_L1B_STD"
    MOSDAC_API_KEY: str = ""  # Deprecated in favor of MOSDAC SSO (MOSDAC_USERNAME / MOSDAC_PASSWORD)
    NASA_GPM_URL: str = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3"
    NASA_GPM_USERNAME: str = ""
    NASA_GPM_PASSWORD: str = ""
    NASA_PPS_URL: str = "https://arthurhouhttps.pps.gsfc.nasa.gov/text"
    NASA_PPS_EMAIL: str = ""
    NASA_PPS_PASSWORD: str = ""
    EARTHDATA_TOKEN: str = ""
    NASA_PPS_KEY: str = ""  # Deprecated legacy reference
    ERA5_SOURCE_URL: str = "https://cds.climate.copernicus.eu/api/v2"
    CDS_API_KEY: str = ""
    ERA5_VARIABLES: str = "sea_surface_temperature,u_component_of_wind,v_component_of_wind,mean_sea_level_pressure"
    WORLDPOP_SOURCE_URL: str = "https://hub.worldpop.org/geodata/summary?id=IND"
    SRTM_SOURCE_URL: str = "https://earthexplorer.usgs.gov"


settings = Settings()


