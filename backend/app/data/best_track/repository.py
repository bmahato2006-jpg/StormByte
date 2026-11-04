"""SQLite Repository for Best Track Catalog."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from backend.app.data.best_track.schemas import (
    BestTrackFix,
    BestTrackStats,
    StormIndexRecord,
)
from backend.app.data.best_track.source_registry import DB_PATH, PROCESSED_DIR


class BestTrackRepository:
    """High-performance SQLite storage and query engine for the Cyclone-X Best Track database."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Creates the database tables and composite indices if not present."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Storms table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS storms (
                    storm_id TEXT PRIMARY KEY,
                    storm_name TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    basin TEXT NOT NULL,
                    source TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    point_count INTEGER NOT NULL,
                    max_wind_kt REAL,
                    min_pressure_hpa REAL,
                    cyclone_category TEXT
                )
            """)
            # Fixes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS track_fixes (
                    id TEXT PRIMARY KEY,
                    storm_id TEXT NOT NULL,
                    storm_name TEXT NOT NULL,
                    basin TEXT NOT NULL,
                    subbasin TEXT,
                    source TEXT NOT NULL,
                    source_agency TEXT NOT NULL,
                    source_record_id TEXT,
                    timestamp_utc TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    max_wind_kt REAL,
                    central_pressure_hpa REAL,
                    cyclone_category TEXT,
                    ci_number REAL,
                    pressure_drop_hpa REAL,
                    grade TEXT,
                    is_best_track INTEGER NOT NULL DEFAULT 1,
                    source_file TEXT NOT NULL,
                    source_url TEXT,
                    retrieved_at_utc TEXT NOT NULL,
                    data_quality TEXT NOT NULL DEFAULT 'VERIFIED_OFFICIAL',
                    validation_status TEXT NOT NULL DEFAULT 'PASSED',
                    cross_source_difference INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (storm_id) REFERENCES storms(storm_id)
                )
            """)
            # Create indices for sub-millisecond lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_storms_year ON storms(year)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_storms_name ON storms(storm_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_storms_source ON storms(source)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fixes_storm ON track_fixes(storm_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fixes_time ON track_fixes(timestamp_utc)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fixes_coords ON track_fixes(latitude, longitude)")
            conn.commit()

    def insert_batch(self, storms: List[StormIndexRecord], fixes: List[BestTrackFix]):
        """Persists storm index entries and fixes within an atomic transaction."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Upsert storms
            cursor.executemany("""
                INSERT OR REPLACE INTO storms (
                    storm_id, storm_name, year, basin, source,
                    start_time, end_time, point_count, max_wind_kt,
                    min_pressure_hpa, cyclone_category
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    s.storm_id, s.storm_name, s.year, s.basin, s.source,
                    s.start_time, s.end_time, s.point_count, s.max_wind_kt,
                    s.min_pressure_hpa, s.cyclone_category
                ) for s in storms
            ])

            # Upsert fixes
            cursor.executemany("""
                INSERT OR REPLACE INTO track_fixes (
                    id, storm_id, storm_name, basin, subbasin, source,
                    source_agency, source_record_id, timestamp_utc, latitude,
                    longitude, max_wind_kt, central_pressure_hpa, cyclone_category,
                    ci_number, pressure_drop_hpa, grade, is_best_track,
                    source_file, source_url, retrieved_at_utc, data_quality,
                    validation_status, cross_source_difference
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    f.id, f.storm_id, f.storm_name, f.basin, f.subbasin, f.source,
                    f.source_agency, f.source_record_id, f.timestamp_utc.isoformat(),
                    f.latitude, f.longitude, f.max_wind_kt, f.central_pressure_hpa,
                    f.cyclone_category, f.ci_number, f.pressure_drop_hpa, f.grade,
                    1 if f.is_best_track else 0, f.source_file, f.source_url,
                    f.retrieved_at_utc.isoformat(), f.data_quality, f.validation_status,
                    1 if f.cross_source_difference else 0
                ) for f in fixes
            ])
            conn.commit()

    def get_storms(
        self,
        source: Optional[str] = None,
        year: Optional[int] = None,
        basin: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Retrieves paginated and filtered storms."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            clauses = []
            params: List[Any] = []

            if source:
                clauses.append("source = ?")
                params.append(source)
            if year:
                clauses.append("year = ?")
                params.append(year)
            if basin:
                clauses.append("basin = ?")
                params.append(basin)
            if search:
                clauses.append("(storm_name LIKE ? OR storm_id LIKE ?)")
                params.extend([f"%{search.upper()}%", f"%{search}%"])

            where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""

            # Count total
            cursor.execute(f"SELECT COUNT(*) FROM storms {where_sql}", params)
            total = cursor.fetchone()[0]

            # Query page
            query_sql = f"""
                SELECT * FROM storms
                {where_sql}
                ORDER BY year DESC, start_time DESC
                LIMIT ? OFFSET ?
            """
            cursor.execute(query_sql, params + [limit, offset])
            rows = [dict(r) for r in cursor.fetchall()]
            return rows, total

    def get_storm(self, storm_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves summary record for a specific storm by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM storms WHERE storm_id = ?", (storm_id,))
            row = cursor.fetchone()
            if not row:
                # Try case-insensitive or partial match
                cursor.execute("SELECT * FROM storms WHERE storm_id LIKE ? OR storm_name = ? LIMIT 1", (f"%{storm_id}%", storm_id.upper()))
                row = cursor.fetchone()
            return dict(row) if row else None

    def get_fixes(self, storm_id: str) -> List[Dict[str, Any]]:
        """Retrieves all ordered genuine fixes for a specific storm."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM track_fixes
                WHERE storm_id = ?
                ORDER BY timestamp_utc ASC
            """, (storm_id,))
            rows = cursor.fetchall()
            if not rows:
                # Fallback to loose match
                cursor.execute("""
                    SELECT * FROM track_fixes
                    WHERE storm_id LIKE ? OR storm_name = ?
                    ORDER BY timestamp_utc ASC
                """, (f"%{storm_id}%", storm_id.upper()))
                rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_stats(self) -> BestTrackStats:
        """Computes master catalog statistics across IMD and IBTrACS archives."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM storms")
            total_storms = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM track_fixes")
            total_fixes = cursor.fetchone()[0]

            cursor.execute("SELECT MIN(year), MAX(year) FROM storms")
            row = cursor.fetchone()
            min_yr = row[0] or 1842
            max_yr = row[1] or 2026

            cursor.execute("SELECT source, COUNT(*) FROM track_fixes GROUP BY source")
            source_counts = dict(cursor.fetchall())

            cursor.execute("SELECT basin, COUNT(*) FROM storms GROUP BY basin")
            basin_counts = dict(cursor.fetchall())

            cursor.execute("SELECT COUNT(*) FROM storms WHERE storm_name != 'UNNAMED'")
            named_count = cursor.fetchone()[0]

            return BestTrackStats(
                total_storms=total_storms,
                total_fixes=total_fixes,
                year_range={"min_year": min_yr, "max_year": max_yr},
                source_counts=source_counts,
                basin_counts=basin_counts,
                named_storms_count=named_count,
            )
