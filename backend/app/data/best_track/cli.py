import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
"""Command Line Interface for Best Track Ingestion."""

import argparse
import sys
from datetime import datetime, timezone
from backend.app.data.best_track.deduplicator import BestTrackDeduplicator
from backend.app.data.best_track.ibtracs_adapter import IbtracsBestTrackAdapter
from backend.app.data.best_track.imd_adapter import ImdBestTrackAdapter
from backend.app.data.best_track.merger import BestTrackMerger
from backend.app.data.best_track.repository import BestTrackRepository
from backend.app.data.best_track.validator import BestTrackValidator


def main():
    parser = argparse.ArgumentParser(description="Cyclone-X Best Track Ingestion Pipeline")
    parser.add_argument(
        "--source",
        choices=["imd", "ibtracs", "all"],
        default="all",
        help="Data source to ingest (default: all)",
    )
    parser.add_argument(
        "--year",
        type=int,
        nargs="*",
        help="Selective year(s) to import, e.g. --year 2020 or --year 2019 2020",
    )
    parser.add_argument(
        "--all-years",
        action="store_true",
        help="Import all available historical years",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and discover data without writing to SQLite database or canonical files",
    )
    args = parser.parse_args()

    years = args.year if args.year else None

    print("=" * 60)
    print("CYCLONE-X: REAL HISTORICAL BEST TRACK INGESTION")
    print(f"Source: {args.source.upper()}")
    print(f"Years: {years if years else 'All available'}")
    print(f"Dry Run: {args.dry_run}")
    print("=" * 60)

    raw_fixes = []
    imd_manifest_rows = []

    # 1. IMD Ingestion
    if args.source in ("imd", "all"):
        print("\n[1/4] Ingesting IMD RSMC New Delhi Best Track...")
        imd_adapter = ImdBestTrackAdapter()
        imd_fixes, manifest_rows = imd_adapter.parse_all(target_years=years)
        raw_fixes.extend(imd_fixes)
        imd_manifest_rows.extend(manifest_rows)
        print(f"   [OK] Extracted {len(imd_fixes)} raw fixes from IMD across {len(manifest_rows)} sheet(s)")

    # 2. IBTrACS Ingestion
    if args.source in ("ibtracs", "all"):
        print("\n[2/4] Ingesting NOAA NCEI IBTrACS North Indian Ocean...")
        ibtracs_adapter = IbtracsBestTrackAdapter()
        ibtracs_fixes = ibtracs_adapter.parse_all(target_years=years)
        raw_fixes.extend(ibtracs_fixes)
        print(f"   [OK] Extracted {len(ibtracs_fixes)} raw fixes from NOAA IBTrACS")

    total_extracted = len(raw_fixes)
    print(f"\nTotal candidate fixes extracted: {total_extracted}")

    # 3. Validation & Quality Control
    print("\n[3/4] Executing Scientific Data Validation...")
    valid_fixes, quarantine, report = BestTrackValidator.audit_fixes(raw_fixes)
    print(f"   [OK] Valid: {report['valid_records']} fixes ({report['validation_pass_rate_percent']}%)")
    print(f"   [WARN] Quarantined: {report['quarantined_records']} fixes")
    print(f"   [INFO] Missing wind: {report['missing_wind']}, Missing pressure: {report['missing_pressure']}")

    # 4. Deduplication
    print("\n[4/4] Deduplication & Cross-Source Analysis...")
    deduped_fixes, dup_count = BestTrackDeduplicator.deduplicate(valid_fixes)
    print(f"   [OK] Deduplicated fixes: {len(deduped_fixes)} (removed {dup_count} exact duplicates)")

    deduped_fixes = BestTrackDeduplicator.flag_cross_source_differences(deduped_fixes)

    if args.dry_run:
        print("\n[DRY RUN COMPLETE] No modifications were written to disk.")
        sys.exit(0)

    # 5. Export and Database Persistence
    print("\nExporting canonical datasets & indexing SQLite database...")
    merger = BestTrackMerger()
    storm_index = merger.build_storm_index(deduped_fixes)
    export_paths = merger.export_all(
        fixes=deduped_fixes,
        quarantine=quarantine,
        validation_report=report,
        imd_manifest_rows=imd_manifest_rows,
    )

    repo = BestTrackRepository()
    repo.insert_batch(storms=storm_index, fixes=deduped_fixes)
    stats = repo.get_stats()

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Total Historical Storms Indexed: {stats.total_storms}")
    print(f"Total Authentic Fixes Stored:    {stats.total_fixes}")
    print(f"Year Range:                      {stats.year_range['min_year']} - {stats.year_range['max_year']}")
    print(f"Source Counts:                   {stats.source_counts}")
    print(f"Basin Counts:                    {stats.basin_counts}")
    print(f"Named Storms:                    {stats.named_storms_count}")
    print(f"SQLite DB:                       {repo.db_path}")
    print(f"Master CSV:                      {export_paths['canonical_csv']}")
    print(f"Master JSON:                     {export_paths['canonical_json']}")
    print(f"Storm Index:                     {export_paths['storm_index_csv']}")
    print(f"Validation Report:               {export_paths['validation_report_json']}")
    print(f"Quarantine CSV:                  {export_paths['quarantine_csv']}")


if __name__ == "__main__":
    main()
