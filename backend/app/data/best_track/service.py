"""Business Logic Service for Best Track Catalog, Satellite Matching, and LSTM Backtesting."""

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional
from backend.app.data.best_track.deduplicator import haversine
from backend.app.data.best_track.repository import BestTrackRepository
from backend.app.data.best_track.schemas import (
    BestTrackFix,
    BestTrackStats,
    ForecastEvaluationResult,
    TrackAssociationResult,
)
from backend.app.data.best_track.source_registry import (
    REQUIRED_FIXES_FOR_FORECAST,
    TRACK_MATCH_MAX_DISTANCE_KM,
    TRACK_MATCH_MAX_TIME_HOURS,
)
from backend.app.ml.trajectory.inference import trajectory_inference_service


class BestTrackService:
    """Coordinates Best Track queries, satellite image association, and offline ML backtesting."""

    def __init__(self, repo: Optional[BestTrackRepository] = None):
        self.repo = repo or BestTrackRepository()

    def list_storms(
        self,
        source: Optional[str] = None,
        year: Optional[int] = None,
        basin: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Returns paginated storm list with total count."""
        records, total = self.repo.get_storms(
            source=source, year=year, basin=basin, search=query, limit=limit, offset=offset
        )
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "storms": records,
        }

    def get_storm(self, storm_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single storm summary record."""
        return self.repo.get_storm(storm_id)

    def get_storm_track(self, storm_id: str) -> Dict[str, Any]:
        """Returns ordered genuine fixes and GeoJSON representation for map rendering."""
        storm = self.repo.get_storm(storm_id)
        if not storm:
            return {"error": f"Storm '{storm_id}' not found in Best Track catalog."}

        fixes = self.repo.get_fixes(storm_id)

        # Build GeoJSON LineString coordinates [lon, lat]
        geojson_coordinates = [[f["longitude"], f["latitude"]] for f in fixes]
        geojson_feature = {
            "type": "Feature",
            "properties": {
                "storm_id": storm["storm_id"],
                "storm_name": storm["storm_name"],
                "year": storm["year"],
                "basin": storm["basin"],
                "source": storm["source"],
                "point_count": len(fixes),
                "is_observed_track": True,
                "label": "REAL / OBSERVED HISTORICAL TRACK",
            },
            "geometry": {
                "type": "LineString",
                "coordinates": geojson_coordinates,
            },
        }

        return {
            "storm": storm,
            "fixes_count": len(fixes),
            "fixes": fixes,
            "geojson": geojson_feature,
        }

    def match_satellite_image_to_track(
        self,
        storm_id: str,
        image_timestamp_utc: Optional[str] = None,
        image_latitude: Optional[float] = None,
        image_longitude: Optional[float] = None,
    ) -> TrackAssociationResult:
        """Evaluates association between an unreferenced or georeferenced satellite image and a track."""
        fixes = self.repo.get_fixes(storm_id)
        if not fixes:
            return TrackAssociationResult(
                status="NO_MATCH",
                details=f"No track fixes found for storm ID '{storm_id}'.",
            )

        # Zero-fabrication check: if coordinates are missing, return UNVERIFIED
        if image_latitude is None or image_longitude is None:
            return TrackAssociationResult(
                status="UNVERIFIED",
                satellite_timestamp_utc=image_timestamp_utc,
                satellite_coordinates={"latitude": None, "longitude": None},
                time_difference_hours=None,
                spatial_distance_km=None,
                candidate_fixes=[],
                details=(
                    "TRACK ASSOCIATION: UNVERIFIED. This historical satellite image does not possess calibrated "
                    "geodetic coordinates (CRS bounds). Geodetic position remains strictly null. "
                    "CYCLONE-X does not assume or fabricate association without verified spatiotemporal metadata."
                ),
            )

        if not image_timestamp_utc:
            return TrackAssociationResult(
                status="UNVERIFIED",
                satellite_coordinates={"latitude": image_latitude, "longitude": image_longitude},
                details="TRACK ASSOCIATION: UNVERIFIED. Satellite image timestamp is missing.",
            )

        # Calculate time delta and distance to all fixes
        try:
            img_dt = datetime.fromisoformat(image_timestamp_utc.replace("Z", "+00:00"))
        except Exception as e:
            return TrackAssociationResult(
                status="UNVERIFIED",
                details=f"Malformed image timestamp: {e}",
            )

        candidates = []
        min_dist = float("inf")
        nearest_fix = None
        min_dt_hours = float("inf")

        for f in fixes:
            f_dt = datetime.fromisoformat(f["timestamp_utc"].replace("Z", "+00:00"))
            dt_h = abs((img_dt - f_dt).total_seconds()) / 3600.0
            dist_km = haversine(image_latitude, image_longitude, f["latitude"], f["longitude"])

            if dist_km < min_dist:
                min_dist = dist_km
                nearest_fix = f
                min_dt_hours = dt_h

            if dt_h <= TRACK_MATCH_MAX_TIME_HOURS and dist_km <= TRACK_MATCH_MAX_DISTANCE_KM:
                candidates.append({
                    "fix_id": f["id"],
                    "timestamp": f["timestamp_utc"],
                    "latitude": f["latitude"],
                    "longitude": f["longitude"],
                    "time_diff_hours": round(dt_h, 2),
                    "dist_km": round(dist_km, 1),
                    "_fix": f,
                })

            if dist_km < min_dist:
                min_dist = dist_km
                nearest_fix = f
                min_dt_hours = dt_h

        # Candidate assessment
        if candidates:
            candidates.sort(key=lambda c: (c["time_diff_hours"], c["dist_km"]))
            best = candidates[0]
            nearest_fix = best.get("_fix")
            for c in candidates:
                c.pop("_fix", None)
            min_dt_hours = best["time_diff_hours"]
            min_dist = best["dist_km"]

            if min_dt_hours <= 1.5 and min_dist <= 50.0:
                status = "VERIFIED_MATCH"
                details = f"Verified spatiotemporal match: nearest fix within {round(min_dt_hours, 1)}h and {round(min_dist, 1)} km."
            else:
                status = "POSSIBLE_MATCH"
                details = f"Candidate match within tolerance: time diff {round(min_dt_hours, 1)}h, distance {round(min_dist, 1)} km."
        else:
            status = "NO_MATCH"
            details = f"No spatiotemporal proximity match (nearest fix is {round(min_dist, 1)} km away, dt={round(min_dt_hours, 1)}h)."

        return TrackAssociationResult(
            status=status,
            satellite_timestamp_utc=image_timestamp_utc,
            satellite_coordinates={"latitude": image_latitude, "longitude": image_longitude},
            time_difference_hours=round(min_dt_hours, 2),
            spatial_distance_km=round(min_dist, 1),
            nearest_fix=nearest_fix,
            candidate_fixes=candidates,
            details=details,
        )

    def evaluate_forecast(
        self,
        storm_id: str,
        lead_steps: int = 4,
    ) -> ForecastEvaluationResult:
        """Executes an offline historical backtest benchmark comparing TrajectoryLSTM vs subsequent real fixes."""
        if lead_steps < REQUIRED_FIXES_FOR_FORECAST:
            raise ValueError(
                f"INSUFFICIENT_TRACK_HISTORY: Phase 8 Trajectory LSTM strictly requires at least "
                f"{REQUIRED_FIXES_FOR_FORECAST} sequential fixes. Provided lead_steps={lead_steps}."
            )

        fixes = self.repo.get_fixes(storm_id)
        if len(fixes) < lead_steps:
            raise ValueError(
                f"INSUFFICIENT_TRACK_HISTORY: Storm '{storm_id}' has {len(fixes)} fixes, "
                f"cannot supply {lead_steps} historical fixes."
            )

        if not trajectory_inference_service.is_available():
            raise RuntimeError("Trajectory LSTM model is not available. Please ensure model checkpoint exists.")

        storm = self.repo.get_storm(storm_id)
        storm_name = storm["storm_name"] if storm else "UNKNOWN"

        # Zero data leakage: split strictly at lead_steps
        input_history = fixes[:lead_steps]
        future_fixes = fixes[lead_steps:]
        eval_fix = input_history[-1]
        eval_time_str = eval_fix["timestamp_utc"]
        eval_dt = datetime.fromisoformat(eval_time_str.replace("Z", "+00:00"))

        # Run model inference solely on input_history
        model_pred = trajectory_inference_service.predict(input_history, forecast_horizons_hours=[6, 12, 24])

        metrics: Dict[str, Dict[str, Optional[float]]] = {}
        all_errors_km: List[float] = []

        predicted_positions = []
        actual_observed_positions = []

        for p in model_pred.get("predictions", []):
            h = p["forecast_lead_hours"]
            pred_lat = p["latitude"]
            pred_lon = p["longitude"]
            pred_time = p["valid_time"]

            predicted_positions.append({
                "lead_hours": h,
                "timestamp": pred_time,
                "latitude": pred_lat,
                "longitude": pred_lon,
                "uncertainty_km": p.get("uncertainty_radius_km_68", 35.0),
            })

            # Search future genuine fixes for actual fix closest to T + h
            target_dt = datetime.fromisoformat(pred_time.replace("Z", "+00:00"))
            matched_actual = None
            min_td_sec = float("inf")

            for f_fix in future_fixes:
                f_dt = datetime.fromisoformat(f_fix["timestamp_utc"].replace("Z", "+00:00"))
                td_sec = abs((f_dt - target_dt).total_seconds())
                # Within 2 hours tolerance of the nominal horizon
                if td_sec < 7200 and td_sec < min_td_sec:
                    min_td_sec = td_sec
                    matched_actual = f_fix

            if matched_actual:
                actual_lat = matched_actual["latitude"]
                actual_lon = matched_actual["longitude"]
                err_km = haversine(pred_lat, pred_lon, actual_lat, actual_lon)
                all_errors_km.append(err_km)

                metrics[f"{h}h"] = {
                    "mae_km": round(err_km, 2),
                    "rmse_km": round(err_km, 2),
                    "target_time": pred_time,
                    "actual_time": matched_actual["timestamp_utc"],
                    "actual_lat": actual_lat,
                    "actual_lon": actual_lon,
                }
                actual_observed_positions.append({
                    "lead_hours": h,
                    "timestamp": matched_actual["timestamp_utc"],
                    "latitude": actual_lat,
                    "longitude": actual_lon,
                    "actual_wind_kt": matched_actual.get("max_wind_kt"),
                })
            else:
                metrics[f"{h}h"] = {
                    "mae_km": None,
                    "rmse_km": None,
                    "note": f"No post-season verification fix found within 2 hours of T+{h}h",
                }

        overall_mae = round(sum(all_errors_km) / len(all_errors_km), 2) if all_errors_km else None
        overall_rmse = round(math.sqrt(sum(e**2 for e in all_errors_km) / len(all_errors_km)), 2) if all_errors_km else None

        return ForecastEvaluationResult(
            storm_id=storm_id,
            storm_name=storm_name,
            evaluation_time_utc=eval_time_str,
            history_fixes_count=len(input_history),
            future_fixes_count=len(future_fixes),
            model_name="trajectory_lstm_v1",
            horizons_evaluated=[6, 12, 24],
            metrics=metrics,
            overall_mae_km=overall_mae,
            overall_rmse_km=overall_rmse,
            input_history=input_history,
            predicted_positions=predicted_positions,
            actual_observed_positions=actual_observed_positions,
        )

    def get_stats(self) -> BestTrackStats:
        """Returns catalog statistics."""
        return self.repo.get_stats()
