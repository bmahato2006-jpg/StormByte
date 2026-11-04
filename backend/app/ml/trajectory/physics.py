"""Physics-Guided Spatio-Temporal Trajectory Forecaster.

Integrates:
1. Inertial translation momentum from historical best-track observations.
2. Environmental 500 hPa deep-layer steering flow advection.
3. Planetary Rossby beta-drift deflection (poleward and westward vortex gyres).
4. Physical intensity trend projection (+6h, +12h, +24h).
"""

from datetime import datetime, timedelta, timezone
import math
from typing import Any, Dict, List, Optional, Tuple


class TrajectoryForecaster:
    """Predicts cyclone forward positions and intensities using physics-guided advection."""

    EARTH_RADIUS_KM = 6371.0
    OMEGA = 7.2921159e-5  # rad/s

    @classmethod
    def compute_beta_drift(cls, latitude: float, max_wind_knots: float) -> Tuple[float, float]:
        """Calculates Rossby beta-drift velocity components (u_beta, v_beta) in knots.

        Beta gyres induce a characteristic northwestward drift in the Northern Hemisphere
        typically between 2.0 and 5.0 knots directed around azimuth 315°.
        """
        lat_rad = math.radians(abs(latitude))
        # Planetary vorticity gradient beta = 2 * Omega * cos(lat) / R_earth
        beta = 2.0 * cls.OMEGA * math.cos(lat_rad) / (cls.EARTH_RADIUS_KM * 1000.0)

        # Empirical scaling based on vortex intensity
        drift_speed_knots = max(1.5, min(5.5, 2.0 + (max_wind_knots / 100.0) * 2.5))
        drift_azimuth_deg = 315.0  # North-West in Northern Hemisphere

        az_rad = math.radians(drift_azimuth_deg)
        u_beta = drift_speed_knots * math.sin(az_rad)  # Eastward (negative if west)
        v_beta = drift_speed_knots * math.cos(az_rad)  # Northward
        return u_beta, v_beta

    @classmethod
    def compute_translation_vector(
        cls,
        track_points: List[Dict[str, Any]]
    ) -> Tuple[float, float, float, float]:
        """Computes current translation speed (knots), direction (deg), and (u, v) components.

        Uses the most recent observation points.
        """
        if len(track_points) < 2:
            p = track_points[-1] if track_points else {}
            speed_val = p.get("movementSpeed")
            dir_val = p.get("movementDirection")
            speed_kt = 0.0
            if speed_val is not None:
                try:
                    speed_kt = float(speed_val) / 1.852 if float(speed_val) > 0 else 0.0
                except (ValueError, TypeError):
                    speed_kt = 0.0

            bearing = 0.0
            if isinstance(dir_val, (int, float)):
                bearing = float(dir_val) % 360.0
            elif isinstance(dir_val, str):
                dir_map = {
                    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
                    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
                    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
                    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5
                }
                bearing = dir_map.get(dir_val.upper().strip(), 0.0)

            az_rad = math.radians(bearing)
            u = speed_kt * math.sin(az_rad)
            v = speed_kt * math.cos(az_rad)
            return speed_kt, bearing, u, v

        p1 = track_points[-2]
        p2 = track_points[-1]

        lat1, lon1 = float(p1["latitude"]), float(p1["longitude"])
        lat2, lon2 = float(p2["latitude"]), float(p2["longitude"])

        # Attempt to parse time difference
        dt_hours = 6.0  # standard synoptic interval
        try:
            t1 = datetime.fromisoformat(str(p1["timestamp"]).replace("Z", "+00:00"))
            t2 = datetime.fromisoformat(str(p2["timestamp"]).replace("Z", "+00:00"))
            diff_sec = (t2 - t1).total_seconds()
            if diff_sec > 600:
                dt_hours = diff_sec / 3600.0
        except Exception:
            dt_hours = 6.0

        # Great-circle distance in nautical miles (1 deg latitude = 60 nm)
        d_lat_nm = (lat2 - lat1) * 60.0
        mid_lat = math.radians((lat1 + lat2) / 2.0)
        d_lon_nm = (lon2 - lon1) * 60.0 * math.cos(mid_lat)

        dist_nm = math.hypot(d_lat_nm, d_lon_nm)
        speed_knots = max(0.0, min(45.0, dist_nm / dt_hours))

        # Heading angle 0-360 degrees (0 = North, 90 = East)
        bearing = (math.degrees(math.atan2(d_lon_nm, d_lat_nm)) + 360.0) % 360.0

        u_trans = d_lon_nm / dt_hours
        v_trans = d_lat_nm / dt_hours
        return speed_knots, bearing, u_trans, v_trans

    @classmethod
    def forecast(
        cls,
        track_history: List[Dict[str, Any]],
        steering_vector_knots: Optional[Tuple[float, float]] = None,
        environmental_sst_c: Optional[float] = None,
        vertical_wind_shear_knots: Optional[float] = None,
        forecast_horizons_hours: List[int] = [6, 12, 24],
    ) -> List[Dict[str, Any]]:
        """Forecasts cyclone forward positions and intensity states for +6h, +12h, and +24h."""
        if not track_history:
            raise ValueError("Track history must contain at least one observation point.")

        latest = track_history[-1]
        curr_lat = float(latest["latitude"])
        curr_lon = float(latest["longitude"])

        # Recover authentic wind & pressure from history backwards
        curr_wind = None
        for pt in reversed(track_history):
            if pt.get("windSpeed") is not None:
                curr_wind = float(pt["windSpeed"])
                break

        curr_pressure = None
        for pt in reversed(track_history):
            if pt.get("pressure") is not None:
                curr_pressure = float(pt["pressure"])
                break

        if curr_pressure is None:
            # Query live surface pressure from Open-Meteo
            from backend.app.services.cyclone_service import fetch_live_surface_pressure
            live_p = fetch_live_surface_pressure(curr_lat, curr_lon)
            curr_pressure = float(live_p) if live_p is not None else 1010.0

        if curr_wind is None:
            # Physical wind-pressure relation for cyclonic vorticity (Knaff-Zehr / Atkinson-Holliday)
            delta_p = max(0.0, 1010.0 - curr_pressure)
            curr_wind = round(max(0.0, 3.4 * (delta_p ** 0.74)), 1)

        # Parse base timestamp
        base_time = datetime.now(timezone.utc)
        if "timestamp" in latest:
            try:
                base_time = datetime.fromisoformat(str(latest["timestamp"]).replace("Z", "+00:00"))
            except Exception:
                pass

        # Compute translation momentum
        speed_knots, bearing_deg, u_inert, v_inert = cls.compute_translation_vector(track_history)

        # If steering flow vector (u_steer, v_steer) is not provided, estimate from translation
        if steering_vector_knots is None:
            # Steering is approximately 75% of forward translation with a slight eastward deflection
            u_steer = u_inert * 0.85
            v_steer = v_inert * 0.85
        else:
            u_steer, v_steer = steering_vector_knots

        # Beta drift components
        u_beta, v_beta = cls.compute_beta_drift(curr_lat, curr_wind)

        predictions: List[Dict[str, Any]] = []
        last_lat = curr_lat
        last_lon = curr_lon
        last_t = 0
        sim_wind = curr_wind
        sim_pressure = curr_pressure

        for lead_h in sorted(forecast_horizons_hours):
            step_dt = lead_h - last_t

            # Dynamic weighting: Inertia dominates early; Steering flow dominates late
            if lead_h <= 6:
                w_inert, w_steer, w_beta = 0.65, 0.25, 0.10
            elif lead_h <= 12:
                w_inert, w_steer, w_beta = 0.40, 0.45, 0.15
            else:
                w_inert, w_steer, w_beta = 0.20, 0.60, 0.20

            u_eff = (w_inert * u_inert) + (w_steer * u_steer) + (w_beta * u_beta)
            v_eff = (w_inert * v_inert) + (w_steer * v_steer) + (w_beta * v_beta)

            # Convert (u_eff, v_eff) in knots to coordinate displacement in degrees
            # 1 degree latitude = 60 nautical miles
            d_lat_deg = (v_eff * step_dt) / 60.0
            mid_lat_rad = math.radians(last_lat + (d_lat_deg / 2.0))
            cos_lat = max(0.1, math.cos(mid_lat_rad))
            d_lon_deg = (u_eff * step_dt) / (60.0 * cos_lat)

            pred_lat = round(last_lat + d_lat_deg, 4)
            pred_lon = round(last_lon + d_lon_deg, 4)

            # Intensity trend: Only evaluate if genuine environmental parameters exist
            if environmental_sst_c is None or vertical_wind_shear_knots is None:
                # Do NOT perform thermodynamic modulation using assumed values
                intensity_trend = "UNMODULATED (NO_ENVIRONMENTAL_OBS)"
                sim_wind = curr_wind
                sim_pressure = curr_pressure
            else:
                if environmental_sst_c >= 28.5 and vertical_wind_shear_knots < 15.0 and pred_lat < 21.0:
                    # Modest intensification
                    dw = min(15.0, 2.5 * (step_dt / 6.0))
                    sim_wind = min(160.0, sim_wind + dw)
                    sim_pressure = max(890.0, sim_pressure - (dw * 0.75))
                    intensity_trend = "THERMODYNAMIC_INTENSIFICATION (SST >= 28.5C, SHEAR < 15kt)"
                elif vertical_wind_shear_knots > 20.0 or pred_lat >= 21.5:
                    # Land interaction or high shear weakening
                    dw = max(5.0, 4.0 * (step_dt / 6.0))
                    sim_wind = max(25.0, sim_wind - dw)
                    sim_pressure = min(1005.0, sim_pressure + (dw * 0.8))
                    intensity_trend = "THERMODYNAMIC_WEAKENING (HIGH_SHEAR_OR_LANDFALL)"
                else:
                    intensity_trend = "STEADY_INTENSITY"

            # Categorize intensity
            if sim_wind >= 120.0:
                cat = "Super Cyclonic Storm"
            elif sim_wind >= 90.0:
                cat = "Extremely Severe Cyclonic Storm"
            elif sim_wind >= 64.0:
                cat = "Very Severe Cyclonic Storm"
            elif sim_wind >= 48.0:
                cat = "Severe Cyclonic Storm"
            elif sim_wind >= 34.0:
                cat = "Cyclonic Storm"
            else:
                cat = "Deep Depression"

            valid_time = (base_time + timedelta(hours=lead_h)).isoformat()

            # Track heading of this segment
            seg_bearing = (math.degrees(math.atan2(d_lon_deg * cos_lat, d_lat_deg)) + 360.0) % 360.0

            predictions.append({
                "forecast_lead_hours": lead_h,
                "forecast_time_label": f"+{lead_h}h",
                "valid_time": valid_time,
                "latitude": pred_lat,
                "longitude": pred_lon,
                "predicted_wind_speed_knots": round(sim_wind, 1),
                "predicted_pressure_hpa": round(sim_pressure, 1),
                "intensity_category": cat,
                "intensity_trend": intensity_trend,
                "heading_deg": round(seg_bearing, 1),
                "translation_speed_knots": round(math.hypot(u_eff, v_eff), 1),
                "model_name": "physics_guided_advection",
                "model_version": "0.5.0",
                "is_experimental": True,
            })

            last_lat = pred_lat
            last_lon = pred_lon
            last_t = lead_h

        return predictions


trajectory_forecaster = TrajectoryForecaster()

