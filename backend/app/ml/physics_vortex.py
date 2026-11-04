"""Holland (1980) Tropical Cyclone Physical Vortex Engine.

Computes:
1. Coriolis parameter f as a function of latitude.
2. Holland shape parameter B bounded in [1.0, 2.5].
3. Radius of Maximum Wind (RMW) via Knaff & Zehr empirical formulation.
4. Continuous radial wind profile curve V(r) from 0 to 500 km.
5. Quadrant gale radii (R34, R50, R64) across NE, SE, SW, NW factoring translation asymmetry.
"""

import math
from typing import Dict, List, Optional, Tuple


class HollandVortexEngine:
    """Physics-based mathematical model of tropical cyclone wind-pressure vortex structure."""

    OMEGA = 7.2921159e-5  # Earth's angular velocity in rad/s
    AIR_DENSITY = 1.15     # Tropical maritime air density at sea level in kg/m^3
    KNOTS_TO_MS = 0.514444
    MS_TO_KNOTS = 1.943844
    KM_TO_METERS = 1000.0
    METERS_TO_KM = 0.001

    @classmethod
    def compute_coriolis(cls, latitude: float) -> float:
        """Compute Coriolis parameter f = 2 * Omega * sin(lat)."""
        lat_rad = math.radians(abs(latitude))
        return 2.0 * cls.OMEGA * math.sin(lat_rad)

    @classmethod
    def compute_holland_b(cls, delta_p: float, latitude: float) -> float:
        """Calculate Holland shape parameter B.

        B = 1.0 + 0.0025 * delta_p - 0.0014 * |lat|
        Physically bounded in [1.0, 2.5].
        """
        raw_b = 1.0 + 0.0025 * max(0.0, delta_p) - 0.0014 * abs(latitude)
        return max(1.0, min(2.5, round(raw_b, 3)))

    @classmethod
    def estimate_rmw(cls, delta_p: float, latitude: float) -> float:
        """Estimate Radius of Maximum Wind (RMW) in kilometers.

        Uses Knaff & Zehr (2007) parametric regression:
        ln(Rmax_nm) = 2.636 - 0.00005086 * delta_p^2 + 0.039489 * |lat|
        Converted to km (1 nm = 1.852 km) and bounded within [15.0, 80.0] km.
        """
        dp = max(5.0, delta_p)
        lat = min(60.0, max(5.0, abs(latitude)))
        ln_rmw_nm = 2.636 - 0.00005086 * (dp ** 2) + 0.039489 * lat
        rmw_nm = math.exp(ln_rmw_nm)
        rmw_km = rmw_nm * 1.852
        return round(max(15.0, min(80.0, rmw_km)), 1)

    @classmethod
    def compute_radial_wind(
        cls,
        radius_km: float,
        central_pressure: float,
        environmental_pressure: float,
        rmw_km: float,
        b_param: float,
        latitude: float,
    ) -> float:
        """Calculate gradient wind speed V(r) at a given radius in knots.

        Holland (1980) Gradient Wind Formula:
        V(r) = sqrt( (B/rho) * (Rmax/r)^B * delta_P * exp(-(Rmax/r)^B) + (r*f/2)^2 ) - (r*f/2)
        """
        if radius_km <= 0.1:
            return 0.0

        delta_p_hpa = max(1.0, environmental_pressure - central_pressure)
        delta_p_pa = delta_p_hpa * 100.0  # convert hPa to Pascals (N/m^2)

        r_m = radius_km * cls.KM_TO_METERS
        rmw_m = rmw_km * cls.KM_TO_METERS
        f = cls.compute_coriolis(latitude)

        ratio = rmw_m / r_m
        exp_term = math.exp(-pow(ratio, b_param))

        pressure_term = (b_param / cls.AIR_DENSITY) * pow(ratio, b_param) * delta_p_pa * exp_term
        coriolis_term = pow(0.5 * r_m * f, 2)

        v_grad_ms = math.sqrt(max(0.0, pressure_term + coriolis_term)) - (0.5 * r_m * f)
        v_knots = v_grad_ms * cls.MS_TO_KNOTS
        return max(0.0, round(v_knots, 1))

    @classmethod
    def generate_radial_profile(
        cls,
        central_pressure: float,
        environmental_pressure: float = 1012.0,
        latitude: float = 15.0,
        custom_rmw_km: Optional[float] = None,
        max_distance_km: float = 400.0,
        step_km: float = 5.0,
    ) -> Dict:
        """Generates continuous radial wind profile samples from r=0 to max_distance_km."""
        delta_p = max(1.0, environmental_pressure - central_pressure)
        b_param = cls.compute_holland_b(delta_p, latitude)
        rmw_km = custom_rmw_km if custom_rmw_km else cls.estimate_rmw(delta_p, latitude)

        curve: List[Dict[str, float]] = []
        r = 0.0
        v_max = 0.0
        v_max_radius = rmw_km

        while r <= max_distance_km:
            v = cls.compute_radial_wind(
                radius_km=r,
                central_pressure=central_pressure,
                environmental_pressure=environmental_pressure,
                rmw_km=rmw_km,
                b_param=b_param,
                latitude=latitude,
            )
            if v > v_max:
                v_max = v
                v_max_radius = r
            curve.append({"radius_km": round(r, 1), "wind_speed_knots": v})
            r += step_km

        # Ensure exact RMW sample is recorded
        v_rmw = cls.compute_radial_wind(
            radius_km=rmw_km,
            central_pressure=central_pressure,
            environmental_pressure=environmental_pressure,
            rmw_km=rmw_km,
            b_param=b_param,
            latitude=latitude,
        )

        return {
            "central_pressure_hpa": central_pressure,
            "environmental_pressure_hpa": environmental_pressure,
            "central_pressure_deficit_hpa": round(delta_p, 1),
            "holland_b": b_param,
            "radius_max_wind_km": rmw_km,
            "peak_gradient_wind_knots": round(v_rmw, 1),
            "curve": curve,
        }

    @classmethod
    def compute_quadrant_gale_radii(
        cls,
        central_pressure: float,
        environmental_pressure: float = 1012.0,
        latitude: float = 15.0,
        translation_speed_knots: float = 8.0,
        translation_direction_deg: float = 315.0,  # Typically NW in North Indian Ocean
        custom_rmw_km: Optional[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate gale wind radii (R34, R50, R64 in km) across four quadrants (NE, SE, SW, NW).

        Applies translation asymmetry: Winds in the forward-right quadrant are enhanced
        by the forward translation speed vector addition.
        """
        delta_p = max(1.0, environmental_pressure - central_pressure)
        b_param = cls.compute_holland_b(delta_p, latitude)
        rmw_km = custom_rmw_km if custom_rmw_km else cls.estimate_rmw(delta_p, latitude)

        # Quadrant azimuth centers in degrees: NE=45, SE=135, SW=225, NW=315
        quadrant_azimuths = {
            "NE": 45.0,
            "SE": 135.0,
            "SW": 225.0,
            "NW": 315.0,
        }

        # Asymmetry factor: Forward-right quadrant in Northern Hemisphere experiences
        # maximum constructive vector addition: V_total = V_rotational + alpha * V_trans * cos(theta - theta_max)
        # where theta_max = translation_dir + 90 deg (to the right of storm motion).
        max_asym_azimuth = (translation_direction_deg + 90.0) % 360.0
        alpha = 0.55  # Translation wind reduction to surface boundary layer

        quadrant_results: Dict[str, Dict[str, float]] = {}

        for quad, az in quadrant_azimuths.items():
            # Angular offset relative to maximum wind azimuth
            angle_diff = math.radians(az - max_asym_azimuth)
            v_trans_component = alpha * translation_speed_knots * math.cos(angle_diff)

            # Search outward from RMW to find R64, R50, R34 thresholds
            r64 = 0.0
            r50 = 0.0
            r34 = 0.0

            # Scan outward up to 450 km in 2 km increments
            for r in range(int(rmw_km), 450, 2):
                r_km = float(r)
                v_rot = cls.compute_radial_wind(
                    radius_km=r_km,
                    central_pressure=central_pressure,
                    environmental_pressure=environmental_pressure,
                    rmw_km=rmw_km,
                    b_param=b_param,
                    latitude=latitude,
                )
                v_eff = v_rot + v_trans_component

                if v_eff >= 64.0 and r_km > r64:
                    r64 = r_km
                if v_eff >= 50.0 and r_km > r50:
                    r50 = r_km
                if v_eff >= 34.0 and r_km > r34:
                    r34 = r_km

            quadrant_results[quad] = {
                "r34_km": round(r34, 1),
                "r50_km": round(r50, 1),
                "r64_km": round(r64, 1),
            }

        return quadrant_results


holland_vortex_engine = HollandVortexEngine()

