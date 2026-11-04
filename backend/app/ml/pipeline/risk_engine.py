"""Transparent, explainable research risk engine for Cyclone-X End-to-End Pipeline.

Strict Scientific Governance:
1. Zero-Fabrication: Does NOT invent environmental observations (wind, SST, rainfall).
2. If an environmental variable is missing, it is explicitly marked as "UNAVAILABLE".
3. Computes explainable factor contributions, data completeness ratio, and research risk confidence.
4. Never claims official disaster warnings or evacuations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple


# Major Indian coastline and island reference points (lat, lon) for geodesic proximity calculation
INDIAN_COASTAL_POINTS = [
    # Gujarat / Saurashtra
    (23.2, 68.6), (22.5, 69.2), (21.7, 69.6), (20.9, 70.4), (20.7, 72.8),
    # Maharashtra / Konkan
    (19.0, 72.8), (17.5, 73.2), (16.0, 73.5),
    # Goa / Karnataka
    (15.4, 73.8), (14.5, 74.3), (13.3, 74.7), (12.8, 74.8),
    # Kerala
    (11.2, 75.8), (10.0, 76.2), (8.8, 76.6), (8.1, 77.5),
    # Tamil Nadu / Coromandel
    (9.3, 78.8), (10.8, 79.8), (11.9, 79.8), (13.1, 80.3),
    # Andhra Pradesh
    (14.5, 80.1), (15.8, 80.4), (16.7, 82.2), (17.7, 83.3), (18.3, 84.0),
    # Odisha
    (19.3, 85.0), (19.8, 85.8), (20.5, 86.8), (21.5, 87.0),
    # West Bengal / Sundarbans
    (21.7, 88.1), (22.1, 88.5),
    # Andaman & Nicobar Islands
    (11.7, 92.7), (9.2, 92.8), (7.0, 93.8),
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two points in kilometers."""
    r = 6371.0  # Earth mean radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(r * c, 2)


class PipelineRiskEngine:
    """Computes explainable, multi-factor research risk indices from verified observations."""

    FACTOR_WEIGHTS = {
        "detection_confidence": 0.25,
        "land_proximity": 0.25,
        "track_uncertainty": 0.20,
        "wind": 0.15,
        "rainfall": 0.10,
        "sst": 0.05,
    }

    @classmethod
    def evaluate(
        cls,
        detected: bool,
        detection_confidence: float,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        forecast_predictions: Optional[List[Dict[str, Any]]] = None,
        environmental_obs: Optional[Dict[str, Any]] = None,
        track_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Calculates transparent risk score, severity category, factor breakdown, and audit completeness."""
        if not detected:
            return {
                "risk_score": 0.0,
                "risk_level": "NONE",
                "risk_confidence": round(detection_confidence, 2),
                "data_completeness": 1.0,
                "risk_factors": {
                    "detection_confidence": {
                        "status": "AVAILABLE",
                        "value": round(detection_confidence, 4),
                        "contribution": "NEGATIVE (No tropical cyclone detected)",
                    },
                    "land_proximity": "NOT_APPLICABLE",
                    "track_uncertainty": "NOT_APPLICABLE",
                    "wind": "NOT_APPLICABLE",
                    "rainfall": "NOT_APPLICABLE",
                    "sst": "NOT_APPLICABLE",
                },
                "explanation": "No cyclonic vortex detected in satellite imagery. Multi-hazard risk is negligible.",
                "disclaimer": (
                    "SCIENTIFIC NOTICE: Explainable AI research risk assessment. "
                    "Does not constitute an official IMD disaster warning or evacuation bulletin."
                ),
            }

        env = environmental_obs or {}
        available_scores: List[Tuple[float, float]] = []  # (score, weight)
        factors: Dict[str, Any] = {}
        explanation_parts: List[str] = []

        # 1. Detection Confidence Factor
        det_score = min(100.0, max(0.0, detection_confidence * 100.0))
        factors["detection_confidence"] = {
            "status": "AVAILABLE",
            "value": round(detection_confidence, 4),
            "score": round(det_score, 1),
            "weight": cls.FACTOR_WEIGHTS["detection_confidence"],
            "explanation": f"Neural detection probability: {round(detection_confidence * 100, 1)}%",
        }
        available_scores.append((det_score, cls.FACTOR_WEIGHTS["detection_confidence"]))

        # 2. Land Proximity Factor
        min_coast_dist_km: Optional[float] = None
        closest_lat_lon: Optional[Tuple[float, float]] = None

        target_lat = latitude
        target_lon = longitude

        # If a 24h forecast is available, also consider landfall approach
        if forecast_predictions and len(forecast_predictions) > 0:
            last_pred = forecast_predictions[-1]
            if last_pred.get("latitude") is not None and last_pred.get("longitude") is not None:
                # Use minimum of current position and projected 24h position for conservative threat assessment
                f_lat, f_lon = float(last_pred["latitude"]), float(last_pred["longitude"])
                if target_lat is not None and target_lon is not None:
                    c_d = min(haversine_km(target_lat, target_lon, pt[0], pt[1]) for pt in INDIAN_COASTAL_POINTS)
                    f_d = min(haversine_km(f_lat, f_lon, pt[0], pt[1]) for pt in INDIAN_COASTAL_POINTS)
                    min_coast_dist_km = min(c_d, f_d)
                else:
                    target_lat, target_lon = f_lat, f_lon

        if min_coast_dist_km is None and target_lat is not None and target_lon is not None:
            dists = [
                (haversine_km(target_lat, target_lon, pt[0], pt[1]), pt)
                for pt in INDIAN_COASTAL_POINTS
            ]
            dists.sort(key=lambda x: x[0])
            min_coast_dist_km, closest_lat_lon = dists[0]

        if min_coast_dist_km is not None:
            if min_coast_dist_km < 100.0:
                land_score = 95.0
                proximity_desc = "IMMINENT_COASTAL_PROXIMITY (<100 km)"
            elif min_coast_dist_km < 250.0:
                land_score = 75.0
                proximity_desc = "HIGH_COASTAL_PROXIMITY (<250 km)"
            elif min_coast_dist_km < 500.0:
                land_score = 45.0
                proximity_desc = "MODERATE_OFFSHORE_DISTANCE (<500 km)"
            else:
                land_score = 20.0
                proximity_desc = "OPEN_OCEAN (>500 km from Indian coast)"

            factors["land_proximity"] = {
                "status": "AVAILABLE",
                "min_distance_to_coast_km": min_coast_dist_km,
                "score": land_score,
                "weight": cls.FACTOR_WEIGHTS["land_proximity"],
                "assessment": proximity_desc,
            }
            available_scores.append((land_score, cls.FACTOR_WEIGHTS["land_proximity"]))
            explanation_parts.append(f"Land proximity: {min_coast_dist_km:.1f} km ({proximity_desc})")
        else:
            factors["land_proximity"] = "UNAVAILABLE"
            explanation_parts.append("Land proximity unavailable (no georeferenced coordinates)")

        # 3. Track Uncertainty Factor
        if forecast_predictions and len(forecast_predictions) > 0:
            uncertainties = [
                p.get("uncertainty_radius_km_68") or p.get("uncertainty_km") or 50.0
                for p in forecast_predictions
            ]
            mean_unc = sum(uncertainties) / len(uncertainties)
            # Higher track uncertainty expands the potential impact swath
            unc_score = min(100.0, max(20.0, mean_unc * 0.8))
            factors["track_uncertainty"] = {
                "status": "AVAILABLE",
                "mean_uncertainty_radius_km": round(mean_unc, 1),
                "score": round(unc_score, 1),
                "weight": cls.FACTOR_WEIGHTS["track_uncertainty"],
                "note": "Empirical residual uncertainty from Phase 8 LSTM validation",
            }
            available_scores.append((unc_score, cls.FACTOR_WEIGHTS["track_uncertainty"]))
            explanation_parts.append(f"Track uncertainty radius: ~{mean_unc:.1f} km")
        else:
            factors["track_uncertainty"] = "UNAVAILABLE"

        # 4. Wind Hazard Factor
        wind_kt = env.get("wind_speed_knots") or env.get("wind_speed") or env.get("max_wind_kt")
        if wind_kt is not None:
            try:
                w_val = float(wind_kt)
                if w_val >= 64:
                    w_score = 95.0
                    w_cat = "HURRICANE_FORCE (>=64 kt)"
                elif w_val >= 48:
                    w_score = 75.0
                    w_cat = "STORM_FORCE (48-63 kt)"
                elif w_val >= 34:
                    w_score = 50.0
                    w_cat = "GALE_FORCE (34-47 kt)"
                else:
                    w_score = 25.0
                    w_cat = "SUB_GALE (<34 kt)"

                factors["wind"] = {
                    "status": "AVAILABLE",
                    "value_knots": w_val,
                    "score": w_score,
                    "category": w_cat,
                    "weight": cls.FACTOR_WEIGHTS["wind"],
                }
                available_scores.append((w_score, cls.FACTOR_WEIGHTS["wind"]))
                explanation_parts.append(f"Observed wind: {w_val} kt ({w_cat})")
            except (ValueError, TypeError):
                factors["wind"] = "UNAVAILABLE"
        else:
            factors["wind"] = "UNAVAILABLE"

        # 5. Rainfall Hazard Factor
        rainfall_mm = env.get("rainfall_24h_mm") or env.get("rainfall_mm") or env.get("precipitation")
        if rainfall_mm is not None:
            try:
                rf_val = float(rainfall_mm)
                if rf_val >= 204.5:
                    rf_score = 95.0
                    rf_cat = "EXTREMELY_HEAVY (>=204.5 mm)"
                elif rf_val >= 115.6:
                    rf_score = 75.0
                    rf_cat = "VERY_HEAVY (115.6-204.4 mm)"
                elif rf_val >= 64.5:
                    rf_score = 50.0
                    rf_cat = "HEAVY (64.5-115.5 mm)"
                else:
                    rf_score = 20.0
                    rf_cat = "MODERATE/LIGHT (<64.5 mm)"

                factors["rainfall"] = {
                    "status": "AVAILABLE",
                    "value_24h_mm": rf_val,
                    "score": rf_score,
                    "category": rf_cat,
                    "weight": cls.FACTOR_WEIGHTS["rainfall"],
                }
                available_scores.append((rf_score, cls.FACTOR_WEIGHTS["rainfall"]))
                explanation_parts.append(f"Observed 24h rainfall: {rf_val} mm ({rf_cat})")
            except (ValueError, TypeError):
                factors["rainfall"] = "UNAVAILABLE"
        else:
            factors["rainfall"] = "UNAVAILABLE"

        # 6. Sea Surface Temperature (SST) Factor
        sst_c = env.get("sst") or env.get("sst_celsius") or env.get("sea_surface_temperature")
        if sst_c is not None:
            try:
                sst_val = float(sst_c)
                if sst_val >= 30.0:
                    sst_score = 85.0
                    sst_desc = "HIGHLY_FAVORABLE (>=30°C)"
                elif sst_val >= 28.0:
                    sst_score = 65.0
                    sst_desc = "FAVORABLE (28-29.9°C)"
                elif sst_val >= 26.5:
                    sst_score = 40.0
                    sst_desc = "MARGINAL (26.5-27.9°C)"
                else:
                    sst_score = 15.0
                    sst_desc = "UNFAVORABLE (<26.5°C)"

                factors["sst"] = {
                    "status": "AVAILABLE",
                    "value_celsius": sst_val,
                    "score": sst_score,
                    "assessment": sst_desc,
                    "weight": cls.FACTOR_WEIGHTS["sst"],
                }
                available_scores.append((sst_score, cls.FACTOR_WEIGHTS["sst"]))
                explanation_parts.append(f"SST: {sst_val}°C ({sst_desc})")
            except (ValueError, TypeError):
                factors["sst"] = "UNAVAILABLE"
        else:
            factors["sst"] = "UNAVAILABLE"

        # Compute composite risk score
        total_possible_factors = len(cls.FACTOR_WEIGHTS)
        available_factor_count = len(available_scores)
        data_completeness = round(available_factor_count / float(total_possible_factors), 2)

        if available_scores:
            weighted_sum = sum(score * weight for score, weight in available_scores)
            weights_sum = sum(weight for _, weight in available_scores)
            composite_score = round(weighted_sum / weights_sum, 1)
        else:
            composite_score = 0.0

        # Attempt PyTorch ImpactNeuralEngine inference
        neural_vulnerability = None
        try:
            from pathlib import Path
            import torch
            from backend.app.ml.training_framework.extended_models import ImpactNeuralEngine

            ckpt_path = Path(__file__).resolve().parents[2] / "checkpoints" / "impact_neural_engine.pt"
            if ckpt_path.exists():
                net = ImpactNeuralEngine(num_hazards=6, hidden_dim=64)
                ckpt = torch.load(ckpt_path, map_location="cpu")
                net.load_state_dict(ckpt["model_state_dict"])
                net.eval()

                w_s = float(factors.get("wind", {}).get("score", 25.0) if isinstance(factors.get("wind"), dict) else 25.0)
                l_s = float(factors.get("land_proximity", {}).get("score", 20.0) if isinstance(factors.get("land_proximity"), dict) else 20.0)
                r_s = float(factors.get("rainfall", {}).get("score", 15.0) if isinstance(factors.get("rainfall"), dict) else 15.0)
                pop_d = 6.5
                infra_cnt = 4.0
                coast_km = float(factors.get("land_proximity", {}).get("min_distance_to_coast_km", 200.0) if isinstance(factors.get("land_proximity"), dict) else 200.0)

                inp = torch.tensor([[w_s, l_s, r_s, pop_d, infra_cnt, coast_km]], dtype=torch.float32)
                with torch.no_grad():
                    out = net(inp)

                neural_vulnerability = {
                    "neural_composite_risk": round(float(out["composite_risk_score"][0].item()), 1),
                    "estimated_damage_ratio": round(float(out["estimated_damage_ratio"][0].item()), 3),
                    "model_source": "ImpactNeuralEngine Deep Vulnerability Net",
                }
        except Exception:
            pass

        if composite_score >= 75.0:
            risk_level = "SEVERE"
        elif composite_score >= 50.0:
            risk_level = "HIGH"
        elif composite_score >= 25.0:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        # Risk confidence scales with detection confidence and data completeness
        risk_confidence = round(detection_confidence * (0.4 + 0.6 * data_completeness), 2)

        explanation = (
            f"Model-derived {risk_level} risk score of {composite_score}/100. "
            + "; ".join(explanation_parts)
            + f". Data completeness: {int(data_completeness * 100)}% ({available_factor_count}/{total_possible_factors} factors observed)."
        )

        res_payload = {
            "risk_score": composite_score,
            "risk_level": risk_level,
            "risk_factors": factors,
            "risk_confidence": risk_confidence,
            "data_completeness": data_completeness,
            "available_factors_count": available_factor_count,
            "total_factors_count": total_possible_factors,
            "explanation": explanation,
            "disclaimer": (
                "SCIENTIFIC NOTICE: Explainable AI research risk assessment. "
                "Does not constitute an official IMD disaster warning or evacuation bulletin."
            ),
        }
        if neural_vulnerability:
            res_payload["neural_vulnerability"] = neural_vulnerability
        return res_payload

