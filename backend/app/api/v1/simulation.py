"""What-If Counterfactual Scenario Simulator API.

Module 23: Enables disaster managers and meteorologists to test counterfactual perturbations:
  - Sea Surface Temperature thermal anomalies (Delta SST -2.0C to +3.0C)
  - Deep-layer vertical wind shear perturbations (-20kt to +30kt)
  - Translation speed deceleration (down to 2kt for stall & extreme rainfall accumulation)
  - Landfall approach angle variations for storm-surge inundation recalculation.

Strict Zero-Fabrication: Clearly tagged with provenance disclaimer:
"COUNTERFACTUAL SIMULATION — EXPERIMENTAL WHAT-IF SANDBOX — NOT AN OFFICIAL FORECAST".
"""

import math
from typing import Any, Dict, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/simulation", tags=["Simulation"])


class PerturbationRequest(BaseModel):
    storm_id: Optional[str] = Field(None, description="Optional baseline storm ID")
    base_wind_kt: float = Field(75.0, ge=20.0, le=185.0, description="Baseline maximum sustained wind (kt)")
    base_pressure_hpa: float = Field(975.0, ge=870.0, le=1015.0, description="Baseline central pressure (hPa)")
    delta_sst_c: float = Field(0.0, ge=-3.0, le=4.0, description="Sea Surface Temperature anomaly (deg C)")
    delta_shear_kt: float = Field(0.0, ge=-25.0, le=35.0, description="Deep-layer vertical wind shear perturbation (kt)")
    translation_speed_kt: float = Field(12.0, ge=2.0, le=35.0, description="Forward translation speed (kt)")
    approach_angle_deg: float = Field(60.0, ge=10.0, le=90.0, description="Landfall approach angle to coast (deg)")


class PerturbationResponse(BaseModel):
    baseline: Dict[str, float]
    perturbed: Dict[str, float]
    impact_deltas: Dict[str, Any]
    surge_projection: Dict[str, Any]
    rainfall_amplification: Dict[str, Any]
    scientific_rationale: Dict[str, str]
    provenance: Dict[str, str]


@router.post("/perturb", response_model=PerturbationResponse)
async def simulate_perturbation(req: PerturbationRequest) -> PerturbationResponse:
    # 1. Kerry Emanuel Maximum Potential Intensity (MPI) thermodynamic sensitivity
    # Sensitivity: ~6.8 kt per 1.0 deg C SST anomaly, -0.75 kt per 1.0 kt shear increase
    delta_v_sst = 6.8 * req.delta_sst_c
    delta_v_shear = -0.75 * req.delta_shear_kt
    net_delta_vmax = delta_v_sst + delta_v_shear

    perturbed_wind_kt = float(max(25.0, min(185.0, req.base_wind_kt + net_delta_vmax)))

    # 2. Central Pressure via Atkinson-Holliday empirical delta
    delta_p = -0.72 * (perturbed_wind_kt - req.base_wind_kt)
    perturbed_pressure_hpa = float(max(880.0, min(1012.0, req.base_pressure_hpa + delta_p)))

    # 3. Translation Speed Deceleration & Rainfall Accumulation Factor
    # Forward translation speed inversely affects localized rain accumulation
    # Standard baseline speed = 12 kt. Stalling storms (< 6 kt) amplify accumulation by 2-5x
    rain_factor = round(12.0 / max(2.0, req.translation_speed_kt), 2)
    accum_risk = (
        "CATASTROPHIC_STALL_INUNDATION" if req.translation_speed_kt <= 4.0
        else "HIGH_ACCUMULATION" if req.translation_speed_kt <= 8.0
        else "NORMAL_TRANSIT"
    )

    # 4. Hydrodynamic Surge & Spectral Ocean Wave Recalculation
    angle_rad = math.radians(req.approach_angle_deg)
    base_surge_m = max(0.5, 0.00045 * (req.base_wind_kt ** 2))
    perturbed_surge_m = max(0.5, 0.00045 * (perturbed_wind_kt ** 2) * (math.sin(angle_rad) ** 0.8))

    # Neural Surrogates Evaluation (SurgeDeepSurrogate + WaveWatchSurrogateNet)
    neural_surge_info = None
    neural_wave_info = None
    try:
        from pathlib import Path
        import torch
        from backend.app.ml.training_framework.extended_models import SurgeDeepSurrogate, WaveWatchSurrogateNet

        ckpt_dir = Path(__file__).resolve().parents[2] / "ml" / "checkpoints"
        surge_ckpt = ckpt_dir / "surge_deep_surrogate.pt"
        ww3_ckpt = ckpt_dir / "wavewatch_deep_surrogate.pt"

        # 4a. SurgeDeepSurrogate inference
        if surge_ckpt.exists():
            surge_model = SurgeDeepSurrogate(hidden_dim=64)
            ckpt_s = torch.load(surge_ckpt, map_location="cpu")
            surge_model.load_state_dict(ckpt_s["model_state_dict"])
            surge_model.eval()

            p_drop = max(5.0, 1010.0 - perturbed_pressure_hpa)
            dist_km = 30.0
            slope = 0.0015
            u_onshore = perturbed_wind_kt * math.sin(angle_rad)
            v_onshore = perturbed_wind_kt * math.cos(angle_rad)
            s_inp = torch.tensor([[p_drop, perturbed_wind_kt, dist_km, slope, u_onshore, v_onshore, req.translation_speed_kt]], dtype=torch.float32)

            with torch.no_grad():
                s_out = surge_model(s_inp)
            neural_surge_info = {
                "neural_peak_surge_m": round(float(s_out["surge_height_meters"][0].item()), 2),
                "inundation_extent_km": round(float(s_out["inundation_extent_km"][0].item()), 1),
                "surge_hazard_index": round(float(s_out["surge_hazard_index"][0].item()), 1),
            }

        # 4b. WaveWatchSurrogateNet inference
        if ww3_ckpt.exists():
            ww3_model = WaveWatchSurrogateNet(input_dim=7, hidden_dim=64)
            ckpt_w = torch.load(ww3_ckpt, map_location="cpu")
            ww3_model.load_state_dict(ckpt_w["model_state_dict"])
            ww3_model.eval()

            p_drop = max(5.0, 1010.0 - perturbed_pressure_hpa)
            w_inp = torch.tensor([[perturbed_wind_kt, p_drop, req.translation_speed_kt, 45.0, 60.0, 35.0, angle_rad]], dtype=torch.float32)
            with torch.no_grad():
                w_out = ww3_model(w_inp)
            neural_wave_info = {
                "significant_wave_height_m": round(float(w_out["significant_wave_height_m"][0].item()), 2),
                "peak_wave_period_s": round(float(w_out["peak_wave_period_s"][0].item()), 1),
                "wave_setup_radiation_stress_m": round(float(w_out["wave_setup_meters"][0].item()), 2),
                "wave_energy_flux_kw_m": round(float(w_out["wave_energy_flux_kw_m"][0].item()), 1),
            }
    except Exception:
        pass

    surge_proj = {
        "estimated_surge_m": round(perturbed_surge_m, 2),
        "coastal_angle_effect": f"{req.approach_angle_deg:.0f} deg incident angle",
        "inundation_risk_level": "EXTREME" if perturbed_surge_m >= 3.5 else "HIGH" if perturbed_surge_m >= 2.0 else "MODERATE",
    }
    if neural_surge_info:
        surge_proj["neural_surrogate_surge"] = neural_surge_info
    if neural_wave_info:
        surge_proj["spectral_wave_dynamics"] = neural_wave_info

    rationale = {
        "thermodynamic_mpi": f"Kerry Emanuel MPI scaling: {req.delta_sst_c:+.1f}C SST yields {delta_v_sst:+.1f} kt.",
        "dynamic_shear": f"Deep shear perturbation: {req.delta_shear_kt:+.1f} kt yields {delta_v_shear:+.1f} kt.",
        "barometric_coupling": f"Atkinson-Holliday wind-pressure coupling yields {delta_p:+.1f} hPa shift.",
    }
    if neural_surge_info:
        rationale["hydrodynamic_surrogate"] = (
            f"SurgeDeepSurrogate 2D hydrodynamic emulation: Peak surge {neural_surge_info['neural_peak_surge_m']}m "
            f"with {neural_surge_info['inundation_extent_km']}km coastal inundation penetration."
        )
    if neural_wave_info:
        rationale["wavewatch_spectral_coupling"] = (
            f"WaveWatchSurrogateNet WW3 3rd-Gen Spectral Wave: Hs = {neural_wave_info['significant_wave_height_m']}m, "
            f"Tp = {neural_wave_info['peak_wave_period_s']}s, setup = {neural_wave_info['wave_setup_radiation_stress_m']}m."
        )

    return PerturbationResponse(
        baseline={
            "vmax_knots": req.base_wind_kt,
            "central_pressure_hpa": req.base_pressure_hpa,
            "translation_speed_kt": 12.0,
            "surge_height_m": round(base_surge_m, 2),
        },
        perturbed={
            "vmax_knots": round(perturbed_wind_kt, 1),
            "central_pressure_hpa": round(perturbed_pressure_hpa, 1),
            "translation_speed_kt": req.translation_speed_kt,
            "surge_height_m": round(perturbed_surge_m, 2),
        },
        impact_deltas={
            "delta_vmax_knots": round(perturbed_wind_kt - req.base_wind_kt, 1),
            "delta_pressure_hpa": round(perturbed_pressure_hpa - req.base_pressure_hpa, 1),
            "delta_surge_m": round(perturbed_surge_m - base_surge_m, 2),
            "category_transition": f"{categorize_wind(req.base_wind_kt)} -> {categorize_wind(perturbed_wind_kt)}",
        },
        surge_projection=surge_proj,
        rainfall_amplification={
            "accumulation_multiplier": rain_factor,
            "risk_classification": accum_risk,
            "commentary": f"Translation speed of {req.translation_speed_kt} kt yields {rain_factor}x baseline rain accumulation.",
        },
        scientific_rationale=rationale,
        provenance={
            "source_type": "COUNTERFACTUAL_SIMULATION",
            "disclaimer": "COUNTERFACTUAL SIMULATION — EXPERIMENTAL WHAT-IF SANDBOX — NOT AN OFFICIAL FORECAST",
            "version": "v2.0-neural-surrogates-coupled",
        },
    )


def categorize_wind(kt: float) -> str:
    if kt < 34:
        return "Depression (D)"
    elif kt < 48:
        return "Cyclonic Storm (CS)"
    elif kt < 64:
        return "Severe Cyclonic Storm (SCS)"
    elif kt < 90:
        return "Very Severe Cyclonic Storm (VSCS)"
    elif kt < 120:
        return "Extremely Severe Cyclonic Storm (ESCS)"
    else:
        return "Super Cyclonic Storm (SuCS)"
