"""Rainfall Risk Assessment Module.

Architecture designed to integrate NASA GPM / IMERG and IMD gauge observations.
Strict Zero-Fabrication: If real rainfall data is missing and scenario is inactive,
returns available=False, status="NO_RAINFALL_DATA".
"""

from typing import Any, Dict, Optional
from backend.app.impact.impact_models import ComponentRisk


class RainfallRiskAnalyzer:
    """Evaluates precipitation accumulation and extreme rainfall inundation danger."""

    @staticmethod
    def evaluate(
        measured_rainfall_mm: Optional[float] = None,
        scenario_rainfall_mm: Optional[float] = None,
        is_demo_mode: bool = False,
        demo_scenario_mm: Optional[float] = None,
    ) -> ComponentRisk:
        # 1. Measured real data (Highest priority)
        if measured_rainfall_mm is not None and measured_rainfall_mm >= 0:
            rain_val = measured_rainfall_mm
            source_name = "NASA GPM / IMERG Satellite Ingestion Adapter"
            source_type = "MEASURED OBSERVATION"
            disclaimer = None
            is_avail = True
        # 2. User Scenario Perturbation
        elif scenario_rainfall_mm is not None and scenario_rainfall_mm >= 0:
            rain_val = scenario_rainfall_mm
            source_name = "User Scenario Rainfall Perturbation"
            source_type = "USER PERTURBATION"
            disclaimer = "USER PERTURBED SCENARIO — NOT A MEASURED METEOROLOGICAL OBSERVATION"
            is_avail = True
        # 3. Curated Demo Scenario
        elif is_demo_mode and demo_scenario_mm is not None and demo_scenario_mm >= 0:
            rain_val = demo_scenario_mm
            source_name = "Curated Historical Scenario Dataset (NASA GPM Demo Archive)"
            source_type = "DEMO / SCENARIO"
            disclaimer = "DEMO / SCENARIO RAINFALL — NOT AN OPERATIONAL REAL-TIME MEASUREMENT"
            is_avail = True
        # 4. Data unavailable (Zero-Fabrication Contract)
        else:
            return ComponentRisk(
                available=False,
                risk_level="UNAVAILABLE",
                status="NO_RAINFALL_DATA",
                source="NASA GPM / IMD Gauge Adapter (Disconnected)",
                source_type="UNAVAILABLE",
                method="None",
                explanation="Rainfall risk unavailable — real-time satellite precipitation feed (NASA GPM / IMERG) is not connected.",
                metrics={},
                disclaimer="NO PRECIPITATION DATA CONNECTED — SYSTEM REFUSES TO INVENT RAINFALL NUMBERS",
            )

        # Categorize hazard thresholds based on IMD 24-hour rainfall classification
        # > 204.4 mm: Extremely Heavy Rain
        # 115.6 - 204.4 mm: Very Heavy Rain
        # 64.5 - 115.5 mm: Heavy Rain
        # < 64.5 mm: Moderate / Light Rain
        if rain_val >= 204.5:
            level = "EXTREME"
            impact_desc = "Extremely heavy rainfall (>204 mm/24h) leading to severe urban waterlogging, flash flooding, and catchment saturation."
        elif rain_val >= 115.6:
            level = "HIGH"
            impact_desc = "Very heavy rainfall (115-204 mm/24h) likely triggering localized stream flooding and low-lying inundation."
        elif rain_val >= 64.5:
            level = "MODERATE"
            impact_desc = "Heavy rainfall (64-115 mm/24h) causing agricultural water ponding and traffic disruption."
        else:
            level = "LOW"
            impact_desc = "Isolated moderate showers with minimal hydrological hazard."

        metrics: Dict[str, Any] = {
            "accumulated_24h_rainfall_mm": round(rain_val, 1),
            "imd_rainfall_classification": "Extremely Heavy" if rain_val >= 204.5 else "Very Heavy" if rain_val >= 115.6 else "Heavy" if rain_val >= 64.5 else "Moderate",
        }

        return ComponentRisk(
            available=is_avail,
            risk_level=level,
            status="ACTIVE_ESTIMATE",
            source=source_name,
            source_type=source_type,
            method="IMD 24h Precipitation Scale & Inundation Index",
            explanation=f"{impact_desc} (Derived from {source_type} input: {rain_val:.1f} mm/24h).",
            metrics=metrics,
            disclaimer=disclaimer,
        )


rainfall_risk_analyzer = RainfallRiskAnalyzer()

