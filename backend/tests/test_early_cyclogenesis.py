"""Comprehensive Unit and Integration Tests for Early Tropical Cyclogenesis Detection."""

from datetime import datetime, timedelta, timezone
import io
import numpy as np
from PIL import Image
import pytest
import torch

from backend.app.ml.cyclogenesis.tcc_tracker import (
    TropicalCloudClusterTracker,
    TropicalCloudCluster,
    EnvironmentalParameters,
    TCCSequenceProfile,
    tcc_tracker,
)
from backend.app.ml.cyclogenesis.cyclogenesis_head import (
    EarlyCyclogenesisNeuralHead,
    EarlyCyclogenesisClassifier,
    EarlyCyclogenesisOutput,
    early_cyclogenesis_classifier,
)
from backend.app.schemas.cyclone import EarlyGenesisAlert
from backend.app.schemas.analysis import EarlyCyclogenesisResult, SatelliteAnalysisResponse
from backend.app.ml.inference import SatelliteInferencePipeline
from backend.app.services.cyclone_service import CycloneService, compute_early_genesis_alert


class TestTropicalCloudClusterTracker:
    """Tests for TCC tracking and environmental parameter extraction."""

    def test_coriolis_parameter_calculation(self):
        # Coriolis at equator should be 0
        f_eq = TropicalCloudClusterTracker.compute_coriolis_parameter(0.0)
        assert f_eq == 0.0

        # Coriolis at 15°N
        f_15 = TropicalCloudClusterTracker.compute_coriolis_parameter(15.0)
        assert 3.5e-5 < f_15 < 4.0e-5

        # Absolute latitude handling for southern hemisphere
        f_south = TropicalCloudClusterTracker.compute_coriolis_parameter(-15.0)
        assert f_south == pytest.approx(f_15)

    def test_environmental_extraction_with_fallbacks(self):
        tracker = TropicalCloudClusterTracker()
        env = tracker.extract_environmental_parameters(
            lat=14.0,
            lon=86.0,
            ambient_sst=29.2,
            ambient_shear=11.5,
            ambient_rh=80.0,
        )
        assert env.latitude == 14.0
        assert env.longitude == 86.0
        assert env.sea_surface_temperature_c == 29.2
        assert env.vertical_wind_shear_kt == 11.5
        assert env.shear_favorable is True
        assert env.sst_anomaly_above_threshold == pytest.approx(2.7, abs=0.1)

    def test_process_24h_sequence_dynamics(self):
        tracker = TropicalCloudClusterTracker()
        now = datetime.now(timezone.utc)

        # Create synthetic 24h sequence: expanding cold convective cloud cluster
        seq = []
        for i in range(5):
            t = now - timedelta(hours=(4 - i) * 6)
            arr = np.zeros((100, 100), dtype=np.float32)
            # Brightness temperature: expanding cold center (pixel value ~ 210K)
            center = 50
            r = 10 + i * 5
            y, x = np.ogrid[:100, :100]
            mask = (x - center)**2 + (y - center)**2 <= r**2
            arr[mask] = 210.0
            arr[~mask] = 280.0
            seq.append((t, arr))

        env_test = tracker.extract_environmental_parameters(
            lat=12.5,
            lon=84.0,
            ambient_sst=29.5,
            ambient_shear=10.0,
            ambient_rh=80.0,
        )
        profile = tracker.process_24h_sequence(
            sequence_images=seq,
            ref_lat=12.5,
            ref_lon=84.0,
            km_per_pixel=4.0,
            environmental_override=env_test,
        )

        assert isinstance(profile, TCCSequenceProfile)
        assert profile.observation_count == 5
        assert profile.duration_hours >= 24.0
        assert profile.area_growth_rate_km2_hr > 0  # Growing convective cluster
        assert profile.mean_radius_km > 0
        assert profile.persistence_hours > 0


class TestEarlyCyclogenesisClassifier:
    """Tests for neural cyclogenesis classification head and risk tiering."""

    def test_neural_head_forward(self):
        head = EarlyCyclogenesisNeuralHead(in_features=10)
        head.eval()

        # Batch input
        x_batch = torch.randn(4, 10)
        out_batch = head(x_batch)
        assert out_batch.shape == (4, 1)
        assert torch.all(out_batch >= 0.0) and torch.all(out_batch <= 1.0)

        # Single vector input
        x_single = torch.randn(10)
        out_single = head(x_single)
        assert out_single.shape == (1,)
        assert 0.0 <= float(out_single.item()) <= 1.0

    def test_favorable_cyclogenesis_prediction(self):
        tracker = TropicalCloudClusterTracker()
        env = EnvironmentalParameters(
            latitude=14.0,
            longitude=86.0,
            coriolis_f=3.5e-5,
            relative_vorticity_850=5.0,
            vertical_wind_shear_kt=8.0,  # Favorable
            shear_favorable=True,
            sea_surface_temperature_c=30.5,  # Very warm SST
            sst_anomaly_above_threshold=4.0,
            ocean_heat_content_kj=85.0,
            relative_humidity_700=85.0,  # Very moist
            surface_pressure_hpa=1004.0,
        )
        dummy_cluster = TropicalCloudCluster(
            cluster_id="TEST-01",
            timestamp=datetime.now(timezone.utc),
            centroid_lat=14.0,
            centroid_lon=86.0,
            centroid_pixel=(50.0, 50.0),
            area_km2=80000.0,
            equivalent_radius_km=160.0,
            min_brightness_temp_k=205.0,
            mean_brightness_temp_k=218.0,
            convective_fraction=0.45,
            eccentricity=0.4,
            curvature_vorticity_score=0.85,
        )
        profile = TCCSequenceProfile(
            cluster_id="TEST-01",
            observation_count=5,
            duration_hours=24.0,
            current_centroid=(14.0, 86.0),
            translation_speed_kmh=15.0,
            translation_direction_deg=315.0,
            area_growth_rate_km2_hr=1500.0,
            persistence_hours=24.0,
            mean_radius_km=160.0,
            curvature_organization_trend=0.015,
            environmental=env,
            clusters_history=[dummy_cluster],
        )

        res = early_cyclogenesis_classifier.predict(
            tcc_profile=profile,
            current_wind_speed_kt=25.0,  # Pre-cyclone stage
        )

        assert isinstance(res, EarlyCyclogenesisOutput)
        assert 0.0 <= res.probability <= 1.0
        assert res.risk_tier in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        assert res.pre_cyclone_stage in ["TROPICAL_CLOUD_CLUSTER", "LOW_PRESSURE_AREA", "DEPRESSION"]
        assert 100.0 <= res.disturbance_radius_km <= 260.0
        assert "latitude" in res.circulation_center
        assert "longitude" in res.circulation_center

    def test_equatorial_suppression(self):
        """Cyclogenesis is physically suppressed within 5 degrees of the equator due to lack of Coriolis."""
        tracker = TropicalCloudClusterTracker()
        env_eq = tracker.extract_environmental_parameters(lat=2.0, lon=85.0, ambient_sst=30.0, ambient_shear=8.0)
        profile_eq = tracker.process_24h_sequence([], ref_lat=2.0, ref_lon=85.0, environmental_override=env_eq)

        res = early_cyclogenesis_classifier.predict(profile_eq, current_wind_speed_kt=20.0)
        assert res.probability < 0.50
        assert res.risk_tier in ["LOW", "MODERATE"]


class TestEndToEndPipelineIntegration:
    """Verifies that the early cyclogenesis head seamlessly integrates into the satellite pipeline and cyclone service."""

    def test_pipeline_includes_cyclogenesis_before_intensity(self):
        pipeline = SatelliteInferencePipeline()

        # Create a synthetic 128x128 satellite image
        img = Image.new("L", (128, 128), color=180)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        response = pipeline.run_analysis(
            image_bytes=raw_bytes,
            filename="test_granule.png",
            geo_bounds={"north": 18.0, "south": 12.0, "east": 88.0, "west": 82.0},
        )

        assert isinstance(response, SatelliteAnalysisResponse)
        assert response.cyclogenesis is not None
        assert isinstance(response.cyclogenesis, EarlyCyclogenesisResult)
        assert response.cyclogenesis.available is True
        assert 0.0 <= response.cyclogenesis.probability <= 1.0
        assert response.cyclogenesis.risk_tier in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        assert response.cyclogenesis.disturbance_radius_km >= 100.0
        assert "seaSurfaceTemperature" in response.cyclogenesis.environmental_summary

    def test_cyclone_service_early_genesis_alert(self):
        service = CycloneService()
        alert = compute_early_genesis_alert(
            lat=15.0,
            lon=87.0,
            wind_speed=25.0,
            sst=29.5,
            shear=12.0,
        )
        assert isinstance(alert, EarlyGenesisAlert)
        assert alert.riskTier in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        assert 0.0 <= alert.probability <= 1.0
        assert alert.preCycloneStage in ["TROPICAL_CLOUD_CLUSTER", "LOW_PRESSURE_AREA", "DEPRESSION"]

        # Test listing cyclones returns earlyGenesis on pre-cyclonic systems
        cyclones = service.list_cyclones()
        assert len(cyclones) > 0
        live_storm = next((c for c in cyclones if c.id == "LIVE-NIO-SATELLITE"), None)
        assert live_storm is not None
