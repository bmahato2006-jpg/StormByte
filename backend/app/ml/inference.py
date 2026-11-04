"""Inference Orchestrator Module for Satellite AI Analysis.

Executes the end-to-end scientific computer-vision pipeline:
1. Preprocessing & Quality Assessment
2. Candidate Storm Region Detection
3. Center Localization & Optional Georeferencing
4. Morphological Classification
5. Intensity Capability Verification
6. Explainable AI & Overlay Rendering
7. Provenance Tracking & Immutable Auditing
"""

from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import uuid
import numpy as np
from PIL import Image
import torch

from backend.app.schemas.analysis import (
    AnalysisMode,
    ConfidenceTier,
    EarlyCyclogenesisResult,
    SatelliteAnalysisResponse,
)
from backend.app.ml.preprocessing import (
    validate_image_bytes,
    preprocess_satellite_image,
)
from backend.app.ml.detector import cyclone_detector
from backend.app.ml.center_detection import center_detector
from backend.app.ml.classification import cyclone_classifier
from backend.app.ml.intensity import intensity_estimator
from backend.app.ml.explainability import explainability_engine
from backend.app.ml.model_registry import model_registry
from backend.app.ml.detection.inference import (
    detection_inference_service_v2,
    detection_inference_service_v3,
)
from backend.app.ml.cyclogenesis.tcc_tracker import tcc_tracker
from backend.app.ml.cyclogenesis.cyclogenesis_head import early_cyclogenesis_classifier
from backend.app.data.provenance.tracker import provenance_tracker
from backend.app.data.schemas.observations import DataMode

logger = logging.getLogger("cyclone_x")


class SatelliteInferencePipeline:
    """Orchestrates scientific satellite image analysis and preserves complete audit trails."""

    def __init__(self):
        # In-memory store for fast retrieval of analysis runs
        self._analysis_cache: Dict[str, SatelliteAnalysisResponse] = {}

    def get_analysis(self, analysis_id: str) -> Optional[SatelliteAnalysisResponse]:
        return self._analysis_cache.get(analysis_id)

    def run_analysis(
        self,
        image_bytes: bytes,
        filename: str = "satellite_granule.png",
        cyclone_id: Optional[str] = None,
        scan_timestamp: Optional[datetime] = None,
        source_name: Optional[str] = None,
        source_type: str = "USER_UPLOAD",
        analysis_mode: AnalysisMode = AnalysisMode.RESEARCH,
        geo_bounds: Optional[Dict[str, float]] = None
    ) -> SatelliteAnalysisResponse:
        """Execute the end-to-end computer-vision analysis pipeline."""
        # 1. Validation
        is_valid, err_msg = validate_image_bytes(image_bytes, filename)
        if not is_valid:
            raise ValueError(err_msg)

        # 2. Cryptographic hash for provenance auditing
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        analysis_id = f"anl-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        obs_time = scan_timestamp if scan_timestamp else now

        # 3. Preprocessing & Quality Assessment
        pil_img, gray_arr, prep_meta = preprocess_satellite_image(
            image_bytes,
            target_size=(512, 512),
            enhance_contrast=True
        )

        # 4. Candidate Cyclone Detection
        # Strict zero-fake-positive rule: Use the trained PyTorch CNN model (v3 NIO benchmark / v2)
        det_service = (
            detection_inference_service_v3
            if detection_inference_service_v3.is_available()
            else (
                detection_inference_service_v2
                if detection_inference_service_v2.is_available()
                else None
            )
        )

        cnn_evaluated = False
        cnn_detected = False
        cnn_confidence = 0.0
        active_model = "baseline_cv"
        model_ver = "0.4.0"

        if det_service is not None:
            try:
                pred_res = det_service.predict(image_bytes, data_mode="UPLOAD", threshold=0.50)
                cnn_detected = pred_res.get("detected", False)
                cnn_confidence = float(pred_res.get("confidence", 0.0))
                active_model = det_service.model_name
                model_ver = det_service.model_version
                cnn_evaluated = True
            except Exception as e:
                logger.warning("CNN detector evaluation fallback to algorithmic baseline: %s", e)

        # Algorithmic detection for cloud mask and geometric metrics
        det_result = cyclone_detector.detect(gray_arr)
        cloud_mask = det_result["cloud_mask"]
        spiral_score = det_result["spiral_index"]
        area_ratio = det_result["convective_area_ratio"]

        if cnn_evaluated:
            if cnn_detected:
                # Meteorological physical reality: A tropical cyclonic storm cannot exist with < 6% cloud cover in frame
                if area_ratio < 0.06:
                    logger.info("CNN detection rejected: scene cloud coverage (%.1f%%) too sparse for cyclonic vortex.", area_ratio * 100)
                    detected = False
                    det_conf_tier = ConfidenceTier.LOW
                else:
                    detected = True
                    if cnn_confidence >= 0.75:
                        det_conf_tier = ConfidenceTier.HIGH
                    elif cnn_confidence >= 0.45:
                        det_conf_tier = ConfidenceTier.MEDIUM
                    else:
                        det_conf_tier = ConfidenceTier.LOW
            elif det_result["detected"] and spiral_score >= 0.58 and area_ratio >= 0.12:
                # Monochromatic / infrared satellite image where physical spiral curvature confirms vortex
                detected = True
                det_conf_tier = ConfidenceTier.HIGH if spiral_score >= 0.65 else ConfidenceTier.MEDIUM
            else:
                detected = False
                det_conf_tier = ConfidenceTier.HIGH if cnn_confidence >= 0.75 else ConfidenceTier.MEDIUM
            bbox = det_result["bounding_box"] if detected else None
        else:
            detected = det_result["detected"]
            bbox = det_result["bounding_box"] if detected else None
            det_conf_tier = det_result["confidence"]

        # 5. Center Detection & Georeferencing
        center_res = center_detector.locate_center(
            gray_array=gray_arr,
            candidate_mask=cloud_mask if detected else None,
            geo_bounds=geo_bounds,
            detected=detected,
        )

        # 5b. Meteorological Physical Verification Guardrail
        # Strictly enforce physical atmospheric laws:
        # A tropical cyclonic vortex requires an active oceanic convective cloud core.
        # Clear-sky continental terrain or landmasses with low cloud cover cannot physically be active cyclones.
        rejection_reason = None
        if detected:
            is_physically_valid, rejection_reason = self._validate_cyclone_physical_signature(
                pil_img=pil_img,
                center_x=center_res.pixel_x,
                center_y=center_res.pixel_y,
                radius=center_res.estimated_radius_pixels,
                cloud_mask=cloud_mask,
            )
            if not is_physically_valid:
                logger.info("Cyclone candidate rejected by physical guardrail: %s", rejection_reason)
                detected = False
                det_conf_tier = ConfidenceTier.HIGH
                bbox = None

        if detected and bbox is None:
            box_r = max(40.0, center_res.estimated_radius_pixels * 1.5)
            w_dim = prep_meta.processed_dimensions["width"]
            h_dim = prep_meta.processed_dimensions["height"]
            bbox = {
                "x_min": int(max(0, center_res.pixel_x - box_r)),
                "y_min": int(max(0, center_res.pixel_y - box_r)),
                "x_max": int(min(w_dim, center_res.pixel_x + box_r)),
                "y_max": int(min(h_dim, center_res.pixel_y + box_r)),
                "width": int(min(w_dim, box_r * 2)),
                "height": int(min(h_dim, box_r * 2)),
            }

        # 6. Classification
        class_res = cyclone_classifier.classify(
            detected=detected,
            area_ratio=area_ratio,
            spiral_index=spiral_score,
            symmetry_score=center_res.confidence_score if detected else 0.0,
            image_array=gray_arr if detected else None,
        )
        if not detected and rejection_reason:
            class_res.criteria_summary = rejection_reason

        # 6b. Early Cyclogenesis Classification Head (Evaluated before intensity regression)
        # Evaluates early pre-cyclonic disturbances (TCCs, LPAs, Depressions, wind < 34 kt)
        # and predicts 24-48h cyclogenesis probability.
        ref_lat = center_res.latitude if center_res.latitude is not None else (
            (geo_bounds["north"] + geo_bounds["south"]) / 2.0 if geo_bounds else 14.5
        )
        ref_lon = center_res.longitude if center_res.longitude is not None else (
            (geo_bounds["east"] + geo_bounds["west"]) / 2.0 if geo_bounds else 85.0
        )

        tcc_profile = tcc_tracker.process_24h_sequence(
            sequence_images=[(obs_time, gray_arr)],
            ref_lat=ref_lat,
            ref_lon=ref_lon,
            km_per_pixel=4.0,
        )

        # Baseline wind speed proxy by classification category prior to regression
        current_wind = 25.0
        if class_res.category in ["DISTURBANCE", "DEPRESSION"]:
            current_wind = 25.0
        elif class_res.category == "CYCLONIC_STORM":
            current_wind = 40.0
        elif class_res.category in [
            "SEVERE_CYCLONIC_STORM",
            "VERY_SEVERE_CYCLONIC_STORM",
            "EXTREMELY_SEVERE_CYCLONIC_STORM",
            "SUPER_CYCLONIC_STORM",
        ]:
            current_wind = 65.0

        genesis_output = early_cyclogenesis_classifier.predict(
            tcc_profile=tcc_profile,
            current_wind_speed_kt=current_wind,
            manual_curvature_score=spiral_score,
        )

        cyclogenesis_res = EarlyCyclogenesisResult(
            available=True,
            probability=genesis_output.probability,
            probability_percent=genesis_output.probability_percent,
            risk_tier=genesis_output.risk_tier,
            intensification_24to48h=genesis_output.intensification_24to48h,
            pre_cyclone_stage=genesis_output.pre_cyclone_stage,
            disturbance_radius_km=genesis_output.disturbance_radius_km,
            circulation_center=genesis_output.circulation_center,
            headline=genesis_output.headline,
            environmental_summary=genesis_output.environmental_summary,
        )

        # 7. Intensity Estimation
        intensity_res = intensity_estimator.estimate(
            gray_array=gray_arr,
            center_x=center_res.pixel_x,
            center_y=center_res.pixel_y,
            classification_category=class_res.category,
        )

        # 8. Explainable AI & Detection Overlay
        xai_res = explainability_engine.explain(
            analysis_id=analysis_id,
            dimensions=(prep_meta.processed_dimensions["width"], prep_meta.processed_dimensions["height"]),
            bbox=bbox,
            center_x=center_res.pixel_x,
            center_y=center_res.pixel_y,
            radius=center_res.estimated_radius_pixels,
            detected=detected,
            spiral_score=spiral_score,
            symmetry_score=center_res.confidence_score,
            cloud_mask=cloud_mask,
            model_version=model_ver,
        )

        # 9. Register Provenance Record
        prov_mode = (
            DataMode.DEMO if analysis_mode == AnalysisMode.DEMO else DataMode.USER_UPLOAD
        )
        prov_rec = provenance_tracker.record_ingestion(
            source=source_name if source_name else "SATELLITE_AI_ANALYZER",
            provider="CYCLONE-X Computer Vision Subsystem",
            product=f"Satellite Analysis ({class_res.category})",
            data_mode=prov_mode,
            status="ANALYSIS_COMPLETED",
            source_reference=f"Hash: {image_hash[:16]}... | File: {filename}",
            dataset_version=f"CV-v{model_ver}",
            acquisition_time=obs_time,
            license_notes="AI-Generated Experimental Analysis — Research Use Only",
        )

        # 10. Assemble Final Structured Response
        src_label = source_name if source_name else (
            f"DEMO_GRANULE_{cyclone_id}" if cyclone_id else "USER_UPLOADED_IMAGE"
        )

        response = SatelliteAnalysisResponse(
            analysis_id=analysis_id,
            source=src_label,
            source_type=source_type,
            timestamp=obs_time,
            processing_timestamp=now,
            image_hash=image_hash,
            analysis_mode=analysis_mode,
            model_name=active_model,
            model_version=model_ver,
            provenance_id=prov_rec.id,
            preprocessing=prep_meta,
            detection_detected=detected,
            detection_confidence=det_conf_tier,
            candidate_bounding_box=bbox,
            center=center_res,
            classification=class_res,
            intensity=intensity_res,
            cyclogenesis=cyclogenesis_res,
            explainability=xai_res,
            scientific_disclaimer=(
                "AI-generated experimental analysis. Not an official IMD warning or operational forecast."
            ),
        )

        # Cache result
        self._analysis_cache[analysis_id] = response
        return response

    @staticmethod
    def _validate_cyclone_physical_signature(
        pil_img: Image.Image,
        center_x: float,
        center_y: float,
        radius: float,
        cloud_mask: np.ndarray,
    ) -> Tuple[bool, Optional[str]]:
        """Validates that candidate center conforms to physical tropical cyclone meteorology.

        Rejects:
        1. Cloud-free continental landmass / terrain (high color saturation, low cloud cover).
        2. Centers lacking an organized convective cloud core (core cloud fraction < 20%).
        """
        rgb_img = pil_img.convert("RGB")
        w_px, h_px = rgb_img.size
        arr_rgb = np.array(rgb_img, dtype=np.float32) / 255.0

        # Monochromatic check (raw grayscale thermal infrared vs visible/optical color)
        diff = np.mean(np.abs(arr_rgb[:, :, 0] - arr_rgb[:, :, 1])) + np.mean(
            np.abs(arr_rgb[:, :, 1] - arr_rgb[:, :, 2])
        )
        is_monochrome = diff < (12.0 / 255.0)

        # Color saturation
        max_c = np.max(arr_rgb, axis=2)
        min_c = np.min(arr_rgb, axis=2)
        sat = (max_c - min_c) / (max_c + 1e-6)

        # Check core region around candidate center in mask space
        r = max(30, int(radius))
        y0 = max(0, int(center_y - r))
        y1 = min(cloud_mask.shape[0], int(center_y + r))
        x0 = max(0, int(center_x - r))
        x1 = min(cloud_mask.shape[1], int(center_x + r))

        core_cloud_patch = cloud_mask[y0:y1, x0:x1]
        core_cloud_frac = float(np.mean(core_cloud_patch)) if core_cloud_patch.size > 0 else 0.0

        # Check corresponding region in original RGB image
        sy0 = int(y0 * h_px / cloud_mask.shape[0])
        sy1 = int(y1 * h_px / cloud_mask.shape[0])
        sx0 = int(x0 * w_px / cloud_mask.shape[1])
        sx1 = int(x1 * w_px / cloud_mask.shape[1])
        patch_sat = sat[sy0:sy1, sx0:sx1]
        core_sat = float(np.mean(patch_sat)) if patch_sat.size > 0 else 0.0

        opt_clouds = (max_c[sy0:sy1, sx0:sx1] > 0.40) & (patch_sat < 0.25)
        opt_cloud_frac = float(np.mean(opt_clouds)) if opt_clouds.size > 0 else 0.0

        # Physical Rule 1: A tropical cyclone vortex core must have organized convective clouds (>= 20%).
        # If candidate center lacks convective clouds, it is non-cyclonic background / clear sky.
        if core_cloud_frac < 0.20:
            if not is_monochrome and core_sat >= 0.35:
                return (
                    False,
                    f"Candidate center is situated over cloud-free continental terrain (mean saturation: {core_sat:.2f}, "
                    f"core convective cloud cover: {core_cloud_frac * 100:.1f}%). Non-cyclonic terrestrial scene.",
                )
            return (
                False,
                f"Candidate center lacks an organized convective cloud core (cloud cover: {core_cloud_frac * 100:.1f}% < 20%). "
                "Non-cyclonic atmospheric environment.",
            )

        return True, None


inference_pipeline = SatelliteInferencePipeline()


def run_detection_pipeline(
    image_or_tensor: Union[torch.Tensor, np.ndarray, Image.Image, bytes, str, Path],
    threshold: float = 0.50,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Production detection pipeline for satellite imagery or normalized PyTorch tensors.

    Accepts:
        - PyTorch tensor of shape [1, H, W], [C, H, W], or [H, W] normalized in [0, 1]
        - NumPy ndarray
        - PIL Image or raw image bytes

    Returns structured detection dict:
        - detected: bool
        - confidence: float
        - predicted_label: str ("CYCLONE" | "NO_CYCLONE")
        - model_name: str
        - model_version: str
        - details: Dict[str, Any]
    """
    det_service = (
        detection_inference_service_v3
        if detection_inference_service_v3.is_available()
        else (
            detection_inference_service_v2
            if detection_inference_service_v2.is_available()
            else None
        )
    )
    if det_service is None:
        raise RuntimeError("No trained cyclone detection model available.")

    pil_img: Image.Image
    if isinstance(image_or_tensor, torch.Tensor):
        t = image_or_tensor.detach().cpu().float()
        if t.ndim == 4:
            t = t.squeeze(0)
        if t.ndim == 3:
            if t.shape[0] == 1:
                t = t.squeeze(0)
            elif t.shape[0] == 3:
                t_arr = (t.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
                pil_img = Image.fromarray(t_arr, mode="RGB")
        if t.ndim == 2:
            t_arr = (t.numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(t_arr, mode="L").convert("RGB")
    elif isinstance(image_or_tensor, np.ndarray):
        arr = image_or_tensor
        if arr.dtype != np.uint8:
            arr = (
                (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                if arr.max() <= 1.0
                else np.clip(arr, 0, 255).astype(np.uint8)
            )
        pil_img = Image.fromarray(arr).convert("RGB")
    elif isinstance(image_or_tensor, Image.Image):
        pil_img = image_or_tensor.convert("RGB")
    elif isinstance(image_or_tensor, (bytes, str, Path)):
        import io
        if isinstance(image_or_tensor, bytes):
            pil_img = Image.open(io.BytesIO(image_or_tensor)).convert("RGB")
        else:
            pil_img = Image.open(str(image_or_tensor)).convert("RGB")
    else:
        raise ValueError(f"Unsupported input type for detection pipeline: {type(image_or_tensor)}")

    res = det_service.predict(pil_img, threshold=threshold, metadata=metadata)
    return {
        "detected": bool(res.get("is_cyclone", False)),
        "confidence": float(res.get("confidence", 0.0)),
        "predicted_label": res.get("predicted_label", "NO_CYCLONE"),
        "model_name": res.get("model_name", det_service.model_name),
        "model_version": res.get("model_version", det_service.model_version),
        "details": res,
    }

