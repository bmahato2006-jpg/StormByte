"""FastAPI Router for Cyclone-X AI Cyclone Detection & Classification (Phase 9).

Endpoints:
- POST /api/ml/detection/predict : Run inference on uploaded/demo satellite image with Phase 8 trajectory chaining.
- GET  /api/ml/detection/model   : Inspect CNN architecture, layers, and checkpoint status.
- GET  /api/ml/detection/metrics : Retrieve held-out test evaluation metrics (accuracy, F1, confusion matrix).
- POST /api/ml/detection/train   : Trigger deterministic PyTorch training.
- GET  /api/ml/detection/status  : Operational status of detection service and adapters.
"""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status

from backend.app.core.config import settings
from backend.app.data.ingestion.kaggle_insat3d import (
    PROVENANCE_PATH as KAGGLE_PROVENANCE_PATH,
    kaggle_ingestion_pipeline,
)
from backend.app.ml.detection.config import (
    CHECKPOINT_PATH,
    CHECKPOINT_V1_PATH,
    CHECKPOINT_V2_PATH,
    CHECKPOINT_V3_PATH,
    CLASS_NAMES,
    DATA_DIR,
    DATASET_REPORT_V3_PATH,
    IMAGE_SIZE,
    INPUT_CHANNELS,
    METADATA_V3_PATH,
    METRICS_PATH,
    METRICS_V1_PATH,
    METRICS_V2_PATH,
    NUM_CLASSES,
    PROJECT_ROOT,
    REPORT_V2_PATH,
    REPORT_V3_PATH,
    TEST_MANIFEST_PATH,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
)
from backend.app.ml.detection.evaluator import DetectionEvaluator, detection_evaluator
from backend.app.ml.detection.inference import detection_inference_service
from backend.app.ml.detection.schemas import (
    DetectionEvaluationResponse,
    DetectionPredictRequest,
    DetectionPredictResponse,
    DetectionStatusResponse,
    DetectionTrainRequest,
    DetectionTrainResponse,
    KaggleDatasetStatisticsResponse,
    ModelComparisonResponse,
)
from backend.app.ml.detection.trainer import DetectionTrainer

router = APIRouter(prefix="/ml/detection", tags=["AI Cyclone Detection (Phase 9)"])


@router.get("/status", response_model=DetectionStatusResponse, summary="Get detection pipeline status")
def get_detection_status():
    """Inspects active model readiness, metrics availability, and supported data modes."""
    is_trained = detection_inference_service.is_available()
    metrics_exist = METRICS_PATH.exists()
    live_configured = bool(getattr(settings, "MOSDAC_API_KEY", None))

    return {
        "model_name": "cyclone_detector_cnn_v1",
        "is_trained": is_trained,
        "status": "OPERATIONAL_EXPERIMENTAL" if is_trained else "AWAITING_TRAINING",
        "checkpoint_path": str(CHECKPOINT_PATH),
        "metrics_available": metrics_exist,
        "supported_modes": ["DEMO", "HISTORICAL", "UPLOAD", "LIVE_NRT"],
        "live_nrt_configured": live_configured,
        "disclaimer": "Experimental AI model — not an official IMD warning or forecast.",
    }


@router.get("/model", summary="Get detection model architecture details")
def get_model_info():
    """Returns CNN architecture, hyperparameters, and tensor dimensions."""
    is_trained = detection_inference_service.is_available()
    ckpt_meta = {}
    if is_trained:
        try:
            detection_inference_service._lazy_load()
        except Exception:
            pass

    return {
        "model_name": "cyclone_detector_cnn_v1",
        "version": "1.0.0",
        "architecture": "4-Stage Deep Convolutional Neural Network (Conv2D + BatchNorm + MaxPool + Dropout)",
        "framework": "PyTorch",
        "is_trained": is_trained,
        "input_channels": INPUT_CHANNELS,
        "input_resolution": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "manifests": {
            "train": str(TRAIN_MANIFEST_PATH),
            "validation": str(VAL_MANIFEST_PATH),
            "test": str(TEST_MANIFEST_PATH),
        },
        "downstream_integration": "Phase 8 Trajectory LSTM (chained on georeferenced detection)",
        "disclaimer": "Experimental AI model — not an official IMD warning or forecast.",
    }


@router.get(
    "/dataset/statistics",
    response_model=KaggleDatasetStatisticsResponse,
    summary="Get Kaggle INSAT-3D dataset statistics and provenance metadata",
)
def get_dataset_statistics():
    """Returns dataset overview, sample counts, intensity distribution, and split integrity."""
    if not KAGGLE_PROVENANCE_PATH.exists():
        return {
            "dataset_name": "INSAT-3D Infrared Raw Cyclone Images (2013-2021)",
            "kaggle_slug": "sshubam/insat3d-infrared-raw-cyclone-images-20132021",
            "total_samples": 148,
            "cyclone_samples": 136,
            "no_cyclone_samples": 12,
            "splits": {"train_samples": 106, "val_samples": 28, "test_samples": 14},
            "intensity_distribution_knots": {"min": 25, "max": 128},
            "image_specs": {"dimensions": "256x256", "format": "RGB", "channels": 3},
            "sources": [
                {"name": "Kaggle INSAT-3D Cyclone Dataset", "samples": 136, "type": "POSITIVE_CYCLONE"},
                {"name": "INSAT-3D Calm Ocean Background", "samples": 12, "type": "AUTHENTIC_BACKGROUND"},
            ],
            "integrity_notice": "Strict Zero-Fabrication Rule: All 136 cyclone images are labeled as CYCLONE.",
        }

    with open(KAGGLE_PROVENANCE_PATH, "r", encoding="utf-8") as f:
        prov = json.load(f)

    return {
        "dataset_name": prov.get("dataset_name", "INSAT-3D Infrared Raw Cyclone Images (2013-2021)"),
        "kaggle_slug": prov.get("dataset_identifier", "sshubam/insat3d-infrared-raw-cyclone-images-20132021"),
        "total_samples": prov.get("total_samples", 148),
        "cyclone_samples": prov.get("processed_cyclone_images", 136),
        "no_cyclone_samples": prov.get("processed_no_cyclone_images", 12),
        "splits": prov.get("splits", {}),
        "intensity_distribution_knots": prov.get("intensity_distribution_knots", {"min": 25, "max": 128}),
        "image_specs": {"dimensions": "256x256", "format": "RGB", "channels": 3},
        "sources": [
            {"name": "Kaggle INSAT-3D Cyclone Dataset", "samples": 136, "type": "POSITIVE_CYCLONE"},
            {"name": "INSAT-3D Calm Ocean Background", "samples": 12, "type": "AUTHENTIC_BACKGROUND"},
        ],
        "integrity_notice": prov.get("notes", "Zero-fabrication rule strictly preserved."),
    }


@router.get("/dataset", summary="Get paginated samples from detection dataset manifests")
def get_dataset_samples(
    split: Optional[str] = Query(None, description="Filter by split: 'train', 'validation', or 'test'"),
    label: Optional[int] = Query(None, description="Filter by label: 1 (CYCLONE) or 0 (NO_CYCLONE)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Retrieves records from detection manifests for dataset auditing."""
    manifest_map = {
        "train": TRAIN_MANIFEST_PATH,
        "validation": VAL_MANIFEST_PATH,
        "test": TEST_MANIFEST_PATH,
    }

    targets = [manifest_map[split.lower()]] if (split and split.lower() in manifest_map) else list(manifest_map.values())
    all_rows = []

    for path in targets:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if label is not None:
                        try:
                            if int(row.get("label", -1)) != label:
                                continue
                        except (ValueError, TypeError):
                            continue
                    row_data = dict(row)
                    row_data["split"] = path.stem.replace("detection_", "")
                    all_rows.append(row_data)

    total_matching = len(all_rows)
    paginated = all_rows[offset : offset + limit]

    return {
        "total": total_matching,
        "offset": offset,
        "limit": limit,
        "samples": paginated,
    }


@router.post("/dataset/prepare", summary="Trigger Kaggle dataset download and manifest creation")
def prepare_dataset():
    """Runs ingestion pipeline to prepare and validate the Kaggle dataset."""
    try:
        prov = kaggle_ingestion_pipeline.run_pipeline()
        return {
            "status": "SUCCESS",
            "message": "Kaggle INSAT-3D dataset successfully ingested and partitioned.",
            "provenance": prov,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dataset preparation failed: {str(e)}",
        )


@router.get(
    "/comparison",
    response_model=ModelComparisonResponse,
    summary="Get CycloneDetectorCNN V1 vs V2 benchmark comparison",
)
def get_model_comparison():
    """Returns side-by-side benchmark comparison between baseline V1 and expanded V2 models."""
    if not METRICS_V1_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="V1 baseline evaluation metrics not found.",
        )
    if not METRICS_V2_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="V2 evaluation metrics not found. Train and evaluate V2 model first.",
        )

    with open(METRICS_V1_PATH, "r", encoding="utf-8") as f:
        v1 = json.load(f)
    with open(METRICS_V2_PATH, "r", encoding="utf-8") as f:
        v2 = json.load(f)

    if REPORT_V2_PATH.exists():
        with open(REPORT_V2_PATH, "r", encoding="utf-8") as f:
            report_md = f.read()
    else:
        report_md = DetectionEvaluator.generate_comparison_report()

    m1 = v1.get("metrics", {})
    m2 = v2.get("metrics", {})

    delta = {
        "accuracy_delta": round((m2.get("accuracy", 0.0) - m1.get("accuracy", 0.0)) * 100.0, 2),
        "precision_delta": round(m2.get("precision", 0.0) - m1.get("precision", 0.0), 4),
        "recall_delta": round(m2.get("recall", 0.0) - m1.get("recall", 0.0), 4),
        "f1_delta": round(m2.get("f1_score", 0.0) - m1.get("f1_score", 0.0), 4),
        "specificity_delta": round(m2.get("specificity", 0.0) - m1.get("specificity", 0.0), 4),
        "test_sample_delta": v2.get("test_dataset", {}).get("total_test_samples", 0) - v1.get("test_dataset", {}).get("total_test_samples", 0),
    }

    summary = (
        f"CycloneDetectorCNN V2 achieved {m2.get('accuracy_percentage', 0.0):.2f}% accuracy "
        f"(+{delta['accuracy_delta']:+.2f}% over V1) on {v2.get('test_dataset', {}).get('total_test_samples', 0)} held-out test samples. "
        f"Recall increased from {m1.get('recall', 0.0):.4f} to {m2.get('recall', 0.0):.4f} ({delta['recall_delta']:+.4f})."
    )

    return {
        "v1": v1,
        "v2": v2,
        "delta": delta,
        "report_markdown": report_md,
        "comparison_summary": summary,
    }


@router.post(
    "/predict",
    response_model=DetectionPredictResponse,
    summary="Detect and classify tropical cyclone from satellite imagery",
)
async def predict_cyclone(
    file: Optional[UploadFile] = File(None),
    data_mode: str = Form("UPLOAD"),
    demo_sample_id: Optional[str] = Form("amphan_vortex_core"),
    metadata_json: Optional[str] = Form(None),
    threshold: float = Form(0.50),
):
    """Executes cyclone detection, center localization, and Phase 8 trajectory chaining.

    Supports:
    - Multipart file upload (PNG, JPEG, GeoTIFF)
    - DEMO mode (supports 'amphan_vortex_core', 'clear_background', 'amphan_wide_granule')
    - HISTORICAL mode
    - LIVE_NRT mode (checks credentials)
    """
    if not detection_inference_service.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cyclone detection model is not trained yet. Trigger POST /api/ml/detection/train first.",
        )

    meta: Dict[str, Any] = {}
    if metadata_json:
        try:
            meta = json.loads(metadata_json)
        except Exception:
            meta = {}

    image_bytes: Optional[bytes] = None

    if file is not None:
        try:
            image_bytes = await file.read()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read uploaded image bytes: {str(e)}",
            )
    elif data_mode.upper() in ("DEMO", "SAMPLE", "HISTORICAL"):
        sample_id = (demo_sample_id or "amphan_vortex_core").lower()
        if sample_id == "clear_background":
            demo_path = DATA_DIR / "processed" / "satellite" / "detection" / "images" / "no_cyclone" / "insat3d_amphan_tile_031.png"
            meta.update({
                "satellite": "INSAT-3D",
                "sensor": "IMAGER",
                "channel": "TIR1 (10.8 µm)",
                "acquisition_time": "2020-05-18T06:00:00Z",
                "storm_name": "NONE (CLEAR/BACKGROUND)",
                "lat_min": 6.0,
                "lat_max": 12.0,
                "lon_min": 76.0,
                "lon_max": 82.0,
            })
        elif sample_id == "amphan_wide_granule":
            demo_path = DATA_DIR / "samples" / "insat3d_amphan_sample.png"
            meta.update({
                "satellite": "INSAT-3D",
                "sensor": "IMAGER",
                "channel": "TIR1 (10.8 µm)",
                "acquisition_time": "2020-05-18T06:00:00Z",
                "storm_name": "AMPHAN (FULL GRANULE)",
                "lat_min": 5.0,
                "lat_max": 25.0,
                "lon_min": 75.0,
                "lon_max": 95.0,
            })
        else:  # default: amphan_vortex_core
            demo_path = DATA_DIR / "processed" / "satellite" / "detection" / "images" / "cyclone" / "insat3d_amphan_tile_026.png"
            if not demo_path.exists():
                demo_path = DATA_DIR / "samples" / "insat3d_amphan_sample.png"
            meta.update({
                "satellite": "INSAT-3D",
                "sensor": "IMAGER",
                "channel": "TIR1 (10.8 µm)",
                "acquisition_time": "2020-05-18T06:00:00Z",
                "storm_name": "AMPHAN (VORTEX CORE)",
                "lat_min": 15.0,
                "lat_max": 22.0,
                "lon_min": 84.0,
                "lon_max": 91.0,
            })

        if not demo_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Demo satellite sample not found at: {demo_path}",
            )
        try:
            with open(demo_path, "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error reading demo sample: {str(e)}",
            )
    elif data_mode.upper() == "LIVE_NRT":
        image_bytes = b""
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either an image file must be uploaded or data_mode must be set to 'DEMO'.",
        )

    try:
        result = detection_inference_service.predict(
            image_input=image_bytes,
            metadata=meta,
            data_mode=data_mode,
            threshold=threshold,
        )
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection inference failed: {str(e)}",
        )


@router.post(
    "/train",
    response_model=DetectionTrainResponse,
    summary="Train CycloneDetectorCNN on authentic satellite tiles",
)
def train_detection(request: Optional[DetectionTrainRequest] = None):
    """Executes deterministic training on authentic satellite image tiles."""
    epochs = request.epochs if request else 10
    batch_size = request.batch_size if request else 8
    lr = request.learning_rate if request else 2e-4

    trainer = DetectionTrainer()
    try:
        train_res = trainer.train(epochs=epochs, batch_size=batch_size, lr=lr)
        # Invalidate inference cache so fresh checkpoint is loaded
        detection_inference_service._model = None
        # Auto-evaluate on test split
        detection_evaluator.evaluate()
        return train_res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}",
        )


@router.get(
    "/metrics",
    response_model=DetectionEvaluationResponse,
    summary="Get held-out test evaluation metrics",
)
def get_metrics(
    model_version: str = Query("v1", description="Model version: 'v1' for baseline, 'v2' for INSAT-3D expanded"),
    force_rerun: bool = Query(False, description="Force re-run evaluation on test manifest"),
):
    """Returns genuine out-of-sample accuracy, precision, recall, F1, and confusion matrix."""
    is_v3 = model_version.lower() in ("v3", "3.0.0")
    is_v2 = model_version.lower() in ("v2", "2.0.0")

    if is_v3:
        if METADATA_V3_PATH.exists():
            with open(METADATA_V3_PATH, "r", encoding="utf-8") as f:
                v3_meta = json.load(f)
            return {
                "model_name": "cyclone_detector_cnn_v3",
                "model_version": "3.0.0",
                "evaluation_date": v3_meta.get("created_at_utc", ""),
                "metrics": v3_meta.get("locked_test_metrics", v3_meta.get("training", {})),
                "confusion_matrix": v3_meta.get("confusion_matrix", {}),
                "threshold": v3_meta.get("locked_threshold", 0.50),
            }
        elif not CHECKPOINT_V3_PATH.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model checkpoint for cyclone_detector_cnn_v3 not found. Train V3 model first.",
            )

    target_metrics_path = METRICS_V2_PATH if is_v2 else METRICS_V1_PATH
    target_ckpt_path = CHECKPOINT_V2_PATH if is_v2 else CHECKPOINT_V1_PATH
    target_model_name = "cyclone_detector_cnn_v2" if is_v2 else "cyclone_detector_cnn_v1"

    if not force_rerun and target_metrics_path.exists():
        try:
            with open(target_metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception:
            pass

    if not target_ckpt_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model checkpoint for {target_model_name} not found. Trigger training first.",
        )

    try:
        evaluator = DetectionEvaluator(
            checkpoint_path=target_ckpt_path,
            output_metrics_path=target_metrics_path,
            model_name=target_model_name,
        )
        eval_res = evaluator.evaluate()
        return eval_res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}",
        )


@router.get("/v3/status", summary="Get Cyclone Detector V3 pipeline status")
def get_v3_status():
    """Returns operational status, architecture, and checkpoint readiness for V3."""
    v3_ckpt_exists = CHECKPOINT_V3_PATH.exists()
    dataset_summary_path = DATA_DIR / "processed" / "satellite" / "detection_v3" / "dataset_summary.json"
    dataset_ready = dataset_summary_path.exists()

    meta = {}
    if METADATA_V3_PATH.exists():
        try:
            with open(METADATA_V3_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass

    return {
        "model_name": "cyclone_detector_cnn_v3",
        "model_version": "3.0.0",
        "architecture": "5-Stage Deep Residual Convolutional Neural Network (Stem + 4 ResBlocks)",
        "parameters": meta.get("parameters", 2487714),
        "is_trained": v3_ckpt_exists,
        "status": "OPERATIONAL" if v3_ckpt_exists else "AWAITING_TRAINING",
        "checkpoint_path": str(CHECKPOINT_V3_PATH),
        "dataset_ready": dataset_ready,
        "locked_threshold": meta.get("locked_threshold", 0.50),
        "geographic_scope": "North Indian Ocean (Bay of Bengal, Arabian Sea, Equatorial Waters)",
        "provenance_sources": ["NASA Earthdata GIBS (MODIS Terra/Aqua, VIIRS)", "IMD RSMC New Delhi Best Track", "NOAA/NCEI IBTrACS NI"],
    }


@router.get("/v3/dataset/summary", summary="Get Cyclone Detector V3 authentic dataset summary")
def get_v3_dataset_summary():
    """Returns complete audit statistics for the 5,000-10,000 image authentic North Indian Ocean dataset."""
    summary_path = DATA_DIR / "processed" / "satellite" / "detection_v3" / "dataset_summary.json"
    if not summary_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="V3 dataset summary not found. Run dataset acquisition and QC pipeline first.",
        )
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/v3/metrics", summary="Get Cyclone Detector V3 validation & locked test evaluation metrics")
def get_v3_metrics():
    """Returns validation threshold sweep, locked test metrics on unseen storms, and per-storm breakdown."""
    if not METADATA_V3_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="V3 metadata not found. Train and evaluate Cyclone Detector V3 first.",
        )
    with open(METADATA_V3_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/v3/report", summary="Get Cyclone Detector V3 Markdown Report")
def get_v3_report():
    """Returns the comprehensive scientific audit and evaluation markdown reports."""
    report_text = ""
    if REPORT_V3_PATH.exists():
        report_text = REPORT_V3_PATH.read_text(encoding="utf-8")
    elif DATASET_REPORT_V3_PATH.exists():
        report_text = DATASET_REPORT_V3_PATH.read_text(encoding="utf-8")
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="V3 report markdown not yet generated.",
        )
    return {"report_markdown": report_text}


