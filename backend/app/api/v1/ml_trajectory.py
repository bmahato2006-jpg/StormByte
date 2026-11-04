"""FastAPI Router for Cyclone-X ML Trajectory Forecasting.

Endpoints:
- POST /api/ml/trajectory/predict : Run inference using TrajectoryTransformer or TrajectoryLSTM.
- POST /api/ml/trajectory/train   : Execute reproducible PyTorch training for transformer or lstm.
- GET  /api/ml/trajectory/evaluate: Run/return comparative evaluation on held-out test storms.
- GET  /api/ml/trajectory/models  : Inspect all available trajectory models (Transformer, LSTM).
- GET  /api/ml/trajectory/model   : Inspect active primary model architecture and specifications.
- GET  /api/ml/trajectory/metrics : Return master comparison table and error statistics.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from backend.app.ml.trajectory.config import (
    CHECKPOINT_PATH,
    EVALUATION_METRICS_PATH,
    FEATURE_NAMES,
    HORIZONS,
    INPUT_DIM,
    NUM_LAYERS,
    OUTPUT_DIM,
    SCALER_PATH,
    SEQ_LENGTH,
    TEST_MANIFEST_PATH,
    TRAIN_MANIFEST_PATH,
    TRANSFORMER_CHECKPOINT_PATH,
    TRANSFORMER_EVALUATION_METRICS_PATH,
    TRANSFORMER_METADATA_PATH,
    VAL_MANIFEST_PATH,
)
from backend.app.ml.trajectory.evaluator import TrajectoryEvaluator, evaluator
from backend.app.ml.trajectory.inference import trajectory_inference_service
from backend.app.ml.trajectory.train_transformer import TransformerTrainer
from backend.app.ml.trajectory.trainer import TrajectoryTrainer
from backend.app.schemas.ml_trajectory import (
    MLTrajectoryEvaluationResponse,
    MLTrajectoryPredictRequest,
    MLTrajectoryPredictResponse,
    MLTrajectoryTrainRequest,
    MLTrajectoryTrainResponse,
)

router = APIRouter(prefix="/ml/trajectory", tags=["ML Trajectory Forecasting"])


@router.get("/models", summary="List all available ML trajectory models and specifications")
def get_all_models():
    """Returns metadata and operational status for all available models (Transformer and LSTM)."""
    models = []

    # 1. TrajectoryTransformer (Phase 13)
    transformer_available = trajectory_inference_service.is_available("transformer")
    trans_meta: Dict[str, Any] = {}
    if TRANSFORMER_METADATA_PATH.exists():
        try:
            with open(TRANSFORMER_METADATA_PATH, "r", encoding="utf-8") as f:
                trans_meta = json.load(f)
        except Exception:
            pass

    models.append({
        "model_id": "trajectory_transformer_v1",
        "name": "TrajectoryTransformer V1",
        "architecture": "Spatio-Temporal Transformer Encoder (Pre-LN)",
        "framework": "PyTorch",
        "is_trained": transformer_available,
        "is_primary": True,
        "status": "OPERATIONAL_EXPERIMENTAL" if transformer_available else "AWAITING_TRAINING",
        "hyperparameters": trans_meta.get("hyperparameters", {
            "input_dim": INPUT_DIM,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "output_dim": OUTPUT_DIM,
            "seq_len": SEQ_LENGTH,
        }),
        "checkpoint_path": str(TRANSFORMER_CHECKPOINT_PATH),
        "metadata_path": str(TRANSFORMER_METADATA_PATH),
    })

    # 2. TrajectoryLSTM (Phase 8 Baseline)
    lstm_available = trajectory_inference_service.is_available("lstm")
    models.append({
        "model_id": "trajectory_lstm_v1",
        "name": "TrajectoryLSTM V1",
        "architecture": "2-Layer Recurrent Neural Network (LSTM)",
        "framework": "PyTorch",
        "is_trained": lstm_available,
        "is_primary": False,
        "status": "OPERATIONAL_EXPERIMENTAL" if lstm_available else "AWAITING_TRAINING",
        "hyperparameters": {
            "input_dim": INPUT_DIM,
            "hidden_dim": 64,
            "num_layers": NUM_LAYERS,
            "output_dim": OUTPUT_DIM,
        },
        "checkpoint_path": str(CHECKPOINT_PATH),
    })

    return {
        "models": models,
        "active_primary_model": "trajectory_transformer_v1" if transformer_available else "trajectory_lstm_v1",
        "input_features": FEATURE_NAMES,
        "forecast_horizons_hours": HORIZONS,
        "scaler_path": str(SCALER_PATH),
    }


@router.get("/model", summary="Get ML trajectory model status and specifications")
def get_model_status(model_type: Optional[str] = Query("lstm", description="Model architecture: 'lstm' (default) or 'transformer'")):
    """Inspects the active status, checkpoint availability, and architecture parameters."""
    trans_avail = trajectory_inference_service.is_available("transformer")
    lstm_avail = trajectory_inference_service.is_available("lstm")

    if model_type == "transformer" and trans_avail:
        try:
            trajectory_inference_service._lazy_load_transformer()
            ckpt_meta = trajectory_inference_service._checkpoint_metadata.get("transformer", {})
        except Exception as e:
            ckpt_meta = {"error": str(e)}

        return {
            "model_name": "trajectory_transformer_v1",
            "version": "1.0.0",
            "architecture": "Spatio-Temporal Transformer Encoder (Pre-LN)",
            "framework": "PyTorch",
            "is_trained": True,
            "status": "OPERATIONAL_EXPERIMENTAL",
            "is_experimental": True,
            "input_features": FEATURE_NAMES,
            "input_dim": INPUT_DIM,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "output_dim": OUTPUT_DIM,
            "forecast_horizons_hours": HORIZONS,
            "checkpoint_path": str(TRANSFORMER_CHECKPOINT_PATH),
            "scaler_path": str(SCALER_PATH),
            "checkpoint_metadata": ckpt_meta,
            "manifests": {
                "train": str(TRAIN_MANIFEST_PATH),
                "validation": str(VAL_MANIFEST_PATH),
                "test": str(TEST_MANIFEST_PATH),
            },
            "disclaimer": (
                "EXPERIMENTAL RESEARCH MODEL: TrajectoryTransformer is a research tool for scientific benchmarking "
                "and is NOT an official IMD operational forecasting system."
            ),
        }
    else:
        try:
            if lstm_avail:
                trajectory_inference_service._lazy_load_lstm()
                ckpt_meta = trajectory_inference_service._checkpoint_metadata.get("lstm", {})
            else:
                ckpt_meta = {}
        except Exception as e:
            ckpt_meta = {"error": str(e)}

        return {
            "model_name": "trajectory_lstm_v1",
            "version": "1.0.0",
            "architecture": "2-Layer Recurrent Neural Network (LSTM)",
            "framework": "PyTorch",
            "is_trained": lstm_avail,
            "status": "OPERATIONAL_EXPERIMENTAL" if lstm_avail else "AWAITING_TRAINING",
            "is_experimental": True,
            "input_features": FEATURE_NAMES,
            "input_dim": INPUT_DIM,
            "hidden_dim": 64,
            "num_layers": NUM_LAYERS,
            "output_dim": OUTPUT_DIM,
            "forecast_horizons_hours": HORIZONS,
            "checkpoint_path": str(CHECKPOINT_PATH),
            "scaler_path": str(SCALER_PATH),
            "checkpoint_metadata": ckpt_meta,
            "manifests": {
                "train": str(TRAIN_MANIFEST_PATH),
                "validation": str(VAL_MANIFEST_PATH),
                "test": str(TEST_MANIFEST_PATH),
            },
            "disclaimer": (
                "EXPERIMENTAL RESEARCH MODEL: TrajectoryLSTM is a research tool for scientific benchmarking "
                "and is NOT an official IMD operational forecasting system."
            ),
        }


@router.post(
    "/predict",
    response_model=MLTrajectoryPredictResponse,
    summary="Predict future trajectory coordinates using trained Transformer or LSTM",
)
def predict_trajectory(request: MLTrajectoryPredictRequest):
    """Executes multi-horizon (+6h, +12h, +24h) trajectory displacement forecasting."""
    requested_model = request.model_type or "transformer"
    if not trajectory_inference_service.is_available(requested_model) and not trajectory_inference_service.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Trajectory model '{requested_model}' is not trained yet. Checkpoint not found. "
                f"Trigger training via POST /api/ml/trajectory/train?model_type={requested_model} first."
            ),
        )

    try:
        result = trajectory_inference_service.predict(
            track_history=request.track_history,
            forecast_horizons_hours=request.forecast_horizons,
            model_type=requested_model,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference failed: {str(e)}",
        )


@router.post(
    "/train",
    response_model=MLTrajectoryTrainResponse,
    summary="Train TrajectoryTransformer or TrajectoryLSTM on historical NIO dataset",
)
def train_model(request: Optional[MLTrajectoryTrainRequest] = None):
    """Executes deterministic PyTorch training on data/processed/manifests/trajectory_train.csv."""
    model_type = (request.model_type if request and request.model_type else "transformer").lower()
    epochs = request.epochs if request else 12
    batch_size = request.batch_size if request else 64
    lr = request.learning_rate if request else 1e-3

    try:
        if model_type == "lstm":
            trainer = TrajectoryTrainer()
            train_res = trainer.train(epochs=epochs, batch_size=batch_size, lr=lr)
            trajectory_inference_service._lstm_model = None
            trajectory_inference_service._scaler = None
        else:
            trainer = TransformerTrainer()
            train_res = trainer.train(epochs=epochs, batch_size=batch_size, lr=lr)
            trajectory_inference_service._transformer_model = None
            trajectory_inference_service._scaler = None

        return train_res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}",
        )


@router.get(
    "/evaluate",
    response_model=MLTrajectoryEvaluationResponse,
    summary="Evaluate TrajectoryTransformer and LSTM vs Physics on held-out test storms",
)
def evaluate_model(force_rerun: bool = Query(False, description="Re-run evaluation even if cached report exists")):
    """Benchmarks TrajectoryTransformer, TrajectoryLSTM, and physics_guided_advection on held-out test storms."""
    if not force_rerun and EVALUATION_METRICS_PATH.exists():
        try:
            with open(EVALUATION_METRICS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception:
            pass

    if not CHECKPOINT_PATH.exists() and not TRANSFORMER_CHECKPOINT_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cannot evaluate models: no checkpoint found. Train the model first.",
        )

    try:
        eval_res = evaluator.evaluate()
        return eval_res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}",
        )


@router.get("/metrics", summary="Get pre-computed comparative evaluation metrics")
def get_metrics():
    """Returns the comparative metrics payload comparing ML models and Physics."""
    target_path = EVALUATION_METRICS_PATH
    if not target_path.exists() and TRANSFORMER_EVALUATION_METRICS_PATH.exists():
        target_path = TRANSFORMER_EVALUATION_METRICS_PATH

    if not target_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation metrics not found. Run GET /api/ml/trajectory/evaluate to generate metrics.",
        )

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "evaluation_id": data.get("evaluation_id"),
        "evaluated_at": data.get("evaluated_at"),
        "test_dataset": data.get("test_dataset"),
        "models_evaluated": data.get("models_evaluated"),
        "comparison_table": data.get("comparison_table"),
        "ml_model_metrics": data.get("ml_model_metrics"),
        "transformer_model_metrics": data.get("transformer_model_metrics"),
        "lstm_model_metrics": data.get("lstm_model_metrics"),
        "physics_model_metrics": data.get("physics_model_metrics"),
        "per_storm_breakdown": data.get("per_storm_breakdown"),
        "disclaimer": data.get("disclaimer"),
    }
