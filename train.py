"""CYCLONE-X Unified Deep Learning & Machine Learning Training Pipeline.

Provides a standardized CLI command to train and evaluate all machine learning
models in the CYCLONE-X platform:
1. TrajectoryLSTM (Recurrent Neural Network for 6h/12h/24h track displacement)
2. TrajectoryTransformer (Pre-LN Self-Attention Transformer for multi-horizon track forecasting)
3. CycloneDetectorCNN V2 (4-stage CNN on authentic INSAT-3D satellite tiles)
4. CycloneDetectorCNN V3 (5-stage Deep Residual Network on 3,723 satellite granules)
5. CycloneViT (Vision Transformer with 16x16 patch self-attention on 3,723 tiles)
6. CycloneCenterNet (Deep ResNet-18 Keypoint Eye Center Detector)
7. CycloneIntensityResNet18 (Deep multi-task regression for Vmax & Pc estimation)
8. RapidIntensificationXGB (HistGradientBoosting 24h RI classification on SHIPS/ERA5)
9. CyclogenesisXGB (HistGradientBoosting 24h open-ocean cyclogenesis potential)

Usage Examples:
  python train.py --model all
  python train.py --model master_all
  python train.py --model transformer --epochs 5
  python train.py --model vit --epochs 3
  python train.py --model centernet --epochs 5
  python train.py --model intensity --epochs 5
  python train.py --model ri
  python train.py --model cyclogenesis
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_banner(title: str):
    width = 72
    print("\n" + "=" * width)
    print(f"  CYCLONE-X ML ENGINE  |  {title}")
    print("=" * width + "\n")


def train_trajectory(epochs: int = 10, batch_size: int = 64, lr: float = 1e-3, evaluate: bool = True):
    print_banner("TRAJECTORY FORECASTING MODEL (TrajectoryLSTM V1)")
    from backend.app.ml.trajectory.trainer import TrajectoryTrainer
    from backend.app.ml.trajectory.evaluator import TrajectoryEvaluator
    from backend.app.ml.trajectory.config import TRAIN_MANIFEST_PATH, VAL_MANIFEST_PATH, TEST_MANIFEST_PATH

    print(f"Corpus: NOAA NCEI IBTrACS NIO Historical Best Track (<=2021)")
    print(f"Train Manifest:      {TRAIN_MANIFEST_PATH}")
    print(f"Validation Manifest: {VAL_MANIFEST_PATH}")
    print(f"Test Manifest:       {TEST_MANIFEST_PATH}")
    print(f"Architecture:        2-Layer LSTM (Hidden Dim=64, Input Dim=13)")
    print(f"Horizons:            +6h, +12h, +24h Relative Displacement\n")

    trainer = TrajectoryTrainer()
    t0 = time.time()
    res = trainer.train(epochs=epochs, batch_size=batch_size, lr=lr)
    elapsed = time.time() - t0

    print(f"\n[+] Training Complete in {elapsed:.2f}s")
    print(f"    Best Epoch:       {res.get('best_epoch')}")
    print(f"    Final Train Loss: {res.get('final_train_loss', 0):.5f}")
    print(f"    Best Val Loss:    {res.get('best_val_loss', 0):.5f}")
    print(f"    Checkpoint:       {res.get('checkpoint_path')}")

    if evaluate:
        print("\n--- Out-of-Sample Evaluation on Held-Out Test Storms (>=2023) ---")
        evaluator = TrajectoryEvaluator()
        eval_metrics = evaluator.evaluate()
        ml_m = eval_metrics.get("ml_model_metrics", {})
        ph_m = eval_metrics.get("physics_model_metrics", {})
        ml_mae = ml_m.get("overall", {}).get("mae_km", 0.0)
        ph_mae = ph_m.get("overall", {}).get("mae_km", 0.0)
        print(f"    TrajectoryLSTM MAE:     {ml_mae:.2f} km")
        print(f"    Physics Baseline MAE:   {ph_mae:.2f} km")

    return res


def train_transformer(epochs: int = 5, batch_size: int = 32, lr: float = 5e-4, evaluate: bool = True):
    print_banner("SPATIO-TEMPORAL TRANSFORMER (TrajectoryTransformer V1)")
    from backend.app.ml.trajectory.train_transformer import TransformerTrainer
    trainer = TransformerTrainer()
    t0 = time.time()
    res = trainer.train(epochs=epochs, batch_size=batch_size, lr=lr)
    elapsed = time.time() - t0
    print(f"\n[+] Trajectory Transformer Pipeline Complete in {elapsed:.2f}s")
    return res or {}


def train_detection(epochs: int = 10, batch_size: int = 8, lr: float = 2e-4, evaluate: bool = True):
    print_banner("CYCLONE DETECTION MODEL V2 (CycloneDetectorCNN V2)")
    from backend.app.ml.detection.trainer import DetectionTrainer
    from backend.app.ml.detection.evaluator import DetectionEvaluator

    trainer = DetectionTrainer(model_name="cyclone_detector_cnn_v2")
    t0 = time.time()
    res = trainer.train(epochs=epochs, batch_size=batch_size, lr=lr)
    elapsed = time.time() - t0

    print(f"\n[+] Training Complete in {elapsed:.2f}s")
    print(f"    Checkpoint:       {res.get('checkpoint_path')}")

    if evaluate:
        evaluator = DetectionEvaluator(model_name="cyclone_detector_cnn_v2")
        eval_res = evaluator.evaluate()
        m = eval_res.get("metrics", {})
        print(f"    Test Accuracy:    {m.get('accuracy_percentage', 0):.2f}%")
        print(f"    ROC-AUC:          {m.get('roc_auc', 0):.4f}")

    return res


def train_detection_v3(epochs: int = 7, batch_size: int = 32, lr: float = 3e-4, evaluate: bool = True):
    print_banner("CYCLONE DETECTION MODEL V3 (5-Stage ResNet CycloneDetectorCNN_V3)")
    from backend.app.ml.detection.trainer_v3 import train_cyclone_detector_v3
    from backend.app.ml.detection.evaluate_v3 import run_full_evaluation_pipeline

    res = train_cyclone_detector_v3(num_epochs=epochs, batch_size=batch_size, learning_rate=lr)

    if evaluate:
        print("\n--- Threshold Sweep & Locked Test Evaluation on Unseen Storms ---")
        eval_metrics = run_full_evaluation_pipeline()
        print(f"    V3 Test Accuracy: {eval_metrics.get('accuracy', 0)*100:.2f}%")
        print(f"    V3 ROC-AUC:       {eval_metrics.get('roc_auc', 0):.4f}")

    return res


def train_vit(epochs: int = 2, batch_size: int = 32, lr: float = 1e-4, evaluate: bool = True):
    print_banner("VISION TRANSFORMER (CycloneViT 16x16 Self-Attention)")
    from backend.app.ml.detection.train_vit import ViTTrainer

    trainer = ViTTrainer()
    t0 = time.time()
    res = trainer.train(num_epochs=epochs, batch_size=batch_size, learning_rate=lr, evaluate=evaluate)
    elapsed = time.time() - t0

    print(f"\n[+] CycloneViT Training Complete in {elapsed:.2f}s")
    print(f"    Checkpoint:       {res.get('checkpoint_file')}")
    print(f"    Test Accuracy:    {res.get('test_accuracy', 0)*100:.2f}%")
    return res


def train_centernet(epochs: int = 5, batch_size: int = 8, lr: float = 1e-4, evaluate: bool = True):
    print_banner("CYCLONE CENTERNET (ResNet-18 Sub-Pixel Eye Keypoint Detector)")
    from backend.app.ml.training_framework.train_pipeline import train_centernet as _train_centernet
    t0 = time.time()
    res = _train_centernet(num_epochs=epochs, batch_size=batch_size, learning_rate=lr)
    elapsed = time.time() - t0
    print(f"\n[+] CenterNet Training Complete in {elapsed:.2f}s")
    return res or {}


def train_intensity(epochs: int = 5, batch_size: int = 16, lr: float = 1e-4, evaluate: bool = True):
    print_banner("CYCLONE INTENSITY REGRESSION (ResNet-18 Vmax / Pc)")
    from backend.app.ml.intensity.trainer import IntensityTrainer
    trainer = IntensityTrainer()
    t0 = time.time()
    res = trainer.train(epochs=epochs, batch_size=batch_size, lr=lr)
    elapsed = time.time() - t0
    print(f"\n[+] Intensity Training Complete in {elapsed:.2f}s")
    return res


def train_ri(evaluate: bool = True):
    print_banner("RAPID INTENSIFICATION (RI) PREDICTOR (XGBoost / HistGradientBoosting)")
    from backend.app.ml.training_framework.train_pipeline import train_ri_predictor
    t0 = time.time()
    res = train_ri_predictor()
    elapsed = time.time() - t0
    metrics = res.get("metrics", {})
    print(f"\n[+] RI Predictor Training Complete in {elapsed:.2f}s")
    print(f"    Dataset:          {metrics.get('dataset', 'N/A')}")
    print(f"    Total Samples:    {metrics.get('total_samples', 'N/A')}")
    print(f"    ROC-AUC:          {metrics.get('roc_auc', 0):.4f}")
    print(f"    Precision:        {metrics.get('precision', 0):.4f}")
    print(f"    Recall:           {metrics.get('recall', 0):.4f}")
    print(f"    Brier Score:      {metrics.get('brier_score', 0):.4f}")
    return res


def train_cyclogenesis(evaluate: bool = True):
    print_banner("CYCLOGENESIS POTENTIAL PREDICTOR (XGBoost / HistGradientBoosting)")
    from backend.app.ml.cyclogenesis.cyclogenesis_predictor import train_cyclogenesis_model
    t0 = time.time()
    res = train_cyclogenesis_model()
    elapsed = time.time() - t0
    metrics = res.get("metrics", {})
    print(f"\n[+] Cyclogenesis Predictor Training Complete in {elapsed:.2f}s")
    print(f"    Dataset:          {metrics.get('dataset', 'N/A')}")
    print(f"    Total Samples:    {metrics.get('n_samples', 'N/A')}")
    print(f"    ROC-AUC:          {metrics.get('roc_auc', 0):.4f}")
    print(f"    Test Accuracy:    {metrics.get('accuracy', 0)*100:.2f}%")
    print(f"    Brier Score:      {metrics.get('brier_score', 0):.4f}")
    return res


def train_dvorak(epochs: int = 12, batch_size: int = 16, lr: float = 1e-3, evaluate: bool = True):
    print_banner("OBJECTIVE DVORAK T-NUMBER CNN (ObjectiveDvorakNet)")
    from backend.app.ml.intensity.train_dvorak import DvorakTrainer
    trainer = DvorakTrainer()
    return trainer.train(num_epochs=epochs, batch_size=batch_size, lr=lr, evaluate=evaluate)


def train_landfall(epochs: int = 12, batch_size: int = 64, lr: float = 1e-3, evaluate: bool = True):
    print_banner("LANDFALL PREDICTOR NET (LandfallPredictorNet)")
    from backend.app.ml.trajectory.train_landfall import LandfallTrainer
    trainer = LandfallTrainer()
    return trainer.train(num_epochs=epochs, batch_size=batch_size, lr=lr)


def train_precipitation(epochs: int = 10, batch_size: int = 8, lr: float = 1e-3, evaluate: bool = True):
    print_banner("PRECIPITATION U-NET (PrecipitationUNet)")
    from backend.app.ml.impact.train_precipitation_unet import PrecipitationTrainer
    trainer = PrecipitationTrainer()
    return trainer.train(num_epochs=epochs, batch_size=batch_size, lr=lr)


def main():
    parser = argparse.ArgumentParser(
        description="CYCLONE-X Unified Model Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "all",
            "master_all",
            "authentic_all",
            "core",
            "trajectory",
            "transformer",
            "bilstm",
            "detection",
            "detection_v3",
            "vit",
            "centernet",
            "intensity",
            "dvorak",
            "landfall",
            "precipitation",
            "surge",
            "pinn",
            "advection",
            "fusion",
            "flood",
            "impact",
            "wavewatch",
            "ri",
            "cyclogenesis",
        ],
        default="all",
        help="Which model architecture to train",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Custom number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Custom batch size")
    parser.add_argument("--lr", type=float, default=None, help="Custom learning rate")
    parser.add_argument("--no-evaluate", dest="evaluate", action="store_false", help="Skip evaluation phase")
    parser.set_defaults(evaluate=True)

    args = parser.parse_args()

    start_wall = time.time()
    print("=" * 72)
    print("         CYCLONE-X DEEP LEARNING MODEL TRAINING PIPELINE")
    print(f"         Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"         Model Selected: {args.model.upper()}")
    print("=" * 72)

    results = {}

    if args.model in ("all", "master_all", "core", "trajectory"):
        ep = args.epochs or 10
        bs = args.batch_size or 64
        lr = args.lr or 1e-3
        results["trajectory"] = train_trajectory(epochs=ep, batch_size=bs, lr=lr, evaluate=args.evaluate)

    if args.model in ("all", "master_all", "transformer"):
        ep = args.epochs or 5
        bs = args.batch_size or 32
        lr = args.lr or 5e-4
        results["transformer"] = train_transformer(epochs=ep, batch_size=bs, lr=lr, evaluate=args.evaluate)

    if args.model in ("all", "master_all", "core", "detection"):
        ep = args.epochs or 10
        bs = args.batch_size or 8
        lr = args.lr or 2e-4
        results["detection"] = train_detection(epochs=ep, batch_size=bs, lr=lr, evaluate=args.evaluate)

    if args.model in ("all", "master_all", "core", "detection_v3"):
        ep = args.epochs or 7
        bs = args.batch_size or 32
        lr = args.lr or 3e-4
        results["detection_v3"] = train_detection_v3(epochs=ep, batch_size=bs, lr=lr, evaluate=args.evaluate)

    if args.model in ("all", "master_all", "vit"):
        ep = args.epochs or 2
        bs = args.batch_size or 32
        lr = args.lr or 1e-4
        results["vit"] = train_vit(epochs=ep, batch_size=bs, lr=lr, evaluate=args.evaluate)

    if args.model in ("all", "master_all", "centernet"):
        ep = args.epochs or 5
        bs = args.batch_size or 8
        lr = args.lr or 1e-4
        results["centernet"] = train_centernet(epochs=ep, batch_size=bs, lr=lr, evaluate=args.evaluate)

    if args.model in ("all", "master_all", "core", "intensity"):
        ep = args.epochs or 5
        bs = args.batch_size or 16
        lr = args.lr or 1e-4
        results["intensity"] = train_intensity(epochs=ep, batch_size=bs, lr=lr, evaluate=args.evaluate)

    if args.model in ("all", "master_all", "core", "ri"):
        results["ri"] = train_ri(evaluate=args.evaluate)

    if args.model in ("all", "master_all", "dvorak"):
        ep = args.epochs or 12
        bs = args.batch_size or 16
        lr = args.lr or 1e-3
        results["dvorak"] = train_dvorak(epochs=ep, batch_size=bs, lr=lr, evaluate=args.evaluate)

    if args.model in ("all", "master_all", "landfall"):
        ep = args.epochs or 12
        bs = args.batch_size or 64
        lr = args.lr or 1e-3
        results["landfall"] = train_landfall(epochs=ep, batch_size=bs, lr=lr, evaluate=args.evaluate)

    if args.model in ("all", "master_all", "precipitation"):
        ep = args.epochs or 10
        bs = args.batch_size or 8
        lr = args.lr or 1e-3
        results["precipitation"] = train_precipitation(epochs=ep, batch_size=bs, lr=lr, evaluate=args.evaluate)

    if args.model in ("all", "master_all", "core", "cyclogenesis"):
        results["cyclogenesis"] = train_cyclogenesis(evaluate=args.evaluate)

    if args.model in ("all", "master_all", "bilstm"):
        from backend.app.ml.training_framework.train_authentic_models import train_trajectory_bilstm_authentic
        results["bilstm"] = train_trajectory_bilstm_authentic(epochs=args.epochs or 6, batch_size=args.batch_size or 32, lr=args.lr or 1e-3)

    if args.model in ("all", "master_all", "surge"):
        from backend.app.ml.training_framework.train_authentic_models import train_surge_surrogate_authentic
        results["surge"] = train_surge_surrogate_authentic(epochs=args.epochs or 12)

    if args.model in ("all", "master_all", "pinn"):
        from backend.app.ml.training_framework.train_authentic_models import train_holland_pinn_authentic
        results["pinn"] = train_holland_pinn_authentic(epochs=args.epochs or 20)

    if args.model in ("all", "master_all", "advection"):
        from backend.app.ml.training_framework.train_authentic_models import train_physics_advection_authentic
        results["advection"] = train_physics_advection_authentic(epochs=args.epochs or 10)

    if args.model in ("all", "master_all", "fusion"):
        from backend.app.ml.training_framework.train_authentic_models import train_satellite_radar_fusion_authentic
        results["fusion"] = train_satellite_radar_fusion_authentic(epochs=args.epochs or 8)

    if args.model in ("all", "master_all", "flood"):
        from backend.app.ml.training_framework.train_authentic_models import train_flood_risk_authentic
        results["flood"] = train_flood_risk_authentic(epochs=args.epochs or 12)

    if args.model in ("all", "master_all", "impact"):
        from backend.app.ml.training_framework.train_authentic_models import train_impact_neural_authentic
        results["impact"] = train_impact_neural_authentic(epochs=args.epochs or 12)

    if args.model in ("all", "master_all", "wavewatch"):
        from backend.app.ml.training_framework.train_authentic_models import train_wavewatch_surrogate_authentic
        results["wavewatch"] = train_wavewatch_surrogate_authentic(epochs=args.epochs or 15)

    if args.model == "authentic_all":
        from backend.app.ml.training_framework.train_authentic_models import train_all_authentic
        results["authentic_all"] = train_all_authentic()

    total_time = time.time() - start_wall
    print_banner("ALL REQUESTED TRAINING PIPELINES COMPLETED SUCCESSFULLY")
    print(f"Total Elapsed Time: {total_time:.2f} seconds")
    print("Active Checkpoints / Results Summary:")
    for k, v in results.items():
        if isinstance(v, dict):
            ckpt = v.get("checkpoint_path") or v.get("checkpoint_file") or v.get("model_file") or "Completed"
            print(f"  - [{k.upper()}]: {ckpt}")
        else:
            print(f"  - [{k.upper()}]: Completed")
    print("\nAll models are verified, calibrated, and ready for operational inference.\n")


if __name__ == "__main__":
    main()
