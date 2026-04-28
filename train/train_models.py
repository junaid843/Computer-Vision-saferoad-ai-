"""
train/train_models.py
──────────────────────
Train YOLOv8 models for:
  1. Seatbelt detection  (Person_with_seatbelt / Person_without_seatbelt)
  2. License plate detection

Datasets from Roboflow:
  Seatbelt:  https://universe.roboflow.com/renuka-ed2ia/seat-belt-detection-c2nia
  Plates:    https://universe.roboflow.com/manasa-n-n/car-license-plate-detection-maehj

Usage:
    python train/train_models.py --task seatbelt --epochs 50 --imgsz 640
    python train/train_models.py --task plate     --epochs 50 --imgsz 640
"""

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
    from roboflow import Roboflow
except ImportError:
    raise ImportError("Run: pip install ultralytics roboflow")


# ── Dataset download helpers ──────────────────────────────────────────────────
def download_seatbelt_dataset(api_key: str, dest: str = "datasets/seatbelt"):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("renuka-ed2ia").project("seat-belt-detection-c2nia")
    dataset = project.version(1).download("yolov8", location=dest)
    return Path(dataset.location) / "data.yaml"


def download_plate_dataset(api_key: str, dest: str = "datasets/plates"):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("manasa-n-n").project("car-license-plate-detection-maehj")
    dataset = project.version(1).download("yolov8", location=dest)
    return Path(dataset.location) / "data.yaml"


# ── Training ──────────────────────────────────────────────────────────────────
def train(
    data_yaml:   str,
    output_name: str,
    epochs:      int   = 50,
    imgsz:       int   = 640,
    batch:       int   = 16,
    device:      str   = "0",          # GPU id or 'cpu'
    base_model:  str   = "yolov8n.pt", # nano → fast; yolov8s.pt for better accuracy
):
    model = YOLO(base_model)
    results = model.train(
        data    = data_yaml,
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = batch,
        device  = device,
        project = "runs/train",
        name    = output_name,
        patience= 15,         # early stopping
        save    = True,
        plots   = True,
    )
    best_weights = Path(f"runs/train/{output_name}/weights/best.pt")
    print(f"\n✓ Training complete. Best weights: {best_weights}")
    return best_weights


# ── Validation ────────────────────────────────────────────────────────────────
def validate(weights: str, data_yaml: str):
    model = YOLO(weights)
    metrics = model.val(data=data_yaml)
    print(f"\nmAP50     : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO models for SafeRoad AI")
    parser.add_argument("--task",      choices=["seatbelt", "plate"], required=True)
    parser.add_argument("--api_key",   default="",   help="Roboflow API key")
    parser.add_argument("--data_yaml", default="",   help="Path to existing data.yaml (skip download)")
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--imgsz",     type=int, default=640)
    parser.add_argument("--batch",     type=int, default=16)
    parser.add_argument("--device",    default="0")
    parser.add_argument("--base",      default="yolov8n.pt")
    parser.add_argument("--validate",  action="store_true")
    args = parser.parse_args()

    # Resolve data.yaml
    if args.data_yaml:
        yaml_path = args.data_yaml
    elif args.api_key:
        if args.task == "seatbelt":
            yaml_path = download_seatbelt_dataset(args.api_key)
        else:
            yaml_path = download_plate_dataset(args.api_key)
    else:
        raise ValueError("Provide either --data_yaml or --api_key to download dataset.")

    # Train
    output_name = f"{args.task}_yolov8"
    weights = train(
        data_yaml   = str(yaml_path),
        output_name = output_name,
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        device      = args.device,
        base_model  = args.base,
    )

    # Optional validation
    if args.validate:
        validate(str(weights), str(yaml_path))
