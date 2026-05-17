"""
evaluate.py — Binary Classifier
================================
Run YOLOv8 validation on the val or test split and print a detailed report.

Usage
-----
    python src/binary/evaluate_b.py
    python src/binary/evaluate_b.py --split test
    python src/binary/evaluate_b.py --weights runs/binary/train/weights/best.pt
    python src/binary/evaluate_b.py --plots
    python src/binary/evaluate_b.py --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERR] pip install ultralytics>=8.0")
    sys.exit(1)

# Resolve repo root regardless of where the script is called from
ROOT = Path(__file__).parent.parent.parent if Path(__file__).parent.parent.name == "src" else Path(__file__).parent

# V1 Run 9 baseline — reference for binary classifier comparison
V1_BASELINE = {
    "map50":     0.8858,
    "map50_95":  0.5829,
    "precision": 0.8743,
    "recall":    0.9098,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate binary classifier")
    p.add_argument("--weights",    default=str(ROOT / "runs/binary/train/weights/best.pt"))
    p.add_argument("--data",       default=str(ROOT / "dataset/dataset.yaml"))
    p.add_argument("--conf",       type=float, default=0.20)
    p.add_argument("--iou",        type=float, default=0.50)
    p.add_argument("--imgsz",      type=int,   default=640)
    p.add_argument("--split",      default="val", choices=["train", "val", "test"])
    p.add_argument("--plots",      action="store_true")
    p.add_argument("--out",        default=str(ROOT / "eval_report_binary.json"))
    p.add_argument("--v1_compare", action="store_true",
                   help="Compare against V1 Run 9 baseline (use with --split test)")
    return p.parse_args()


def main() -> None:
    print("\n╔══════════════════════════════════════════════╗")
    print("║  HammerTime — Binary Classifier Evaluation ║")
    print("╚══════════════════════════════════════════════╝\n")

    args = parse_args()

    for label, path in [("Weights", args.weights), ("Dataset", args.data)]:
        if not Path(path).exists():
            print(f"[ERR] {label} not found: {path}")
            sys.exit(1)

    if args.split == "test":
        print("  Evaluating on held-out test set.\n")

    model   = YOLO(args.weights)
    metrics = model.val(
        data=args.data, conf=args.conf, iou=args.iou,
        imgsz=args.imgsz, split=args.split,
        plots=args.plots, verbose=True,
    )

    try:
        results_dict = metrics.results_dict
        map50 = float(results_dict.get("metrics/mAP50(B)", 0))
        map95 = float(results_dict.get("metrics/mAP50-95(B)", 0))
        prec  = float(results_dict.get("metrics/precision(B)", 0))
        rec   = float(results_dict.get("metrics/recall(B)", 0))
    except Exception as e:
        print(f"[WARN] Could not parse metrics: {e}")
        map50 = map95 = prec = rec = None

    split_label = f"{args.split.upper()} SET"
    print("\n" + "=" * 52)
    print(f"  EVALUATION RESULTS  [{split_label}]")
    print("=" * 52)
    if map50 is not None: print(f"  mAP@50      : {map50:.4f}")
    if map95  is not None: print(f"  mAP@50-95   : {map95:.4f}")
    if prec   is not None: print(f"  Precision   : {prec:.4f}")
    if rec    is not None: print(f"  Recall      : {rec:.4f}")

    if map50 is not None:
        if   map50 >= 0.90: v = "Excellent"
        elif map50 >= 0.75: v = "Target achieved (0.75+)"
        elif map50 >= 0.60: v = "Below target — collect more data"
        else:               v = "Poor — review labels and data quality"
        print(f"\n  {v}")

    if args.v1_compare and args.split == "test":
        print(f"\n  V1 Run 9 baseline comparison:")
        print(f"  {'Metric':<14} {'V1':>8} {'V2':>8} {'Delta':>8}")
        print(f"  {'-'*42}")
        pairs = [
            ("mAP@50",    V1_BASELINE["map50"],     map50),
            ("mAP@50-95", V1_BASELINE["map50_95"],  map95),
            ("Precision",  V1_BASELINE["precision"], prec),
            ("Recall",     V1_BASELINE["recall"],    rec),
        ]
        for label, v1, v2 in pairs:
            if v2 is not None:
                delta = v2 - v1
                sign  = "+" if delta >= 0 else ""
                print(f"  {label:<14} {v1:>8.4f} {v2:>8.4f} {sign}{delta:>7.4f}")

    print("=" * 52)

    report = dict(
        weights=args.weights, data=args.data, split=args.split,
        conf=args.conf, iou=args.iou,
        map50=map50, map50_95=map95, precision=prec, recall=rec,
    )
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[OK ] Report -> {args.out}\n")


if __name__ == "__main__":
    main()
