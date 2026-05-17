"""
evaluate.py — Multi-Class Detector
=====================================
Run YOLOv8 validation on the val or test split.
Reports per-class AP@50, mAP@50-95, Precision, and Recall.

Usage
-----
    python src/multi/evaluate_m.py
    python src/multi/evaluate_m.py --split test
    python src/multi/evaluate_m.py --weights runs/multi/train/weights/best.pt
    python src/multi/evaluate_m.py --split test --v1_compare
    python src/multi/evaluate_m.py --plots
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

CLASS_NAMES = ["hammer", "axe", "wrench", "pickaxe"]

# V1 Run 9 hammer baseline — used for capstone comparison
V1_BASELINE = {
    "map50":     0.8858,
    "map50_95":  0.5829,
    "precision": 0.8743,
    "recall":    0.9098,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate multi-class tool detector")
    p.add_argument("--weights",    default=str(ROOT / "runs/multi/train/weights/best.pt"))
    p.add_argument("--data",       default=str(ROOT / "dataset/dataset.yaml"))
    p.add_argument("--conf",       type=float, default=0.20)
    p.add_argument("--iou",        type=float, default=0.50)
    p.add_argument("--imgsz",      type=int,   default=640)
    p.add_argument("--split",      default="val", choices=["train", "val", "test"])
    p.add_argument("--plots",      action="store_true")
    p.add_argument("--out",        default=str(ROOT / "eval_report_multi.json"))
    p.add_argument("--v1_compare", action="store_true",
                   help="Compare hammer class against V1 Run 9 baseline (use with --split test)")
    return p.parse_args()


def main() -> None:
    print("\n╔══════════════════════════════════════════════╗")
    print("║  HammerTime — Multi-Class Evaluation       ║")
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

    per_class = {}
    try:
        ap50_per_class = metrics.box.ap50
        p_per_class    = metrics.box.p
        r_per_class    = metrics.box.r
        maps_per_class = metrics.box.maps
        for i, name in enumerate(CLASS_NAMES):
            if i < len(ap50_per_class):
                per_class[name] = {
                    "ap50":      round(float(ap50_per_class[i]), 4),
                    "map50_95":  round(float(maps_per_class[i]), 4) if i < len(maps_per_class) else None,
                    "precision": round(float(p_per_class[i]), 4)    if i < len(p_per_class)    else None,
                    "recall":    round(float(r_per_class[i]), 4)    if i < len(r_per_class)    else None,
                }
    except Exception:
        pass

    split_label = f"{args.split.upper()} SET"
    print("\n" + "=" * 62)
    print(f"  EVALUATION RESULTS  [{split_label}]")
    print("=" * 62)
    if map50 is not None: print(f"  mAP@50      : {map50:.4f}")
    if map95  is not None: print(f"  mAP@50-95   : {map95:.4f}")
    if prec   is not None: print(f"  Precision   : {prec:.4f}")
    if rec    is not None: print(f"  Recall      : {rec:.4f}")

    if per_class:
        print(f"\n  {'Class':<12} {'AP@50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
        print(f"  {'-'*52}")
        for name, vals in per_class.items():
            marker = " *" if name == "hammer" else ""
            p_str  = f"{vals['precision']:.4f}" if vals['precision'] is not None else "  n/a"
            r_str  = f"{vals['recall']:.4f}"    if vals['recall']    is not None else "  n/a"
            m_str  = f"{vals['map50_95']:.4f}"  if vals['map50_95']  is not None else "  n/a"
            print(f"  {name:<12} {vals['ap50']:>8.4f} {m_str:>10} {p_str:>10} {r_str:>8}{marker}")
        print(f"  * primary target")

    if args.v1_compare and args.split == "test":
        h     = per_class.get("hammer", {})
        h_p   = h.get("precision")
        h_r   = h.get("recall")
        h_ap  = h.get("ap50")
        h_m95 = h.get("map50_95")
        print(f"\n  V1 Run 9 baseline comparison (hammer class, same 93-image test set):")
        print(f"  {'Metric':<14} {'V1':>8} {'V2':>8} {'Delta':>8}")
        print(f"  {'-'*42}")
        pairs = [
            ("AP@50",     V1_BASELINE["map50"],     h_ap),
            ("mAP@50-95", V1_BASELINE["map50_95"],  h_m95),
            ("Precision", V1_BASELINE["precision"], h_p),
            ("Recall",    V1_BASELINE["recall"],    h_r),
        ]
        for label, v1, v2 in pairs:
            if v2 is not None:
                delta = v2 - v1
                sign  = "+" if delta >= 0 else ""
                print(f"  {label:<14} {v1:>8.4f} {v2:>8.4f} {sign}{delta:>7.4f}")

    print("=" * 62)

    report = dict(
        weights=args.weights, data=args.data, split=args.split,
        conf=args.conf, iou=args.iou,
        map50=map50, map50_95=map95, precision=prec, recall=rec,
        per_class=per_class,
    )
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[OK ] Report -> {args.out}\n")


if __name__ == "__main__":
    main()
