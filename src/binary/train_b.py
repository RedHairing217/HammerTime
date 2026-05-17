"""
train.py — Binary Classifier
=============================
Train YOLOv8-Nano on a single-class binary dataset prepared by prepare_dataset.py
with --target_class.

Run prepare_dataset.py --target_class <class> first.

Usage
-----
    python src/binary/train_b.py
    python src/binary/train_b.py --epochs 300 --batch 8
    python src/binary/train_b.py --resume
    python src/binary/train_b.py --no_aug
    python src/binary/train_b.py --export

Outputs
-------
    runs/binary/train/
        weights/best.pt
        weights/last.pt
        results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERR] ultralytics not found.  Run:  pip install ultralytics>=8.0")
    sys.exit(1)

# Resolve repo root regardless of where the script is called from
ROOT = Path(__file__).parent.parent.parent if Path(__file__).parent.parent.name == "src" else Path(__file__).parent

DEFAULTS = dict(
    weights       = "yolov8n.pt",
    data          = str(ROOT / "dataset" / "dataset.yaml"),
    imgsz         = 640,
    batch         = 16,
    workers       = 4,
    device        = "mps",
    cache         = False,
    epochs        = 400,
    patience      = 50,
    lr0           = 1e-2,
    lrf           = 1e-2,
    warmup_epochs = 3,
    cos_lr        = True,
    hsv_h         = 0.015,
    hsv_s         = 0.7,
    hsv_v         = 0.4,
    degrees       = 45,
    translate     = 0.1,
    scale         = 0.6,
    shear         = 10,
    flipud        = 0.1,
    fliplr        = 0.5,
    mosaic        = 1.0,
    mixup         = 0.1,
    copy_paste    = 0.1,
    weight_decay  = 5e-4,
    dropout       = 0.0,
    project       = str(ROOT / "runs" / "binary"),
    name          = "train",
    exist_ok      = False,
    save_period   = -1,
    plots         = True,
    verbose       = True,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLOv8-Nano binary classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights",  default=DEFAULTS["weights"])
    p.add_argument("--data",     default=DEFAULTS["data"])
    p.add_argument("--epochs",   type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--patience", type=int,   default=DEFAULTS["patience"])
    p.add_argument("--imgsz",    type=int,   default=DEFAULTS["imgsz"])
    p.add_argument("--batch",    type=int,   default=DEFAULTS["batch"])
    p.add_argument("--workers",  type=int,   default=DEFAULTS["workers"])
    p.add_argument("--device",   default=DEFAULTS["device"])
    p.add_argument("--lr0",      type=float, default=DEFAULTS["lr0"])
    p.add_argument("--project",  default=DEFAULTS["project"])
    p.add_argument("--name",     default=DEFAULTS["name"])
    p.add_argument("--exist_ok", action="store_true")
    p.add_argument("--resume",   action="store_true")
    p.add_argument("--export",   action="store_true")
    p.add_argument("--no_aug",   action="store_true")
    return p.parse_args()


def preflight(args: argparse.Namespace) -> None:
    if not args.resume and not Path(args.data).exists():
        print(f"[ERR] dataset.yaml not found: {args.data}")
        print("      Run: python src/shared/prepare_dataset.py --target_class <class>")
        sys.exit(1)
    if not args.resume and not Path(args.weights).exists():
        print(f"[INFO] {args.weights} not local — ultralytics will download it.")
    print("[OK ] Pre-flight passed.")


def build_kwargs(args: argparse.Namespace) -> dict:
    kw = {k: v for k, v in DEFAULTS.items() if k not in ("weights", "data")}
    kw.update(dict(
        data=args.data, epochs=args.epochs, patience=args.patience,
        imgsz=args.imgsz, batch=args.batch, workers=args.workers,
        lr0=args.lr0, project=args.project, name=args.name,
        exist_ok=args.exist_ok,
    ))
    if args.device:
        kw["device"] = args.device
    if args.no_aug:
        kw.update(degrees=0, scale=0.5, shear=0,
                  mixup=0.0, copy_paste=0.0, mosaic=0.0, flipud=0.0)
        print("[WARN] Heavy augmentations disabled")
    return kw


def run_training(args: argparse.Namespace) -> Path:
    if args.resume:
        last_pt = Path(args.project) / args.name / "weights" / "last.pt"
        if not last_pt.exists():
            print(f"[ERR] Cannot resume — not found: {last_pt}")
            sys.exit(1)
        print(f"[INFO] Resuming from {last_pt}")
        model   = YOLO(str(last_pt))
        results = model.train(resume=True)
    else:
        model   = YOLO(args.weights)
        kw      = build_kwargs(args)
        print("\n" + "─" * 54)
        print("  Training config")
        print("─" * 54)
        for k, v in sorted(kw.items()):
            print(f"    {k:<22s}: {v}")
        print("─" * 54 + "\n")
        results = model.train(**kw)

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n[OK ] Training complete -> {best_pt}")
    return best_pt


def print_metrics(best_pt: Path) -> None:
    csv_path = best_pt.parent.parent / "results.csv"
    if not csv_path.exists():
        return
    try:
        rows = list(csv.DictReader(open(csv_path)))
        if not rows:
            return
        last = {k.strip(): v.strip() for k, v in rows[-1].items()}

        def _f(keys):
            for k in keys:
                if k in last:
                    try: return float(last[k])
                    except ValueError: pass
            return None

        map50 = _f(["metrics/mAP50(B)", "metrics/mAP_0.5"])
        map95 = _f(["metrics/mAP50-95(B)", "metrics/mAP_0.5:0.95"])

        print("\n" + "=" * 48)
        print("  FINAL METRICS")
        print("=" * 48)
        if map50 is not None: print(f"  mAP@50      : {map50:.4f}")
        if map95  is not None: print(f"  mAP@50-95   : {map95:.4f}")
        if map50  is not None:
            if   map50 >= 0.90: verdict = "Excellent"
            elif map50 >= 0.75: verdict = "Target achieved"
            elif map50 >= 0.60: verdict = "Below target — add more data"
            else:               verdict = "Poor — review data quality"
            print(f"  Verdict     : {verdict}")
        print("=" * 48 + "\n")
    except Exception as e:
        print(f"[WARN] Could not parse results.csv: {e}")


def export_onnx(best_pt: Path) -> None:
    print("[INFO] Exporting to ONNX ...")
    out = YOLO(str(best_pt)).export(format="onnx", imgsz=640, simplify=True)
    print(f"[OK ] ONNX -> {out}")


def main() -> None:
    print("\n╔══════════════════════════════════════════════╗")
    print("║  HammerTime — Binary Classifier Training   ║")
    print("╚══════════════════════════════════════════════╝\n")

    args = parse_args()
    preflight(args)

    t0      = time.time()
    best_pt = run_training(args)
    elapsed = time.time() - t0

    print_metrics(best_pt)
    print(f"[INFO] Total time: {elapsed/60:.1f} min")

    if args.export:
        export_onnx(best_pt)

    print(f"\n[NEXT] Evaluate: python src/binary/evaluate_b.py")
    print(f"       Predict:  python src/binary/predict_b.py --source <image_or_dir>\n")


if __name__ == "__main__":
    main()
