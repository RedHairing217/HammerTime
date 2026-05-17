"""
predict.py — Binary Classifier
================================
Run inference with the trained binary classifier.

Usage
-----
    python src/binary/predict_b.py --source path/to/image.jpg
    python src/binary/predict_b.py --source data/classes/hammer/
    python src/binary/predict_b.py --source 0                      # webcam
    python src/binary/predict_b.py --source data/ --save
    python src/binary/predict_b.py --source data/ --report results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERR] pip install ultralytics>=8.0")
    sys.exit(1)

# Resolve repo root regardless of where the script is called from
ROOT = Path(__file__).parent.parent.parent if Path(__file__).parent.parent.name == "src" else Path(__file__).parent

_G = "\033[92m"; _R = "\033[91m"; _RST = "\033[0m"
def green(s): return f"{_G}{s}{_RST}"
def red(s):   return f"{_R}{s}{_RST}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binary classifier inference")
    p.add_argument("--source",   required=True)
    p.add_argument("--weights",  default=str(ROOT / "runs/binary/train/weights/best.pt"))
    p.add_argument("--conf",     type=float, default=0.20)
    p.add_argument("--iou",      type=float, default=0.45)
    p.add_argument("--imgsz",    type=int,   default=640)
    p.add_argument("--device",   default="")
    p.add_argument("--save",     action="store_true")
    p.add_argument("--save_txt", action="store_true")
    p.add_argument("--show",     action="store_true")
    p.add_argument("--report",   default="")
    p.add_argument("--quiet",    action="store_true")
    return p.parse_args()


def load_model(weights: str) -> YOLO:
    p = Path(weights)
    if not p.exists():
        print(f"[ERR] Weights not found: {p}")
        print("      Train first: python src/binary/train_b.py")
        sys.exit(1)
    return YOLO(str(p))


def parse_result(result) -> dict[str, Any]:
    src   = Path(result.path) if hasattr(result, "path") else Path("unknown")
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return dict(file=src.name, path=str(src), detected=False, detections=[])
    dets = []
    for box in boxes:
        cls_id = int(box.cls[0])
        dets.append(dict(
            class_id   = cls_id,
            class_name = result.names[cls_id],
            confidence = round(float(box.conf[0]), 4),
            bbox_xyxy  = [round(v, 1) for v in box.xyxy[0].tolist()],
        ))
    return dict(file=src.name, path=str(src), detected=True, detections=dets)


def fmt_line(r: dict) -> str:
    if r["detected"]:
        top  = max(r["detections"], key=lambda d: d["confidence"])
        name = top["class_name"].upper()
        conf = top["confidence"]
        return f"{green(f'[{name}]'):<20} {r['file']:<40s}  conf {green(f'{conf:.2f}')}"
    return f"{red('[NOT DETECTED]')} {r['file']}"


def main() -> None:
    print("\n╔══════════════════════════════════════════════╗")
    print("║  HammerTime — Binary Classifier Inference  ║")
    print("╚══════════════════════════════════════════════╝\n")

    args  = parse_args()
    model = load_model(args.weights)

    kw = dict(
        source=args.source, conf=args.conf, iou=args.iou,
        imgsz=args.imgsz, save=args.save, save_txt=args.save_txt,
        show=args.show, verbose=False, stream=True,
    )
    if args.device:
        kw["device"] = args.device

    print(f"[INFO] Source  : {args.source}")
    print(f"[INFO] Conf    : {args.conf}")
    print(f"[INFO] Weights : {args.weights}\n")

    log: list[dict] = []
    t0 = time.time()
    for result in model.predict(**kw):
        r = parse_result(result)
        log.append(r)
        if not args.quiet:
            print(fmt_line(r))

    n_total    = len(log)
    n_detected = sum(r["detected"] for r in log)
    elapsed    = time.time() - t0

    print("\n" + "─" * 54)
    print(f"  Total      : {n_total}")
    print(f"  Detected   : {green(str(n_detected))}")
    print(f"  Not found  : {red(str(n_total - n_detected))}")
    if n_total:
        print(f"  Speed      : {n_total/elapsed:.1f} img/s")
    print("─" * 54)

    if args.report:
        report = dict(
            source=args.source, weights=args.weights,
            conf_threshold=args.conf,
            total=n_total, detected=n_detected,
            not_detected=n_total - n_detected,
            results=log,
        )
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"[OK ] Report -> {args.report}")

    print()


if __name__ == "__main__":
    main()
