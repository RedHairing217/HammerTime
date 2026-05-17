"""
predict.py — Multi-Class Detector
=====================================
Run inference with the trained multi-class tool detector.

Usage
-----
    python src/multi/predict_m.py --source path/to/image.jpg
    python src/multi/predict_m.py --source data/classes/hammer/
    python src/multi/predict_m.py --source 0
    python src/multi/predict_m.py --source data/ --save
    python src/multi/predict_m.py --source data/ --report results.json
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

CLASS_NAMES = ["hammer", "axe", "wrench", "pickaxe"]

_G = "\033[92m"; _R = "\033[91m"; _Y = "\033[93m"; _RST = "\033[0m"
def green(s):  return f"{_G}{s}{_RST}"
def red(s):    return f"{_R}{s}{_RST}"
def yellow(s): return f"{_Y}{s}{_RST}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-class tool detector inference")
    p.add_argument("--source",   required=True)
    p.add_argument("--weights",  default=str(ROOT / "runs/multi/train/weights/best.pt"))
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
        print("      Train first: python src/multi/train_m.py")
        sys.exit(1)
    return YOLO(str(p))


def parse_result(result) -> dict[str, Any]:
    src   = Path(result.path) if hasattr(result, "path") else Path("unknown")
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return dict(file=src.name, path=str(src), detections=[])
    dets = []
    for box in boxes:
        cls_id = int(box.cls[0])
        dets.append(dict(
            class_id   = cls_id,
            class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
            confidence = round(float(box.conf[0]), 4),
            bbox_xyxy  = [round(v, 1) for v in box.xyxy[0].tolist()],
        ))
    return dict(file=src.name, path=str(src), detections=dets)


def fmt_line(r: dict) -> str:
    if not r["detections"]:
        return f"{red('[NO DETECTION]')} {r['file']}"
    lines = []
    for det in r["detections"]:
        name   = det["class_name"].upper()
        conf   = det["confidence"]
        colour = green if det["class_id"] == 0 else yellow
        lines.append(f"{colour(f'[{name}]'):<20} {r['file']:<40s}  conf {colour(f'{conf:.2f}')}")
    return "\n".join(lines)


def main() -> None:
    print("\n╔══════════════════════════════════════════════╗")
    print("║  HammerTime — Multi-Class Inference        ║")
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

    n_total = len(log)
    elapsed = time.time() - t0

    class_counts: dict[str, int] = {}
    for r in log:
        for det in r["detections"]:
            name = det["class_name"]
            class_counts[name] = class_counts.get(name, 0) + 1
    n_empty = sum(1 for r in log if not r["detections"])

    print("\n" + "─" * 54)
    print(f"  Total images : {n_total}")
    for name in CLASS_NAMES:
        if name in class_counts:
            colour = green if name == "hammer" else yellow
            print(f"  {name:<14}: {colour(str(class_counts[name]))}")
    if n_empty:
        print(f"  No detection : {red(str(n_empty))}")
    if n_total:
        print(f"  Speed        : {n_total/elapsed:.1f} img/s")
    print("─" * 54)

    if args.report:
        report = dict(
            source=args.source, weights=args.weights,
            conf_threshold=args.conf,
            total=n_total, class_counts=class_counts, no_detection=n_empty,
            results=log,
        )
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"[OK ] Report -> {args.report}")

    print()


if __name__ == "__main__":
    main()
