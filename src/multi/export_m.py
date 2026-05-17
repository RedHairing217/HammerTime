"""
export.py — Multi-Class Detector
=====================================
Export trained best.pt to a deployment format.

Usage
-----
    python src/multi/export.py
    python src/multi/export.py --format tflite
    python src/multi/export.py --format coreml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERR] pip install ultralytics>=8.0")
    sys.exit(1)

# Resolve repo root regardless of where the script is called from
ROOT = Path(__file__).parent.parent.parent if Path(__file__).parent.parent.name == "src" else Path(__file__).parent

FORMATS = ["onnx", "tflite", "coreml", "torchscript", "engine"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export multi-class tool detector")
    p.add_argument("--weights", default=str(ROOT / "runs/multi/train/weights/best.pt"))
    p.add_argument("--format",  nargs="+", default=["onnx"], choices=FORMATS)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--half",    action="store_true")
    p.add_argument("--int8",    action="store_true")
    return p.parse_args()


def main() -> None:
    print("\n╔══════════════════════════════════════════════╗")
    print("║  HammerTime — Multi-Class Export           ║")
    print("╚══════════════════════════════════════════════╝\n")

    args = parse_args()
    pt   = Path(args.weights)

    if not pt.exists():
        print(f"[ERR] Weights not found: {pt}")
        print("      Train first: python src/multi/train_m.py")
        sys.exit(1)

    model = YOLO(str(pt))

    for fmt in args.format:
        print(f"[INFO] Exporting -> {fmt.upper()} ...")
        try:
            out = model.export(
                format=fmt, imgsz=args.imgsz,
                simplify=(fmt == "onnx"),
                half=args.half, int8=args.int8,
            )
            print(f"[OK ] -> {out}")
        except Exception as e:
            print(f"[ERR] {fmt} export failed: {e}")

    print()


if __name__ == "__main__":
    main()
