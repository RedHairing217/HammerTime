"""
count_dataset.py
================
Counts dataset composition by reading from data/classes/ subdirectories.
Reports per-folder image counts and per-class annotation counts from data/labels/.

Usage
-----
    python src/shared/count_dataset.py
    python src/shared/count_dataset.py --verbose
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Resolve repo root regardless of where the script is called from
ROOT = Path(__file__).parent.parent.parent if Path(__file__).parent.parent.name == "src" else Path(__file__).parent

CLASS_NAMES     = ["hammer", "axe", "wrench", "pickaxe"]
BACKGROUND_DIRS = {"background"}
SUPPORTED_IMG   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Count dataset composition by class folder")
    p.add_argument("--data_dir", default=str(ROOT / "data"))
    p.add_argument("--verbose",  action="store_true", help="List all filenames per folder")
    return p.parse_args()


def main() -> None:
    args        = parse_args()
    data_dir    = Path(args.data_dir)
    classes_dir = data_dir / "classes"
    lbl_dir     = data_dir / "labels"

    if not classes_dir.exists():
        print(f"[ERR] data/classes/ not found at {classes_dir}")
        return

    class_dirs = sorted([d for d in classes_dir.iterdir() if d.is_dir()])

    folder_counts     = {}
    missing_labels    = []
    annotation_counts = {name: 0 for name in CLASS_NAMES}
    total_images      = 0
    total_bg          = 0

    for class_dir in class_dirs:
        is_bg = class_dir.name.lower() in BACKGROUND_DIRS
        images = sorted([f for f in class_dir.iterdir()
                         if f.suffix.lower() in SUPPORTED_IMG])
        folder_counts[class_dir.name] = images
        total_images += len(images)
        if is_bg:
            total_bg += len(images)
            continue

        for img in images:
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                missing_labels.append(img.name)
                continue
            for line in lbl.read_text().strip().splitlines():
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id < len(CLASS_NAMES):
                        annotation_counts[CLASS_NAMES[cls_id]] += 1

    print(f"\n{'='*52}")
    print(f"  DATASET COMPOSITION")
    print(f"{'='*52}")
    print(f"  Total images     : {total_images}")
    print(f"  Labeled images   : {total_images - total_bg}")
    print(f"  Background images: {total_bg}")

    print(f"\n  Images per folder:")
    for name, images in folder_counts.items():
        is_bg = name.lower() in BACKGROUND_DIRS
        tag   = " (background)" if is_bg else ""
        print(f"    {name:<14} {len(images):>4}{tag}")

    print(f"\n  Annotations per class:")
    for name, count in annotation_counts.items():
        print(f"    {name:<14} {count:>5}")

    if missing_labels:
        print(f"\n  Missing labels   : {len(missing_labels)}")
        for name in missing_labels:
            print(f"    {name}")

    print(f"{'='*52}\n")

    if args.verbose:
        for name, images in folder_counts.items():
            print(f"{name}/")
            for img in images:
                print(f"  {img.name}")
            print()


if __name__ == "__main__":
    main()
