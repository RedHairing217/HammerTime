"""
analyze_failures.py — Binary Classifier False Negative Analysis

Usage:
    python src/binary/analyze_failures.py
    python src/binary/analyze_failures.py --split test
    python src/binary/analyze_failures.py --conf 0.20 --iou 0.50
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

# Resolve repo root regardless of where the script is called from
ROOT = Path(__file__).parent.parent.parent if Path(__file__).parent.parent.name == "src" else Path(__file__).parent

DEFAULTS = {
    "weights":    str(ROOT / "runs/binary/train/weights/best.pt"),
    "data":       str(ROOT / "dataset/dataset.yaml"),
    "conf":       0.20,
    "iou":        0.50,
    "split":      "val",
    "out":        str(ROOT / "failure_report_binary.json"),
    "output_dir": str(ROOT / "outputs/binary/false_negatives"),
}

GT_COLOUR   = (0, 0, 255)
MISS_COLOUR = (0, 0, 255)
HIT_COLOUR  = (0, 200, 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="False negative analysis for binary classifier")
    p.add_argument("--weights",    default=DEFAULTS["weights"])
    p.add_argument("--data",       default=DEFAULTS["data"])
    p.add_argument("--conf",       type=float, default=DEFAULTS["conf"])
    p.add_argument("--iou",        type=float, default=DEFAULTS["iou"])
    p.add_argument("--split",      default=DEFAULTS["split"], choices=["train", "val", "test"])
    p.add_argument("--out",        default=DEFAULTS["out"])
    p.add_argument("--output_dir", default=DEFAULTS["output_dir"])
    return p.parse_args()


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_gt_labels(label_path, img_w, img_h):
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        if int(parts[0]) != 0:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append(yolo_to_xyxy(cx, cy, w, h, img_w, img_h))
    return boxes


def box_area_pct(x1, y1, x2, y2, img_w, img_h):
    return round(100 * (x2 - x1) * (y2 - y1) / (img_w * img_h), 2) if img_w * img_h > 0 else 0.0


def box_position(cx_norm, cy_norm):
    h = "left" if cx_norm < 0.33 else ("right" if cx_norm > 0.67 else "centre")
    v = "top"  if cy_norm < 0.33 else ("bottom" if cy_norm > 0.67 else "middle")
    return f"{v}-{h}"


def aspect_ratio(x1, y1, x2, y2):
    w = x2 - x1; h = y2 - y1
    return round(w / h, 2) if h > 0 else 0.0


def size_label(area_pct):
    if area_pct < 1.0:  return "tiny"
    if area_pct < 5.0:  return "small"
    if area_pct < 15.0: return "medium"
    return "large"


def main() -> None:
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[ERR] Weights not found: {weights_path}")
        return

    import yaml
    with open(args.data) as f:
        dataset_cfg = yaml.safe_load(f)

    dataset_root = Path(dataset_cfg.get("path", "."))
    img_dir = dataset_root / dataset_cfg.get(args.split, f"images/{args.split}")
    lbl_dir = img_dir.parent.parent / "labels" / args.split

    if not img_dir.exists():
        print(f"[ERR] Image directory not found: {img_dir}")
        return

    image_paths = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ])

    print(f"\n{'='*56}")
    print(f"  FALSE NEGATIVE ANALYSIS  [BINARY]")
    print(f"{'='*56}")
    print(f"  Weights : {weights_path}")
    print(f"  Images  : {len(image_paths)} ({args.split} split)")
    print(f"  Conf    : {args.conf}   IoU threshold: {args.iou}")
    print(f"{'='*56}\n")

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    model = YOLO(str(weights_path))

    false_negatives = []
    total_gt        = 0
    total_matched   = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        label_path = lbl_dir / (img_path.stem + ".txt")
        gt_boxes   = load_gt_labels(label_path, img_w, img_h)

        if not gt_boxes:
            continue

        total_gt += len(gt_boxes)

        results    = model(str(img_path), conf=args.conf, verbose=False)[0]
        pred_boxes = [(int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                       int(b.xyxy[0][2]), int(b.xyxy[0][3]),
                       float(b.conf[0])) for b in results.boxes]

        matched_gt = set()
        for pred in pred_boxes:
            best_iou = 0.0; best_gt_i = -1
            for i, gt in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                score = iou(pred[:4], gt)
                if score > best_iou:
                    best_iou = score; best_gt_i = i
            if best_iou >= args.iou and best_gt_i >= 0:
                matched_gt.add(best_gt_i)

        missed = [gt for i, gt in enumerate(gt_boxes) if i not in matched_gt]
        total_matched += len(matched_gt)

        if not missed:
            continue

        annotated = img.copy()
        for i, gt in enumerate(gt_boxes):
            colour = HIT_COLOUR if i in matched_gt else MISS_COLOUR
            cv2.rectangle(annotated, (gt[0], gt[1]), (gt[2], gt[3]), colour, 2)
            cv2.putText(annotated, "HIT" if i in matched_gt else "MISS",
                        (gt[0], gt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        for pred in pred_boxes:
            cv2.rectangle(annotated, (pred[0], pred[1]), (pred[2], pred[3]), (255, 150, 0), 1)
            cv2.putText(annotated, f"{pred[4]:.2f}", (pred[0], pred[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 0), 1)

        cv2.imwrite(str(out_dir / img_path.name), annotated)

        for gt in missed:
            x1, y1, x2, y2 = gt
            cx_norm  = ((x1 + x2) / 2) / img_w
            cy_norm  = ((y1 + y2) / 2) / img_h
            area_pct = box_area_pct(x1, y1, x2, y2, img_w, img_h)
            ar       = aspect_ratio(x1, y1, x2, y2)
            false_negatives.append({
                "image":        img_path.name,
                "box_xyxy":     [x1, y1, x2, y2],
                "area_pct":     area_pct,
                "size":         size_label(area_pct),
                "aspect_ratio": ar,
                "orientation":  "landscape" if ar > 1.2 else ("portrait" if ar < 0.8 else "square"),
                "position":     box_position(cx_norm, cy_norm),
                "cx_norm":      round(cx_norm, 3),
                "cy_norm":      round(cy_norm, 3),
            })

    total_fn = len(false_negatives)
    recall   = round(total_matched / total_gt, 4) if total_gt > 0 else 0.0

    print(f"  Total GT boxes   : {total_gt}")
    print(f"  Matched          : {total_matched}")
    print(f"  False negatives  : {total_fn}")
    print(f"  Recall (IoU={args.iou}) : {recall}")

    if false_negatives:
        sizes = {}
        for fn in false_negatives:
            sizes[fn["size"]] = sizes.get(fn["size"], 0) + 1
        print("\n  Size distribution:")
        for s, count in sorted(sizes.items(), key=lambda x: -x[1]):
            print(f"    {s:<10} {count:>3}  ({round(100*count/total_fn,1)}%)")

    print(f"\n  Annotated images saved to: {out_dir}/")

    with open(args.out, "w") as f:
        json.dump({
            "weights": str(weights_path), "split": args.split,
            "conf_threshold": args.conf, "iou_threshold": args.iou,
            "total_gt": total_gt, "total_matched": total_matched,
            "total_fn": total_fn, "recall": recall,
            "false_negatives": false_negatives,
        }, f, indent=2)

    print(f"  Report saved to: {args.out}")
    print(f"\n{'='*56}\n")


if __name__ == "__main__":
    main()
