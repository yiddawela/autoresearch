"""
Cross-dataset transfer evaluation on PubTables-1M.

Tests whether caption-aware training (Formulation B) degrades table-body
localisation when transferring to PubTables-1M, which has NO caption annotations.

This evaluates:
  - Pre-trained TATR (table-only baseline)
  - Fine-tuned TATR (Formulation B, merged table+caption)
  - AP@50 on table-body detection only

Usage:
    uv run cross_dataset_eval.py --n-samples 500
    uv run cross_dataset_eval.py --n-samples 2000 --output cross_dataset_results.json
"""

import _paths  # noqa: F401
import argparse
import json
import os
import random
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from crop_tables import detect_tables, get_device, load_model, suppress_duplicates

DEFAULT_FINETUNED = os.path.join(
    os.path.expanduser("~"), ".cache", "table_cropper", "checkpoints",
    "phase5_resumed_final"
)

PUBTABLES_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "table_cropper", "pubtables1m"
)


def _iou(a: tuple, b: tuple) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap_at_threshold(
    detections: list[dict],
    gt_boxes: list[dict],
    iou_threshold: float = 0.5,
) -> float:
    """Compute AP at a single IoU threshold."""
    if not gt_boxes:
        return 1.0 if not detections else 0.0
    if not detections:
        return 0.0

    gt_by_image = {}
    for g in gt_boxes:
        gt_by_image.setdefault(g["image_id"], []).append(g["bbox"])

    det_sorted = sorted(detections, key=lambda d: -d["score"])
    tp = np.zeros(len(det_sorted))
    fp = np.zeros(len(det_sorted))
    n_gt = len(gt_boxes)
    matched = {}

    for i, det in enumerate(det_sorted):
        img_id = det["image_id"]
        gt_list = gt_by_image.get(img_id, [])
        if img_id not in matched:
            matched[img_id] = set()

        best_iou = 0.0
        best_j = -1
        for j, gt_box in enumerate(gt_list):
            iou = _iou(tuple(det["bbox"]), tuple(gt_box))
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j not in matched[img_id]:
            tp[i] = 1
            matched[img_id].add(best_j)
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / n_gt
    precision = cum_tp / (cum_tp + cum_fp)

    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recall >= t
        if mask.any():
            ap += np.max(precision[mask])
    ap /= 11.0
    return float(ap)


def recall_at_iou(detections, gt_all, threshold):
    matched = set()
    gt_by_img = {}
    for g in gt_all:
        gt_by_img.setdefault(g["image_id"], []).append(g["bbox"])
    for det in sorted(detections, key=lambda d: -d["score"]):
        img_id = det["image_id"]
        for j, gt_box in enumerate(gt_by_img.get(img_id, [])):
            key = (img_id, j)
            if key not in matched and _iou(tuple(det["bbox"]), tuple(gt_box)) >= threshold:
                matched.add(key)
                break
    return len(matched) / max(1, len(gt_all))


def evaluate_on_pubtables(
    n_samples: int = 500,
    finetuned_path: str = DEFAULT_FINETUNED,
    confidence: float = 0.5,
    output_path: str | None = None,
    seed: int = 42,
) -> dict:
    """Run cross-dataset transfer evaluation on PubTables-1M local files."""
    ann_dir = os.path.join(PUBTABLES_DIR, "test_raw")
    img_dir = os.path.join(PUBTABLES_DIR, "test_images")

    if not os.path.isdir(ann_dir) or not os.path.isdir(img_dir):
        print(f"Error: PubTables-1M not extracted. Expected dirs:")
        print(f"  Annotations: {ann_dir}")
        print(f"  Images: {img_dir}")
        return {}

    # Find matching annotation-image pairs
    ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.xml')])
    random.seed(seed)
    random.shuffle(ann_files)
    if n_samples > 0:
        ann_files = ann_files[:n_samples]

    print("=" * 70)
    print("CROSS-DATASET TRANSFER EVALUATION")
    print(f"PubTables-1M test set → {len(ann_files)} samples")
    print("=" * 70)

    device = get_device()
    print(f"\nDevice: {device}")

    print("Loading pre-trained TATR...")
    proc_pt, model_pt = load_model(device)
    print(f"Loading fine-tuned TATR from {finetuned_path}...")
    proc_ft, model_ft = load_model(device, model_path=finetuned_path)

    pt_detections = []
    ft_detections = []
    gt_all = []
    ious_pt = []
    ious_ft = []

    skipped = 0
    for ann_file in tqdm(ann_files, desc="Evaluating"):
        stem = ann_file.replace('.xml', '')
        img_path = os.path.join(img_dir, f"{stem}.jpg")
        ann_path = os.path.join(ann_dir, ann_file)

        if not os.path.exists(img_path):
            skipped += 1
            continue

        # Parse PASCAL VOC XML
        try:
            tree = ET.parse(ann_path)
            root = tree.getroot()
        except ET.ParseError:
            skipped += 1
            continue

        gt_tables = []
        for obj in root.findall("object"):
            bbox_el = obj.find("bndbox")
            x0 = float(bbox_el.find("xmin").text)
            y0 = float(bbox_el.find("ymin").text)
            x1 = float(bbox_el.find("xmax").text)
            y1 = float(bbox_el.find("ymax").text)
            gt_tables.append((x0, y0, x1, y1))

        if not gt_tables:
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        image_id = stem

        for gt_box in gt_tables:
            gt_all.append({"image_id": image_id, "bbox": gt_box})

        # Pre-trained
        dets_pt = detect_tables(image, proc_pt, model_pt, device, confidence)
        dets_pt = suppress_duplicates(dets_pt)
        for d in dets_pt:
            pt_detections.append({
                "image_id": image_id,
                "bbox": tuple(d["bbox"]),
                "score": d["score"],
            })
            best_iou = max(_iou(tuple(d["bbox"]), gt) for gt in gt_tables)
            ious_pt.append(best_iou)

        # Fine-tuned
        dets_ft = detect_tables(image, proc_ft, model_ft, device, confidence)
        dets_ft = suppress_duplicates(dets_ft)
        for d in dets_ft:
            ft_detections.append({
                "image_id": image_id,
                "bbox": tuple(d["bbox"]),
                "score": d["score"],
            })
            best_iou = max(_iou(tuple(d["bbox"]), gt) for gt in gt_tables)
            ious_ft.append(best_iou)

    print(f"\nProcessed {len(ann_files) - skipped} images (skipped {skipped})")
    print(f"  Pre-trained: {len(pt_detections)} detections")
    print(f"  Fine-tuned:  {len(ft_detections)} detections")
    print(f"  GT tables:   {len(gt_all)}")

    # Compute metrics
    ap50_pt = compute_ap_at_threshold(pt_detections, gt_all, 0.5)
    ap50_ft = compute_ap_at_threshold(ft_detections, gt_all, 0.5)
    ap75_pt = compute_ap_at_threshold(pt_detections, gt_all, 0.75)
    ap75_ft = compute_ap_at_threshold(ft_detections, gt_all, 0.75)

    mean_iou_pt = float(np.mean(ious_pt)) if ious_pt else 0.0
    mean_iou_ft = float(np.mean(ious_ft)) if ious_ft else 0.0

    recall_pt = recall_at_iou(pt_detections, gt_all, 0.5)
    recall_ft = recall_at_iou(ft_detections, gt_all, 0.5)

    results = {
        "pretrained": {
            "AP50": ap50_pt, "AP75": ap75_pt,
            "recall_50": recall_pt, "mean_iou": mean_iou_pt,
            "n_detections": len(pt_detections),
        },
        "finetuned_B": {
            "AP50": ap50_ft, "AP75": ap75_ft,
            "recall_50": recall_ft, "mean_iou": mean_iou_ft,
            "n_detections": len(ft_detections),
        },
        "n_images": len(ann_files) - skipped,
        "n_gt": len(gt_all),
    }

    print("\n" + "=" * 70)
    print("CROSS-DATASET TRANSFER RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Pre-trained':>15} {'Fine-tuned B':>15} {'Δ':>10}")
    print("-" * 65)
    print(f"{'AP@50':<25} {ap50_pt:>15.4f} {ap50_ft:>15.4f} {ap50_ft - ap50_pt:>+10.4f}")
    print(f"{'AP@75':<25} {ap75_pt:>15.4f} {ap75_ft:>15.4f} {ap75_ft - ap75_pt:>+10.4f}")
    print(f"{'Recall@50':<25} {recall_pt:>15.4f} {recall_ft:>15.4f} {recall_ft - recall_pt:>+10.4f}")
    print(f"{'Mean IoU':<25} {mean_iou_pt:>15.4f} {mean_iou_ft:>15.4f} {mean_iou_ft - mean_iou_pt:>+10.4f}")
    print(f"{'Detections':<25} {len(pt_detections):>15d} {len(ft_detections):>15d}")

    degradation = ap50_pt - ap50_ft
    if degradation > 0.05:
        print(f"\n⚠ Fine-tuning degraded AP@50 by {degradation:.3f} ({degradation*100:.1f}%)")
    elif degradation > 0:
        print(f"\n✅ Minor AP@50 degradation: {degradation:.3f} ({degradation*100:.1f}%)")
    else:
        print(f"\n✅ Fine-tuning IMPROVED AP@50 by {-degradation:.3f} ({-degradation*100:.1f}%)")

    if output_path:
        with open(output_path, "w") as fp:
            json.dump(results, fp, indent=2)
        print(f"\nSaved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset transfer evaluation")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--finetuned-path", default=DEFAULT_FINETUNED)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--output", default="cross_dataset_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate_on_pubtables(
        n_samples=args.n_samples,
        finetuned_path=args.finetuned_path,
        confidence=args.confidence,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
