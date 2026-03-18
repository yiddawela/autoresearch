"""
COCO-standard Average Precision (AP) evaluation for table detection.

Computes AP@50, AP@75, and AP@[50:95] following the COCO evaluation protocol:
  - For each IoU threshold, compute precision-recall curve
  - Interpolate precision at 101 recall points
  - AP = mean of interpolated precisions

This module is self-contained and does not depend on pycocotools,
which avoids a heavy dependency and gives us full control over the
evaluation protocol (important for the paper).

Usage:
    from coco_eval import compute_ap, compute_map
    ap50 = compute_ap(detections, ground_truths, iou_threshold=0.5)
"""

from __future__ import annotations

import numpy as np


def _iou(a: tuple, b: tuple) -> float:
    """IoU between two (x0, y0, x1, y1) boxes."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap(
    detections: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """
    Compute Average Precision at a single IoU threshold.

    Args:
        detections: list of {"image_id": str, "bbox": (x0,y0,x1,y1), "score": float}
        ground_truths: list of {"image_id": str, "bbox": (x0,y0,x1,y1)}
        iou_threshold: IoU threshold for matching

    Returns:
        dict with keys: ap, precision, recall, n_gt, n_det, tp, fp
    """
    if not ground_truths:
        return {"ap": 0.0, "precision": 0.0, "recall": 0.0,
                "n_gt": 0, "n_det": len(detections), "tp": 0, "fp": len(detections)}

    # Group GTs by image
    gt_by_image: dict[str, list[dict]] = {}
    for gt in ground_truths:
        img_id = gt["image_id"]
        gt_by_image.setdefault(img_id, []).append(gt)

    n_gt = len(ground_truths)

    # Track which GTs have been matched (per image)
    gt_matched = {
        img_id: [False] * len(gts) for img_id, gts in gt_by_image.items()
    }

    # Sort detections by score (descending)
    sorted_dets = sorted(detections, key=lambda d: d["score"], reverse=True)

    tp = np.zeros(len(sorted_dets))
    fp = np.zeros(len(sorted_dets))

    for det_idx, det in enumerate(sorted_dets):
        img_id = det["image_id"]
        det_bbox = tuple(det["bbox"])

        if img_id not in gt_by_image:
            fp[det_idx] = 1
            continue

        gts = gt_by_image[img_id]
        matched = gt_matched[img_id]

        # Find best matching GT
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gts):
            if matched[gt_idx]:
                continue  # Already matched
            iou = _iou(det_bbox, tuple(gt["bbox"]))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[det_idx] = 1
            matched[best_gt_idx] = True
        else:
            fp[det_idx] = 1

    # Compute cumulative TP/FP
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / n_gt
    precisions = cum_tp / (cum_tp + cum_fp)

    # 101-point interpolation (COCO standard)
    recall_points = np.linspace(0, 1, 101)
    interp_precisions = np.zeros(101)
    for i, r in enumerate(recall_points):
        # Precision at recall >= r
        mask = recalls >= r
        if mask.any():
            interp_precisions[i] = precisions[mask].max()

    ap = interp_precisions.mean()

    total_tp = int(cum_tp[-1]) if len(cum_tp) > 0 else 0
    total_fp = int(cum_fp[-1]) if len(cum_fp) > 0 else 0
    final_recall = float(recalls[-1]) if len(recalls) > 0 else 0.0
    final_precision = float(precisions[-1]) if len(precisions) > 0 else 0.0

    return {
        "ap": float(ap),
        "precision": final_precision,
        "recall": final_recall,
        "n_gt": n_gt,
        "n_det": len(sorted_dets),
        "tp": total_tp,
        "fp": total_fp,
    }


def compute_map(
    detections: list[dict],
    ground_truths: list[dict],
    iou_thresholds: list[float] | None = None,
) -> dict:
    """
    Compute mAP across multiple IoU thresholds (COCO-style).

    Default thresholds: [0.50, 0.55, ..., 0.95] (10 thresholds).

    Returns:
        dict with keys: mAP, AP50, AP75, per_threshold (dict of threshold -> AP)
    """
    if iou_thresholds is None:
        iou_thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]

    per_threshold = {}
    for t in iou_thresholds:
        result = compute_ap(detections, ground_truths, iou_threshold=t)
        per_threshold[t] = result["ap"]

    ap50 = per_threshold.get(0.5, 0.0)
    ap75 = per_threshold.get(0.75, 0.0)
    map_val = float(np.mean(list(per_threshold.values())))

    return {
        "mAP": map_val,
        "AP50": ap50,
        "AP75": ap75,
        "per_threshold": per_threshold,
        "n_gt": ground_truths[0]["image_id"] if ground_truths else 0,  # for debug
    }


def compute_iou_distribution(
    detections: list[dict],
    ground_truths: list[dict],
) -> dict:
    """
    Compute IoU distribution for matched detection-GT pairs.

    Returns dict with: ious (list), mean, median, std, histogram bins/counts.
    """
    # Group GTs by image
    gt_by_image: dict[str, list[dict]] = {}
    for gt in ground_truths:
        gt_by_image.setdefault(gt["image_id"], []).append(gt)

    ious = []
    for det in detections:
        img_id = det["image_id"]
        if img_id not in gt_by_image:
            continue
        for gt in gt_by_image[img_id]:
            iou = _iou(tuple(det["bbox"]), tuple(gt["bbox"]))
            if iou > 0.3:  # Only consider plausible matches
                ious.append(iou)

    if not ious:
        return {"ious": [], "mean": 0.0, "median": 0.0, "std": 0.0}

    ious_arr = np.array(ious)
    counts, bin_edges = np.histogram(ious_arr, bins=10, range=(0, 1))

    return {
        "ious": ious,
        "mean": float(ious_arr.mean()),
        "median": float(np.median(ious_arr)),
        "std": float(ious_arr.std()),
        "histogram": {
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
        },
    }
