"""
Evaluate baseline table cropping against SCI-3000 ground truth.

Parses SCI-3000 annotations (W3C format), runs our table detection pipeline,
and computes:
  - Table detection precision/recall (IoU ≥ 0.5)
  - Caption inclusion rate (what % of ground-truth captions overlap with crops)
  - Figure false positive rate (crops that overlap figures instead of tables)

Usage:
    uv run evaluate.py                           # full evaluation
    uv run evaluate.py --max-pages 100           # quick test on 100 pages
    uv run evaluate.py --data-dir /path/to/SCI-3000
"""

import _paths  # noqa: F401
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Import from our crop_tables module
from crop_tables import (
    detect_tables,
    get_device,
    load_model,
    refine_crop,
    suppress_duplicates,
)

DEFAULT_DATA_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "table_cropper", "SCI-3000"
)


# ---------------------------------------------------------------------------
# SCI-3000 annotation parsing
# ---------------------------------------------------------------------------


def parse_bbox_value(value: str) -> tuple[float, float, float, float]:
    """
    Parse W3C fragment selector value like 'xywh=pixel:194,309,851,445'
    Returns (x, y, w, h).
    """
    match = re.match(r"xywh=pixel:([\d.]+),([\d.]+),([\d.]+),([\d.]+)", value)
    if not match:
        raise ValueError(f"Cannot parse bbox: {value}")
    return tuple(float(match.group(i)) for i in range(1, 5))


def parse_page_annotations(json_path: str) -> dict:
    """
    Parse a SCI-3000 per-page annotation JSON file.

    Returns dict with:
      - canvas_size: (width, height)
      - tables: list of (x, y, w, h) bboxes
      - figures: list of (x, y, w, h) bboxes
      - captions: list of dict with keys: bbox, parent_id
      - annotations_by_id: dict mapping id -> annotation info
    """
    with open(json_path) as f:
        data = json.load(f)

    canvas_w = data.get("canvasWidth", 0)
    canvas_h = data.get("canvasHeight", 0)

    tables = []
    figures = []
    captions = []
    annotations_by_id = {}

    for ann in data.get("annotations", []):
        ann_id = ann.get("id", "")
        bodies = ann.get("body", [])

        # Determine type
        ann_type = None
        parent_id = None
        for body in bodies:
            if body.get("purpose") == "img-cap-enum":
                ann_type = body.get("value")  # "Table", "Figure", "Caption"
            elif body.get("purpose") == "parent":
                parent_id = body.get("value")

        # Parse bbox
        try:
            selector_value = ann["target"]["selector"]["value"]
            bbox = parse_bbox_value(selector_value)
        except (KeyError, ValueError):
            continue

        info = {"id": ann_id, "type": ann_type, "bbox": bbox, "parent_id": parent_id}
        annotations_by_id[ann_id] = info

        if ann_type == "Table":
            tables.append(info)
        elif ann_type == "Figure":
            figures.append(info)
        elif ann_type == "Caption":
            captions.append(info)

    return {
        "canvas_size": (canvas_w, canvas_h),
        "tables": tables,
        "figures": figures,
        "captions": captions,
        "annotations_by_id": annotations_by_id,
    }


def get_table_with_captions_bbox(
    table_info: dict,
    captions: list[dict],
    canvas_w: int,
    canvas_h: int,
) -> tuple[float, float, float, float]:
    """
    Compute the merged bounding box of a table + its linked captions.

    Returns (x0, y0, x1, y1) in pixel coordinates.
    """
    tx, ty, tw, th = table_info["bbox"]
    x0, y0, x1, y1 = tx, ty, tx + tw, ty + th

    table_id = table_info["id"]
    for cap in captions:
        if cap["parent_id"] == table_id:
            cx, cy, cw, ch = cap["bbox"]
            x0 = min(x0, cx)
            y0 = min(y0, cy)
            x1 = max(x1, cx + cw)
            y1 = max(y1, cy + ch)

    # Clamp to canvas
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(canvas_w, x1)
    y1 = min(canvas_h, y1)

    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------


def compute_iou(
    box_a: tuple[float, ...], box_b: tuple[float, ...]
) -> float:
    """Compute IoU between two (x0, y0, x1, y1) boxes."""
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])

    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def compute_overlap_ratio(
    inner: tuple[float, ...], outer: tuple[float, ...]
) -> float:
    """What fraction of 'inner' is covered by 'outer'?"""
    x0 = max(inner[0], outer[0])
    y0 = max(inner[1], outer[1])
    x1 = min(inner[2], outer[2])
    y1 = min(inner[3], outer[3])

    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_inner = (inner[2] - inner[0]) * (inner[3] - inner[1])

    return inter / area_inner if area_inner > 0 else 0.0


# ---------------------------------------------------------------------------
# Scale conversion
# ---------------------------------------------------------------------------


def scale_bbox(
    bbox: tuple[float, ...],
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Scale a bbox from src_size to dst_size coordinate space."""
    sw, sh = src_size
    dw, dh = dst_size
    scale_x = dw / sw if sw > 0 else 1.0
    scale_y = dh / sh if sh > 0 else 1.0
    return (
        bbox[0] * scale_x,
        bbox[1] * scale_y,
        bbox[2] * scale_x,
        bbox[3] * scale_y,
    )


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate(
    data_dir: str,
    max_pages: int | None = None,
    dpi: int = 200,
    confidence: float = 0.5,
    iou_threshold: float = 0.5,
):
    """Run evaluation against SCI-3000."""
    from pdf2image import convert_from_path

    annot_dir = os.path.join(data_dir, "Annotations")
    pdf_dir = os.path.join(data_dir, "PDFs")

    if not os.path.isdir(annot_dir):
        print(f"Error: {annot_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Load model
    print("Loading Table Transformer model...")
    device = get_device()
    print(f"  Device: {device}")
    processor, model = load_model(device)
    print("  Model loaded ✓")

    # Collect annotation files that have tables
    print("\nScanning annotations for pages with tables...")
    annot_files = sorted(Path(annot_dir).glob("*.json"))

    pages_with_tables = []
    for af in annot_files:
        parsed = parse_page_annotations(str(af))
        if parsed["tables"]:
            pages_with_tables.append((af, parsed))

    print(f"  Found {len(pages_with_tables)} pages with table annotations")

    if max_pages:
        pages_with_tables = pages_with_tables[:max_pages]
        print(f"  Limiting to {max_pages} pages")

    # Metrics accumulators
    total_gt_tables = 0  # ground truth tables (table+caption merged)
    total_detected = 0   # our detections
    true_positives = 0   # correct detections (IoU >= threshold)
    false_positives = 0  # detections not matching any GT
    false_negatives = 0  # GT tables we missed

    total_gt_captions = 0   # captions linked to tables
    captions_included = 0   # captions overlapping with our crops

    figure_false_pos = 0  # detections that overlap figures

    # Track per-page results
    page_results = []

    # Group pages by PDF to avoid re-rendering
    pdf_pages = {}
    for af, parsed in pages_with_tables:
        # filename format: {pdf_id}-{page_num}.json
        basename = af.stem  # e.g. "000361855c304169bdf874a7be8ff192-03"
        parts = basename.rsplit("-", 1)
        if len(parts) != 2:
            continue
        pdf_id, page_str = parts
        page_num = int(page_str)
        pdf_path = os.path.join(pdf_dir, f"{pdf_id}.pdf")

        if not os.path.exists(pdf_path):
            continue

        if pdf_id not in pdf_pages:
            pdf_pages[pdf_id] = []
        pdf_pages[pdf_id].append((page_num, parsed, str(af)))

    print(f"  Spanning {len(pdf_pages)} unique PDFs")
    print()

    # Process each PDF
    for pdf_id, page_list in tqdm(
        pdf_pages.items(), desc="Evaluating", unit="PDF"
    ):
        pdf_path = os.path.join(pdf_dir, f"{pdf_id}.pdf")

        # Get the max page number needed
        max_page = max(pn for pn, _, _ in page_list)

        try:
            # Render only needed pages
            pages = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=1,
                last_page=max_page,
            )
        except Exception as e:
            tqdm.write(f"  ⚠ Failed to render {pdf_id}: {e}")
            continue

        for page_num, parsed, annot_path in page_list:
            if page_num > len(pages):
                continue

            page_image = pages[page_num - 1]  # 0-indexed
            img_w, img_h = page_image.size
            canvas_size = parsed["canvas_size"]

            # --- Ground truth ---
            gt_table_bboxes = []
            gt_caption_bboxes = []

            for table_info in parsed["tables"]:
                # Merged table+caption bbox in canvas coordinates
                merged_bbox = get_table_with_captions_bbox(
                    table_info, parsed["captions"],
                    canvas_size[0], canvas_size[1],
                )
                # Scale to image coordinates
                gt_bbox_img = scale_bbox(merged_bbox, canvas_size, (img_w, img_h))
                gt_table_bboxes.append(gt_bbox_img)

                # Track individual captions
                for cap in parsed["captions"]:
                    if cap["parent_id"] == table_info["id"]:
                        cx, cy, cw, ch = cap["bbox"]
                        cap_bbox = (cx, cy, cx + cw, cy + ch)
                        cap_bbox_img = scale_bbox(cap_bbox, canvas_size, (img_w, img_h))
                        gt_caption_bboxes.append(cap_bbox_img)

            # Ground truth figures (for false positive check)
            gt_figure_bboxes = []
            for fig_info in parsed["figures"]:
                fx, fy, fw, fh = fig_info["bbox"]
                fig_bbox = (fx, fy, fx + fw, fy + fh)
                fig_bbox_img = scale_bbox(fig_bbox, canvas_size, (img_w, img_h))
                gt_figure_bboxes.append(fig_bbox_img)

            total_gt_tables += len(gt_table_bboxes)
            total_gt_captions += len(gt_caption_bboxes)

            # --- Our detections ---
            detections = detect_tables(
                page_image, processor, model, device, confidence
            )
            detections = suppress_duplicates(detections)
            total_detected += len(detections)

            # Refine crops
            refined_bboxes = []
            for det in detections:
                refined = refine_crop(page_image, det["bbox"])
                refined_bboxes.append(refined)

            # --- Match detections to ground truth ---
            gt_matched = [False] * len(gt_table_bboxes)

            for det_bbox in refined_bboxes:
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt_bbox in enumerate(gt_table_bboxes):
                    iou = compute_iou(det_bbox, gt_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                    true_positives += 1
                    gt_matched[best_gt_idx] = True
                else:
                    false_positives += 1

                    # Check if it overlaps a figure
                    for fig_bbox in gt_figure_bboxes:
                        if compute_iou(det_bbox, fig_bbox) >= 0.3:
                            figure_false_pos += 1
                            break

            false_negatives += sum(1 for m in gt_matched if not m)

            # --- Caption inclusion check ---
            for cap_bbox in gt_caption_bboxes:
                cap_included = False
                for det_bbox in refined_bboxes:
                    overlap = compute_overlap_ratio(cap_bbox, det_bbox)
                    if overlap >= 0.5:
                        cap_included = True
                        break
                if cap_included:
                    captions_included += 1

    # --- Report ---
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    precision = true_positives / total_detected if total_detected > 0 else 0
    recall = true_positives / total_gt_tables if total_gt_tables > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    caption_rate = (
        captions_included / total_gt_captions if total_gt_captions > 0 else 0
    )

    print(f"\n  Pages evaluated:         {len(pages_with_tables)}")
    print(f"  Ground truth tables:     {total_gt_tables}")
    print(f"  Our detections:          {total_detected}")
    print()
    print(f"  True positives (IoU≥{iou_threshold}):  {true_positives}")
    print(f"  False positives:         {false_positives}")
    print(f"  False negatives:         {false_negatives}")
    print()
    print(f"  Precision:               {precision:.3f}")
    print(f"  Recall:                  {recall:.3f}")
    print(f"  F1 Score:                {f1:.3f}")
    print()
    print(f"  GT captions (linked):    {total_gt_captions}")
    print(f"  Captions included:       {captions_included}")
    print(f"  Caption inclusion rate:  {caption_rate:.3f}")
    print()
    print(f"  Figure false positives:  {figure_false_pos}")
    print("=" * 60)

    return {
        "pages_evaluated": len(pages_with_tables),
        "gt_tables": total_gt_tables,
        "detections": total_detected,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gt_captions": total_gt_captions,
        "captions_included": captions_included,
        "caption_inclusion_rate": caption_rate,
        "figure_false_positives": figure_false_pos,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate table cropping against SCI-3000"
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Path to SCI-3000 directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit evaluation to N pages (for quick testing)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF rendering (default: 200)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching detections to GT (default: 0.5)",
    )
    args = parser.parse_args()

    evaluate(
        data_dir=args.data_dir,
        max_pages=args.max_pages,
        dpi=args.dpi,
        confidence=args.confidence,
        iou_threshold=args.iou_threshold,
    )


if __name__ == "__main__":
    main()
