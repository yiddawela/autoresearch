"""
WP0 Pilot Validation: Compare three table extraction formulations on 50 pages.

Selects 50 diverse pages from SCI-3000 (mix of single/multi-column, caption
above/below, with/without footnotes) and evaluates:
  - Formulation A: Table-only detection (pre-trained TATR, no refinement)
  - Formulation B: Merged semantically complete crop (fine-tuned TATR)
  - Formulation C: Table-only detection + heuristic caption expansion

Computes semantic completeness metrics (CIR, SCS, CUCR, over-crop ratio)
and outputs a comparison table.

Usage:
    uv run pilot_evaluation.py
    uv run pilot_evaluation.py --n-pages 50 --dpi 200
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from completeness_metrics import compute_all_metrics
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
DEFAULT_FINETUNED = os.path.join(
    os.path.expanduser("~"), ".cache", "table_cropper", "checkpoints", "round01_best"
)


# ---------------------------------------------------------------------------
# SCI-3000 annotation parsing
# ---------------------------------------------------------------------------


def parse_bbox_value(value: str) -> tuple[float, float, float, float]:
    """Parse 'xywh=pixel:x,y,w,h' → (x, y, w, h)."""
    match = re.match(r"xywh=pixel:([\d.]+),([\d.]+),([\d.]+),([\d.]+)", value)
    if not match:
        raise ValueError(f"Cannot parse bbox: {value}")
    return tuple(float(match.group(i)) for i in range(1, 5))


def parse_page_annotations(json_path: str) -> dict:
    """Parse SCI-3000 per-page annotation JSON.

    Returns dict with tables, figures, captions (each with id, bbox, parent_id).
    """
    with open(json_path) as f:
        data = json.load(f)

    canvas_w = data.get("canvasWidth", 0)
    canvas_h = data.get("canvasHeight", 0)

    tables = []
    figures = []
    captions = []

    for ann in data.get("annotations", []):
        ann_id = ann.get("id", "")
        bodies = ann.get("body", [])

        ann_type = None
        parent_id = None
        for body in bodies:
            if body.get("purpose") == "img-cap-enum":
                ann_type = body.get("value")
            elif body.get("purpose") == "parent":
                parent_id = body.get("value")

        try:
            selector_value = ann["target"]["selector"]["value"]
            bbox = parse_bbox_value(selector_value)
        except (KeyError, ValueError):
            continue

        info = {"id": ann_id, "type": ann_type, "bbox": bbox, "parent_id": parent_id}

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
    }


def get_table_components(
    table_info: dict,
    captions: list[dict],
    canvas_w: int,
    canvas_h: int,
    img_w: int,
    img_h: int,
) -> dict:
    """Get separate table body and caption boxes in image coordinates.

    Returns dict with:
        table_box: (x0, y0, x1, y1) in image coords
        caption_boxes: list of (x0, y0, x1, y1) in image coords
        merged_box: merged table+caption bounding box
        caption_position: 'above', 'below', or 'none'
    """
    tx, ty, tw, th = table_info["bbox"]
    scale_x = img_w / canvas_w if canvas_w > 0 else 1.0
    scale_y = img_h / canvas_h if canvas_h > 0 else 1.0

    table_box = (
        tx * scale_x,
        ty * scale_y,
        (tx + tw) * scale_x,
        (ty + th) * scale_y,
    )

    # Find linked captions
    table_id = table_info["id"]
    caption_boxes = []
    caption_positions = []
    for cap in captions:
        if cap["parent_id"] == table_id:
            cx, cy, cw, ch = cap["bbox"]
            cap_box = (
                cx * scale_x,
                cy * scale_y,
                (cx + cw) * scale_x,
                (cy + ch) * scale_y,
            )
            caption_boxes.append(cap_box)
            # Determine position relative to table
            cap_center_y = (cap_box[1] + cap_box[3]) / 2
            table_center_y = (table_box[1] + table_box[3]) / 2
            caption_positions.append("above" if cap_center_y < table_center_y else "below")

    # Merged box
    if caption_boxes:
        all_boxes = [table_box] + caption_boxes
        merged_box = (
            min(b[0] for b in all_boxes),
            min(b[1] for b in all_boxes),
            max(b[2] for b in all_boxes),
            max(b[3] for b in all_boxes),
        )
        caption_position = caption_positions[0] if len(set(caption_positions)) == 1 else "mixed"
    else:
        merged_box = table_box
        caption_position = "none"

    return {
        "table_box": table_box,
        "caption_boxes": caption_boxes,
        "merged_box": merged_box,
        "caption_position": caption_position,
    }


# ---------------------------------------------------------------------------
# Page selection: diverse 50-page sample
# ---------------------------------------------------------------------------


def select_diverse_pages(
    data_dir: str,
    n_pages: int = 50,
    seed: int = 42,
) -> list[tuple[str, dict]]:
    """Select diverse pages with tables from SCI-3000.

    Uses the pre-rendered val split as the candidate pool (~1,106 pages)
    to avoid scanning all 34K+ annotations.

    Ensures mix of:
    - Pages with 1 vs multiple tables
    - Caption above vs below
    - Pages with and without figures (for false positive risk)
    """
    annot_dir = os.path.join(data_dir, "Annotations")

    # Use val split page IDs as candidate pool (much faster than scanning all)
    val_img_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "table_cropper", "finetune_data", "val", "images"
    )
    if os.path.isdir(val_img_dir):
        val_page_ids = [Path(f).stem for f in os.listdir(val_img_dir) if f.endswith(".png")]
        annot_files = [Path(annot_dir) / f"{pid}.json" for pid in val_page_ids]
        annot_files = [f for f in annot_files if f.exists()]
        print(f"  Using val split: {len(annot_files)} pre-rendered pages")
    else:
        # Fallback: scan all annotations (slow)
        annot_files = sorted(Path(annot_dir).glob("*.json"))
        print(f"  Scanning all annotations: {len(annot_files)} files")

    candidates = []
    for af in tqdm(annot_files, desc="Scanning annotations", leave=False):
        parsed = parse_page_annotations(str(af))
        if not parsed["tables"]:
            continue

        # Check for linked captions
        table_ids = {t["id"] for t in parsed["tables"]}
        linked_captions = [c for c in parsed["captions"] if c["parent_id"] in table_ids]

        if not linked_captions:
            continue  # Need captions for completeness metrics

        # Classify page
        n_tables = len(parsed["tables"])
        has_figures = len(parsed["figures"]) > 0

        # Determine caption positions
        positions = set()
        for table in parsed["tables"]:
            for cap in linked_captions:
                if cap["parent_id"] == table["id"]:
                    ty = table["bbox"][1]
                    cy = cap["bbox"][1]
                    positions.add("above" if cy < ty else "below")

        candidates.append({
            "path": str(af),
            "parsed": parsed,
            "n_tables": n_tables,
            "has_figures": has_figures,
            "caption_positions": positions,
        })

    print(f"  Found {len(candidates)} pages with tables + linked captions")

    # Stratified sampling
    rng = random.Random(seed)

    # Buckets
    buckets = {
        "single_table_cap_above": [],
        "single_table_cap_below": [],
        "multi_table": [],
        "with_figures": [],
        "other": [],
    }

    for c in candidates:
        if c["n_tables"] > 1:
            buckets["multi_table"].append(c)
        elif c["has_figures"]:
            buckets["with_figures"].append(c)
        elif "above" in c["caption_positions"] and "below" not in c["caption_positions"]:
            buckets["single_table_cap_above"].append(c)
        elif "below" in c["caption_positions"]:
            buckets["single_table_cap_below"].append(c)
        else:
            buckets["other"].append(c)

    print("  Page diversity buckets:")
    for k, v in buckets.items():
        print(f"    {k}: {len(v)}")

    # Sample proportionally, minimum 5 per non-empty bucket
    selected = []
    non_empty = {k: v for k, v in buckets.items() if v}
    per_bucket = max(5, n_pages // len(non_empty))

    for k, pages in non_empty.items():
        rng.shuffle(pages)
        n = min(per_bucket, len(pages))
        selected.extend(pages[:n])

    # If we have too many, trim; if too few, add more from largest bucket
    if len(selected) > n_pages:
        rng.shuffle(selected)
        selected = selected[:n_pages]
    elif len(selected) < n_pages:
        remaining = [c for c in candidates if c not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[:n_pages - len(selected)])

    print(f"  Selected {len(selected)} pages for pilot")
    return [(s["path"], s["parsed"]) for s in selected]


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _match_detection_to_gt(
    det_box: tuple,
    gt_components: list[dict],
    iou_threshold: float = 0.3,
) -> int | None:
    """Match a detection to the best GT table by IoU. Returns GT index or None."""
    best_iou = 0.0
    best_idx = None
    for i, gt in enumerate(gt_components):
        iou = _compute_iou(det_box, gt["merged_box"])
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    if best_iou >= iou_threshold and best_idx is not None:
        return best_idx
    return None


def _compute_iou(a: tuple, b: tuple) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Main pilot evaluation
# ---------------------------------------------------------------------------


def run_pilot(
    data_dir: str,
    finetuned_path: str | None,
    n_pages: int = 50,
    dpi: int = 200,
    confidence: float = 0.5,
    seed: int = 42,
):
    """Run the WP0 pilot evaluation."""
    from pdf2image import convert_from_path

    print("=" * 70)
    print("WP0 PILOT VALIDATION")
    print("Comparing three formulations on semantic completeness metrics")
    print("=" * 70)
    print()

    # Select pages
    print("Step 1: Selecting diverse pages...")
    pages = select_diverse_pages(data_dir, n_pages, seed)
    print()

    # Load models
    print("Step 2: Loading models...")
    device = get_device()
    print(f"  Device: {device}")

    # Pre-trained model (for Formulation A and C)
    print("  Loading pre-trained TATR...")
    proc_pretrained, model_pretrained = load_model(device)
    print("  Pre-trained model loaded ✓")

    # Fine-tuned model (for Formulation B)
    if finetuned_path and os.path.isdir(finetuned_path):
        print(f"  Loading fine-tuned model from {finetuned_path}...")
        proc_finetuned, model_finetuned = load_model(device, model_path=finetuned_path)
        print("  Fine-tuned model loaded ✓")
    else:
        print("  ⚠ No fine-tuned model found, using pre-trained for Formulation B too")
        proc_finetuned, model_finetuned = proc_pretrained, model_pretrained
    print()

    # Group pages by PDF for efficient rendering
    pdf_pages = {}
    for annot_path, parsed in pages:
        basename = Path(annot_path).stem
        parts = basename.rsplit("-", 1)
        if len(parts) != 2:
            continue
        pdf_id, page_str = parts
        page_num = int(page_str)
        pdf_path = os.path.join(data_dir, "PDFs", f"{pdf_id}.pdf")
        if not os.path.exists(pdf_path):
            continue
        if pdf_id not in pdf_pages:
            pdf_pages[pdf_id] = []
        pdf_pages[pdf_id].append((page_num, parsed, annot_path))

    # Collect metrics per formulation
    formulation_metrics = {
        "A_table_only": [],
        "B_merged_finetuned": [],
        "C_heuristic_expansion": [],
    }

    # Track divergence examples
    divergence_examples = []

    print("Step 3: Running evaluation...")
    total_tables = 0
    matched_tables = {"A": 0, "B": 0, "C": 0}

    for pdf_id, page_list in tqdm(pdf_pages.items(), desc="Processing PDFs"):
        pdf_path = os.path.join(data_dir, "PDFs", f"{pdf_id}.pdf")
        max_page = max(pn for pn, _, _ in page_list)

        try:
            rendered = convert_from_path(
                pdf_path, dpi=dpi, first_page=1, last_page=max_page
            )
        except Exception as e:
            tqdm.write(f"  ⚠ Failed to render {pdf_id}: {e}")
            continue

        for page_num, parsed, annot_path in page_list:
            if page_num > len(rendered):
                continue

            page_image = rendered[page_num - 1]
            img_w, img_h = page_image.size
            canvas_w, canvas_h = parsed["canvas_size"]

            # Build ground-truth components for each table
            gt_components = []
            for table_info in parsed["tables"]:
                comp = get_table_components(
                    table_info, parsed["captions"],
                    canvas_w, canvas_h, img_w, img_h
                )
                if comp["caption_boxes"]:  # Only include tables with captions
                    gt_components.append(comp)

            if not gt_components:
                continue

            total_tables += len(gt_components)

            # --- Formulation A: Table-only (pre-trained, raw detection) ---
            dets_a = detect_tables(
                page_image, proc_pretrained, model_pretrained, device, confidence
            )
            dets_a = suppress_duplicates(dets_a)

            for gt_idx, gt in enumerate(gt_components):
                # Match detections to this GT
                best_det_a = None
                best_iou_a = 0
                for det in dets_a:
                    iou = _compute_iou(
                        (det["bbox"][0], det["bbox"][1], det["bbox"][2], det["bbox"][3]),
                        gt["table_box"]
                    )
                    if iou > best_iou_a:
                        best_iou_a = iou
                        best_det_a = det

                if best_det_a and best_iou_a >= 0.3:
                    pred_box_a = tuple(best_det_a["bbox"])
                    metrics_a = compute_all_metrics(
                        gt["table_box"], gt["caption_boxes"], pred_box_a
                    )
                    formulation_metrics["A_table_only"].append(metrics_a)
                    matched_tables["A"] += 1
                else:
                    # Missed table
                    formulation_metrics["A_table_only"].append({
                        "cir": 0, "scs": 0, "cucr": False, "over_crop": 0,
                        "table_iou": 0, "table_coverage": 0, "caption_coverage": 0,
                    })

            # --- Formulation B: Merged crop (fine-tuned model) ---
            dets_b = detect_tables(
                page_image, proc_finetuned, model_finetuned, device, confidence
            )
            dets_b = suppress_duplicates(dets_b)

            for gt_idx, gt in enumerate(gt_components):
                best_det_b = None
                best_iou_b = 0
                for det in dets_b:
                    iou = _compute_iou(
                        (det["bbox"][0], det["bbox"][1], det["bbox"][2], det["bbox"][3]),
                        gt["merged_box"]
                    )
                    if iou > best_iou_b:
                        best_iou_b = iou
                        best_det_b = det

                if best_det_b and best_iou_b >= 0.3:
                    pred_box_b = tuple(best_det_b["bbox"])
                    metrics_b = compute_all_metrics(
                        gt["table_box"], gt["caption_boxes"], pred_box_b
                    )
                    formulation_metrics["B_merged_finetuned"].append(metrics_b)
                    matched_tables["B"] += 1
                else:
                    formulation_metrics["B_merged_finetuned"].append({
                        "cir": 0, "scs": 0, "cucr": False, "over_crop": 0,
                        "table_iou": 0, "table_coverage": 0, "caption_coverage": 0,
                    })

            # --- Formulation C: Table detection + heuristic expansion ---
            # Re-use pre-trained detections, apply heuristic refinement
            for gt_idx, gt in enumerate(gt_components):
                best_det_c = None
                best_iou_c = 0
                for det in dets_a:  # Same detections as A
                    iou = _compute_iou(
                        (det["bbox"][0], det["bbox"][1], det["bbox"][2], det["bbox"][3]),
                        gt["table_box"]
                    )
                    if iou > best_iou_c:
                        best_iou_c = iou
                        best_det_c = det

                if best_det_c and best_iou_c >= 0.3:
                    # Apply heuristic refinement
                    refined = refine_crop(page_image, best_det_c["bbox"])
                    pred_box_c = refined
                    metrics_c = compute_all_metrics(
                        gt["table_box"], gt["caption_boxes"], pred_box_c
                    )
                    formulation_metrics["C_heuristic_expansion"].append(metrics_c)
                    matched_tables["C"] += 1

                    # Track divergence with formulation A
                    if len(formulation_metrics["A_table_only"]) > 0:
                        metrics_a_for_this = formulation_metrics["A_table_only"][
                            -(len(gt_components) - gt_idx)
                        ]
                        cir_diff = metrics_c["cir"] - metrics_a_for_this["cir"]
                        if abs(cir_diff) > 0.1:
                            divergence_examples.append({
                                "page": Path(annot_path).stem,
                                "gt_idx": gt_idx,
                                "caption_position": gt.get("caption_position", "unknown"),
                                "a_cir": metrics_a_for_this["cir"],
                                "c_cir": metrics_c["cir"],
                                "a_scs": metrics_a_for_this["scs"],
                                "c_scs": metrics_c["scs"],
                                "diff": cir_diff,
                            })
                else:
                    formulation_metrics["C_heuristic_expansion"].append({
                        "cir": 0, "scs": 0, "cucr": False, "over_crop": 0,
                        "table_iou": 0, "table_coverage": 0, "caption_coverage": 0,
                    })

    # --- Report ---
    print()
    print("=" * 70)
    print("WP0 PILOT RESULTS")
    print("=" * 70)
    print(f"\nTotal GT tables (with captions): {total_tables}")
    print(f"Matched detections: A={matched_tables['A']}, B={matched_tables['B']}, C={matched_tables['C']}")
    print()

    # Compute aggregate metrics
    print(f"{'Metric':<25} {'A: Table-only':>15} {'B: Merged (FT)':>15} {'C: Heuristic':>15}")
    print("-" * 70)

    metric_keys = [
        ("cir", "Caption Incl. Rate"),
        ("scs", "Semantic Coverage"),
        ("cucr", "Complete Unit Cap."),
        ("over_crop", "Over-crop Ratio"),
        ("table_iou", "Table IoU"),
        ("table_coverage", "Table Coverage"),
        ("caption_coverage", "Caption Coverage"),
    ]

    results = {}
    for key, label in metric_keys:
        row = []
        for form_name in ["A_table_only", "B_merged_finetuned", "C_heuristic_expansion"]:
            values = formulation_metrics[form_name]
            if not values:
                row.append("N/A")
                continue
            if key == "cucr":
                val = sum(1 for v in values if v[key]) / len(values)
            else:
                val = np.mean([v[key] for v in values])
            row.append(f"{val:.4f}")
            results.setdefault(form_name, {})[key] = val

        print(f"{label:<25} {row[0]:>15} {row[1]:>15} {row[2]:>15}")

    print()
    print("=" * 70)

    # Key differences
    if results:
        print("\nKEY DIFFERENCES (C vs A):")
        for key, label in metric_keys:
            a_val = results.get("A_table_only", {}).get(key, 0)
            c_val = results.get("C_heuristic_expansion", {}).get(key, 0)
            diff = c_val - a_val
            direction = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"  {label}: {direction} {abs(diff):.4f}")

        print("\nKEY DIFFERENCES (B vs A):")
        for key, label in metric_keys:
            a_val = results.get("A_table_only", {}).get(key, 0)
            b_val = results.get("B_merged_finetuned", {}).get(key, 0)
            diff = b_val - a_val
            direction = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"  {label}: {direction} {abs(diff):.4f}")

    # Divergence examples
    print(f"\nDIVERGENCE EXAMPLES (|CIR diff| > 0.1 between A and C): {len(divergence_examples)}")
    for ex in sorted(divergence_examples, key=lambda x: -abs(x["diff"]))[:15]:
        print(f"  {ex['page']} (table {ex['gt_idx']}, caption {ex['caption_position']}): "
              f"A_CIR={ex['a_cir']:.3f} → C_CIR={ex['c_cir']:.3f} (Δ={ex['diff']:+.3f})")

    # Pilot gate assessment
    print()
    print("=" * 70)
    print("PILOT GATE ASSESSMENT")
    print("=" * 70)

    a_cir = results.get("A_table_only", {}).get("cir", 0)
    b_cir = results.get("B_merged_finetuned", {}).get("cir", 0)
    c_cir = results.get("C_heuristic_expansion", {}).get("cir", 0)

    max_cir_diff = max(abs(b_cir - a_cir), abs(c_cir - a_cir))
    n_divergences = len(divergence_examples)

    print(f"  Max CIR difference from baseline: {max_cir_diff:.4f}")
    print(f"  Divergence examples found: {n_divergences}")

    if max_cir_diff > 0.05 and n_divergences >= 5:
        print("  ✅ Formulation differences are MEASURABLE. Proceed to full experiments.")
    elif max_cir_diff > 0.02 or n_divergences >= 3:
        print("  ⚠ Formulation differences are MODEST. Review qualitatively before proceeding.")
    else:
        print("  ❌ Formulation differences are TRIVIAL. Reassess task definition and metrics.")

    return results


def main():
    parser = argparse.ArgumentParser(description="WP0 Pilot Validation")
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help="Path to SCI-3000 dataset",
    )
    parser.add_argument(
        "--finetuned-path", default=DEFAULT_FINETUNED,
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--n-pages", type=int, default=50,
        help="Number of pages to evaluate (default: 50)",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="DPI for PDF rendering (default: 200)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for page selection (default: 42)",
    )
    args = parser.parse_args()

    run_pilot(
        data_dir=args.data_dir,
        finetuned_path=args.finetuned_path,
        n_pages=args.n_pages,
        dpi=args.dpi,
        confidence=args.confidence,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
