"""
Metric robustness analysis for paper strengthening.

Computes:
  1. CUCR at multiple IoU thresholds (0.4, 0.5, 0.6, 0.7, 0.8)
  2. SCS under alternative weighting schemes
  3. Over-crop vs SCS scatter data for diagnostic figure

Re-uses the experiment runner infrastructure to collect per-table metrics.

Usage:
    uv run robustness_analysis.py
    uv run robustness_analysis.py --output ../results/robustness_analysis.json
"""

import _paths  # noqa: F401
import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from completeness_metrics import (
    caption_inclusion_rate,
    complete_unit_capture_rate,
    over_crop_ratio,
    semantic_coverage_score,
    _iou,
    _coverage,
)
from experiment_runner import (
    FormulationResult,
    build_gt_components,
    load_val_pages,
    _iou as exp_iou,
)
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
    os.path.expanduser("~"), ".cache", "table_cropper", "checkpoints",
    "phase5_resumed_final"
)

CUCR_THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]

SCS_WEIGHT_SCHEMES = {
    "equal": (1.0, 1.0, 1.0),
    "body_heavy": (2.0, 1.0, 1.0),
    "caption_heavy": (1.0, 2.0, 1.0),
}


def collect_per_table_data(
    data_dir: str = DEFAULT_DATA_DIR,
    finetuned_path: str = DEFAULT_FINETUNED,
    confidence: float = 0.5,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Run all three formulations and collect per-table completeness components.

    Returns dict mapping formulation -> list of per-table records with:
      - table_box, caption_boxes, pred_box (raw boxes for recomputation)
      - cir, scs, over_crop, table_iou, table_coverage, caption_coverage
    """
    from pdf2image import convert_from_path

    print("Loading pages...")
    pages = load_val_pages(data_dir, seed=seed)
    print(f"  {len(pages)} pages loaded")

    device = get_device()
    print(f"  Device: {device}")
    print("  Loading pre-trained TATR...")
    proc_pt, model_pt = load_model(device)
    print(f"  Loading fine-tuned model from {finetuned_path}...")
    proc_ft, model_ft = load_model(device, model_path=finetuned_path)

    # Group by PDF
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
        pdf_pages.setdefault(pdf_id, []).append((page_num, parsed, annot_path))

    per_table = {"A": [], "B": [], "C": []}

    print(f"\nProcessing {len(pdf_pages)} PDFs...")
    for pdf_id, page_list in tqdm(pdf_pages.items(), desc="PDFs"):
        pdf_path = os.path.join(data_dir, "PDFs", f"{pdf_id}.pdf")
        max_page = max(pn for pn, _, _ in page_list)

        try:
            rendered = convert_from_path(
                pdf_path, dpi=200, first_page=1, last_page=max_page
            )
        except Exception:
            continue

        for page_num, parsed, annot_path in page_list:
            if page_num > len(rendered):
                continue

            image = rendered[page_num - 1]
            img_w, img_h = image.size
            image_id = Path(annot_path).stem

            gt_components = build_gt_components(parsed, img_w, img_h)
            gt_with_captions = [g for g in gt_components if g["caption_boxes"]]
            if not gt_with_captions:
                continue

            # Formulation A
            dets_a = detect_tables(image, proc_pt, model_pt, device, confidence)
            dets_a = suppress_duplicates(dets_a)
            for det in dets_a:
                bbox = tuple(det["bbox"])
                best_iou, best_gt = 0, None
                for gt in gt_with_captions:
                    iou = exp_iou(bbox, gt["table_box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt
                if best_gt and best_iou >= 0.3:
                    per_table["A"].append({
                        "image_id": image_id,
                        "table_box": list(best_gt["table_box"]),
                        "caption_boxes": [list(c) for c in best_gt["caption_boxes"]],
                        "pred_box": list(bbox),
                    })

            # Formulation B
            dets_b = detect_tables(image, proc_ft, model_ft, device, confidence)
            dets_b = suppress_duplicates(dets_b)
            for det in dets_b:
                bbox = tuple(det["bbox"])
                best_iou, best_gt = 0, None
                for gt in gt_with_captions:
                    iou = exp_iou(bbox, gt["merged_box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt
                if best_gt and best_iou >= 0.3:
                    per_table["B"].append({
                        "image_id": image_id,
                        "table_box": list(best_gt["table_box"]),
                        "caption_boxes": [list(c) for c in best_gt["caption_boxes"]],
                        "pred_box": list(bbox),
                    })

            # Formulation C
            for det in dets_a:  # reuse pretrained detections
                raw_bbox = tuple(det["bbox"])
                refined = refine_crop(image, det["bbox"])
                best_iou, best_gt = 0, None
                for gt in gt_with_captions:
                    iou = exp_iou(raw_bbox, gt["table_box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt
                if best_gt and best_iou >= 0.3:
                    per_table["C"].append({
                        "image_id": image_id,
                        "table_box": list(best_gt["table_box"]),
                        "caption_boxes": [list(c) for c in best_gt["caption_boxes"]],
                        "pred_box": list(refined),
                    })

    for f in ["A", "B", "C"]:
        print(f"  Formulation {f}: {len(per_table[f])} tables")

    return per_table


def compute_cucr_sensitivity(per_table: dict[str, list[dict]]) -> dict:
    """Compute CUCR at multiple IoU thresholds."""
    results = {}
    for f in ["A", "B", "C"]:
        results[f] = {}
        for thresh in CUCR_THRESHOLDS:
            cucr_vals = []
            for record in per_table[f]:
                tb = tuple(record["table_box"])
                caps = [tuple(c) for c in record["caption_boxes"]]
                pred = tuple(record["pred_box"])
                cu = complete_unit_capture_rate(tb, caps, pred, iou_threshold=thresh)
                cucr_vals.append(1.0 if cu else 0.0)
            rate = np.mean(cucr_vals) if cucr_vals else 0.0
            results[f][str(thresh)] = float(rate)
    return results


def compute_scs_weighting(per_table: dict[str, list[dict]]) -> dict:
    """Compute SCS under alternative weighting schemes."""
    results = {}
    for f in ["A", "B", "C"]:
        results[f] = {}
        for scheme_name, weights in SCS_WEIGHT_SCHEMES.items():
            scs_vals = []
            for record in per_table[f]:
                tb = tuple(record["table_box"])
                caps = [tuple(c) for c in record["caption_boxes"]]
                pred = tuple(record["pred_box"])
                scs = semantic_coverage_score(tb, caps, pred, weights=weights)
                scs_vals.append(scs)
            results[f][scheme_name] = float(np.mean(scs_vals)) if scs_vals else 0.0
    return results


def compute_scatter_data(per_table: dict[str, list[dict]]) -> dict:
    """Compute per-table over-crop and SCS for scatter plot."""
    scatter = {}
    for f in ["A", "B", "C"]:
        points = []
        for record in per_table[f]:
            tb = tuple(record["table_box"])
            caps = [tuple(c) for c in record["caption_boxes"]]
            pred = tuple(record["pred_box"])
            oc = over_crop_ratio(tb, caps, pred)
            scs = semantic_coverage_score(tb, caps, pred)
            points.append({"over_crop": float(oc), "scs": float(scs)})
        scatter[f] = points
    return scatter


def print_cucr_table(cucr_results: dict):
    """Print CUCR sensitivity table."""
    print("\n" + "=" * 70)
    print("CUCR SENSITIVITY TO IoU THRESHOLD")
    print("=" * 70)
    header = f"{'Threshold':<15}"
    for f in ["A", "B", "C"]:
        header += f" {'Form. ' + f:>15}"
    print(header)
    print("-" * 60)
    for thresh in CUCR_THRESHOLDS:
        row = f"{thresh:<15.1f}"
        for f in ["A", "B", "C"]:
            val = cucr_results[f][str(thresh)]
            row += f" {val:>15.3f}"
        print(row)


def print_scs_table(scs_results: dict):
    """Print SCS weighting sensitivity."""
    print("\n" + "=" * 70)
    print("SCS SENSITIVITY TO COMPONENT WEIGHTING")
    print("=" * 70)
    header = f"{'Scheme':<20}"
    for f in ["A", "B", "C"]:
        header += f" {'Form. ' + f:>15}"
    print(header)
    print("-" * 65)
    for scheme in SCS_WEIGHT_SCHEMES:
        row = f"{scheme:<20}"
        for f in ["A", "B", "C"]:
            val = scs_results[f][scheme]
            row += f" {val:>15.3f}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Metric robustness analysis")
    parser.add_argument("--output", default=None,
                        help="Save results JSON to this path")
    parser.add_argument("--per-table-cache", default=None,
                        help="Load/save per-table data cache (skip re-evaluation)")
    args = parser.parse_args()

    cache_path = args.per_table_cache
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached per-table data from {cache_path}...")
        with open(cache_path) as fp:
            per_table = json.load(fp)
    else:
        per_table = collect_per_table_data()
        if cache_path:
            with open(cache_path, "w") as fp:
                json.dump(per_table, fp)
            print(f"  Per-table cache saved to {cache_path}")

    # 1. CUCR threshold sensitivity
    cucr_results = compute_cucr_sensitivity(per_table)
    print_cucr_table(cucr_results)

    # 2. SCS weighting sensitivity
    scs_results = compute_scs_weighting(per_table)
    print_scs_table(scs_results)

    # 3. Scatter data for diagnostic plot
    scatter_data = compute_scatter_data(per_table)
    for f in ["A", "B", "C"]:
        n = len(scatter_data[f])
        oc_mean = np.mean([p["over_crop"] for p in scatter_data[f]]) if n else 0
        scs_mean = np.mean([p["scs"] for p in scatter_data[f]]) if n else 0
        print(f"\n  Form. {f}: {n} points, mean over-crop={oc_mean:.3f}, mean SCS={scs_mean:.3f}")

    # Save
    output = {
        "cucr_sensitivity": cucr_results,
        "scs_weighting": scs_results,
        "scatter_summary": {
            f: {
                "n": len(scatter_data[f]),
                "over_crop_mean": float(np.mean([p["over_crop"] for p in scatter_data[f]])),
                "scs_mean": float(np.mean([p["scs"] for p in scatter_data[f]])),
            }
            for f in ["A", "B", "C"]
        },
    }

    if args.output:
        # Save full results with scatter data
        output["scatter_data"] = scatter_data
        with open(args.output, "w") as fp:
            json.dump(output, fp, indent=2)
        print(f"\nResults saved to {args.output}")

    return output


if __name__ == "__main__":
    main()
