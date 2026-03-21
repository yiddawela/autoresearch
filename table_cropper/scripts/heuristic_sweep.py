"""
Heuristic sensitivity sweep for Formulation C.

Tests 5 configurations of the refine_crop() heuristic parameters
to show how sensitive Formulation C results are to parameter choices.

Usage:
    uv run heuristic_sweep.py
    uv run heuristic_sweep.py --output heuristic_sweep_results.json
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

from coco_eval import compute_ap
from completeness_metrics import compute_all_metrics
from crop_tables import (
    _scan_for_content_boundary,
    detect_tables,
    get_device,
    load_model,
    suppress_duplicates,
)
from experiment_runner import (
    FormulationResult,
    _iou,
    build_gt_components,
    load_val_pages,
)

DEFAULT_DATA_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "table_cropper", "SCI-3000"
)


# ---------------------------------------------------------------------------
# Parameterised refine_crop
# ---------------------------------------------------------------------------

def refine_crop_parametric(
    image: Image.Image,
    bbox: list[float],
    pad_top_max: int = 250,
    pad_bottom_max: int = 400,
    pad_side: int = 20,
    large_gap: int = 60,
) -> tuple[int, int, int, int]:
    """refine_crop with explicit parameters for sweep."""
    img_w, img_h = image.size
    bx0, by0, bx1, by1 = [int(round(c)) for c in bbox]

    col_margin = 40
    scan_x0 = max(0, bx0 - col_margin)
    scan_x1 = min(img_w, bx1 + col_margin)

    # Upward scan
    scan_top = max(0, by0 - pad_top_max)
    new_top = by0
    if scan_top < by0:
        region_above = image.crop((scan_x0, scan_top, scan_x1, by0))
        gray_above = np.array(region_above.convert("L"))
        result = _scan_for_content_boundary(gray_above, "up", large_gap)
        if result is not None:
            new_top = max(0, scan_top + result - 8)

    # Downward scan
    scan_bottom = min(img_h, by1 + pad_bottom_max)
    new_bottom = by1
    if scan_bottom > by1:
        region_below = image.crop((scan_x0, by1, scan_x1, scan_bottom))
        gray_below = np.array(region_below.convert("L"))
        result = _scan_for_content_boundary(gray_below, "down", large_gap)
        if result is not None:
            new_bottom = min(img_h, by1 + result + 8)

    x0 = max(0, bx0 - pad_side)
    x1 = min(img_w, bx1 + pad_side)
    return (x0, new_top, x1, new_bottom)


# Five configurations from conservative to aggressive
CONFIGS = {
    "tight": {
        "label": "Tight (minimal expansion)",
        "pad_top_max": 80,
        "pad_bottom_max": 100,
        "pad_side": 10,
        "large_gap": 30,
    },
    "conservative": {
        "label": "Conservative",
        "pad_top_max": 150,
        "pad_bottom_max": 200,
        "pad_side": 15,
        "large_gap": 40,
    },
    "default": {
        "label": "Default (current)",
        "pad_top_max": 250,
        "pad_bottom_max": 400,
        "pad_side": 20,
        "large_gap": 60,
    },
    "aggressive": {
        "label": "Aggressive",
        "pad_top_max": 350,
        "pad_bottom_max": 500,
        "pad_side": 30,
        "large_gap": 80,
    },
    "maximal": {
        "label": "Maximal (very wide)",
        "pad_top_max": 500,
        "pad_bottom_max": 700,
        "pad_side": 40,
        "large_gap": 100,
    },
}


def run_sweep(
    data_dir: str = DEFAULT_DATA_DIR,
    output_path: str | None = None,
    seed: int = 42,
    confidence: float = 0.5,
):
    """Run the sensitivity sweep across all configs."""
    from pdf2image import convert_from_path

    pages = load_val_pages(data_dir, seed=seed)
    print(f"  {len(pages)} pages loaded")

    device = get_device()
    print(f"  Device: {device}")
    print("  Loading pre-trained TATR...")
    proc, model = load_model(device)

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

    # Run all configs
    results_all = {name: FormulationResult(cfg["label"]) for name, cfg in CONFIGS.items()}

    print(f"\nProcessing {len(pdf_pages)} PDFs with {len(CONFIGS)} configs...")
    for pdf_id, page_list in tqdm(pdf_pages.items(), desc="PDFs"):
        pdf_path = os.path.join(data_dir, "PDFs", f"{pdf_id}.pdf")
        max_page = max(pn for pn, _, _ in page_list)

        try:
            rendered = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=max_page)
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

            # Run detection once (shared across configs)
            dets = detect_tables(image, proc, model, device, confidence)
            dets = suppress_duplicates(dets)

            # Evaluate each config
            for config_name, cfg in CONFIGS.items():
                result = results_all[config_name]

                for gt in gt_with_captions:
                    result.add_gt_merged(image_id, gt["merged_box"])

                for det in dets:
                    raw_bbox = tuple(det["bbox"])
                    refined = refine_crop_parametric(
                        image, det["bbox"],
                        pad_top_max=cfg["pad_top_max"],
                        pad_bottom_max=cfg["pad_bottom_max"],
                        pad_side=cfg["pad_side"],
                        large_gap=cfg["large_gap"],
                    )
                    result.add_detection(image_id, refined, det["score"])

                    best_iou, best_gt = 0, None
                    for gt in gt_with_captions:
                        iou = _iou(raw_bbox, gt["table_box"])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = gt

                    if best_gt and best_iou >= 0.3:
                        metrics = compute_all_metrics(
                            best_gt["table_box"], best_gt["caption_boxes"], refined
                        )
                        result.add_completeness(metrics)

    # Print results
    print("\n" + "=" * 90)
    print("HEURISTIC SENSITIVITY SWEEP RESULTS")
    print("=" * 90)

    header = f"{'Metric':<25}"
    for name, cfg in CONFIGS.items():
        header += f" {name:>12}"
    print(header)
    print("-" * 90)

    summary = {}
    for config_name in CONFIGS:
        result = results_all[config_name]
        samples = result.completeness

        cir_vals = [s["cir"] for s in samples if "cir" in s]
        scs_vals = [s["scs"] for s in samples if "scs" in s]
        over_vals = [s["over_crop"] for s in samples if "over_crop" in s]
        cucr_vals = [1.0 if s.get("cucr", False) else 0.0 for s in samples]
        tiou_vals = [s["table_iou"] for s in samples if "table_iou" in s]

        summary[config_name] = {
            "label": CONFIGS[config_name]["label"],
            "params": {k: v for k, v in CONFIGS[config_name].items() if k != "label"},
            "cir": float(np.mean(cir_vals)) if cir_vals else 0,
            "scs": float(np.mean(scs_vals)) if scs_vals else 0,
            "cucr": float(np.mean(cucr_vals)) if cucr_vals else 0,
            "over_crop": float(np.mean(over_vals)) if over_vals else 0,
            "table_iou": float(np.mean(tiou_vals)) if tiou_vals else 0,
            "n_samples": len(samples),
        }

    metrics_to_show = [
        ("CIR", "cir"),
        ("SCS", "scs"),
        ("CUCR", "cucr"),
        ("Over-crop", "over_crop"),
        ("Table IoU", "table_iou"),
    ]

    for label, key in metrics_to_show:
        row = f"{label:<25}"
        for config_name in CONFIGS:
            row += f" {summary[config_name][key]:>12.3f}"
        print(row)

    samples_row = f"{'N samples':<25}"
    for config_name in CONFIGS:
        samples_row += f" {summary[config_name]['n_samples']:>12d}"
    print(samples_row)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved to {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Heuristic sensitivity sweep")
    parser.add_argument("--output", default="heuristic_sweep_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 90)
    print("HEURISTIC SENSITIVITY SWEEP")
    print("=" * 90)
    print("\nLoading pages...")

    run_sweep(output_path=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
