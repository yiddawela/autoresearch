"""
Unified experiment harness for table extraction formulation comparison.

Runs all three formulations on a dataset and computes both COCO-standard
AP metrics and semantic completeness metrics. This is the main evaluation
entry point for WP2 experiments.

Formulations:
  A: Table-only detection (baseline)
  B: Merged semantically complete crop (fine-tuned model)
  C: Table-only detection + heuristic expansion (or linked-region)

Usage:
    uv run experiment_runner.py --formulation all
    uv run experiment_runner.py --formulation B --model-path /path/to/checkpoint
    uv run experiment_runner.py --formulation C --linking-mode spatial
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from coco_eval import compute_ap, compute_map, compute_iou_distribution
from completeness_metrics import compute_all_metrics
from crop_tables import (
    detect_tables,
    get_device,
    load_model,
    refine_crop,
    suppress_duplicates,
)
from linked_region import (
    DetectedRegion,
    LinkedUnit,
    link_from_gt_annotations,
    link_tables_captions,
)

DEFAULT_DATA_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "table_cropper", "SCI-3000"
)
DEFAULT_FINETUNED = os.path.join(
    os.path.expanduser("~"), ".cache", "table_cropper", "checkpoints", "round01_best"
)


# ---------------------------------------------------------------------------
# SCI-3000 annotation parsing (shared)
# ---------------------------------------------------------------------------


def parse_bbox_value(value: str) -> tuple[float, float, float, float]:
    match = re.match(r"xywh=pixel:([\d.]+),([\d.]+),([\d.]+),([\d.]+)", value)
    if not match:
        raise ValueError(f"Cannot parse bbox: {value}")
    return tuple(float(match.group(i)) for i in range(1, 5))


def parse_page_annotations(json_path: str) -> dict:
    with open(json_path) as f:
        data = json.load(f)
    canvas_w = data.get("canvasWidth", 0)
    canvas_h = data.get("canvasHeight", 0)
    tables, figures, captions = [], [], []
    for ann in data.get("annotations", []):
        ann_id = ann.get("id", "")
        bodies = ann.get("body", [])
        ann_type, parent_id = None, None
        for body in bodies:
            if body.get("purpose") == "img-cap-enum":
                ann_type = body.get("value")
            elif body.get("purpose") == "parent":
                parent_id = body.get("value")
        try:
            bbox = parse_bbox_value(ann["target"]["selector"]["value"])
        except (KeyError, ValueError):
            continue
        info = {"id": ann_id, "type": ann_type, "bbox": bbox, "parent_id": parent_id}
        if ann_type == "Table":
            tables.append(info)
        elif ann_type == "Figure":
            figures.append(info)
        elif ann_type == "Caption":
            captions.append(info)
    return {"canvas_size": (canvas_w, canvas_h), "tables": tables,
            "figures": figures, "captions": captions}


def _xywh_to_xyxy(bbox: tuple, scale_x: float, scale_y: float) -> tuple:
    """Convert (x, y, w, h) in canvas coords to (x0, y0, x1, y1) in image coords."""
    x, y, w, h = bbox
    return (x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y)


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


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_val_pages(
    data_dir: str,
    max_pages: int | None = None,
    seed: int = 42,
) -> list[tuple[str, dict]]:
    """Load pages from val split (pre-prepared finetune data)."""
    annot_dir = os.path.join(data_dir, "Annotations")
    val_img_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "table_cropper", "finetune_data", "val", "images"
    )

    if os.path.isdir(val_img_dir):
        page_ids = [Path(f).stem for f in os.listdir(val_img_dir) if f.endswith(".png")]
    else:
        # Fallback: scan annotations
        page_ids = [Path(f).stem for f in Path(annot_dir).glob("*.json")]

    results = []
    for pid in sorted(page_ids):
        ann_path = os.path.join(annot_dir, f"{pid}.json")
        if not os.path.exists(ann_path):
            continue
        parsed = parse_page_annotations(ann_path)
        if parsed["tables"]:
            results.append((ann_path, parsed))

    if max_pages:
        import random
        rng = random.Random(seed)
        rng.shuffle(results)
        results = results[:max_pages]

    return results


# ---------------------------------------------------------------------------
# Per-formulation evaluation
# ---------------------------------------------------------------------------


class FormulationResult:
    """Accumulates results for one formulation across all pages."""

    def __init__(self, name: str):
        self.name = name
        self.detections = []      # For COCO AP: {"image_id", "bbox", "score"}
        self.gt_merged = []       # For COCO AP: {"image_id", "bbox"} (merged target)
        self.gt_table_only = []   # For COCO AP: {"image_id", "bbox"} (table-only target)
        self.completeness = []    # Per-sample completeness metrics

    def add_detection(self, image_id: str, bbox: tuple, score: float):
        self.detections.append({"image_id": image_id, "bbox": bbox, "score": score})

    def add_gt_merged(self, image_id: str, bbox: tuple):
        self.gt_merged.append({"image_id": image_id, "bbox": bbox})

    def add_gt_table_only(self, image_id: str, bbox: tuple):
        self.gt_table_only.append({"image_id": image_id, "bbox": bbox})

    def add_completeness(self, metrics: dict):
        self.completeness.append(metrics)

    def compute_summary(self) -> dict:
        """Compute aggregate metrics."""
        summary = {"name": self.name}

        # COCO AP against merged target
        if self.detections and self.gt_merged:
            coco = compute_map(self.detections, self.gt_merged)
            summary["AP50_merged"] = coco["AP50"]
            summary["AP75_merged"] = coco["AP75"]
            summary["mAP_merged"] = coco["mAP"]

        # COCO AP against table-only target
        if self.detections and self.gt_table_only:
            coco_table = compute_map(self.detections, self.gt_table_only)
            summary["AP50_table"] = coco_table["AP50"]
            summary["AP75_table"] = coco_table["AP75"]
            summary["mAP_table"] = coco_table["mAP"]

        # IoU distribution against merged target
        if self.detections and self.gt_merged:
            iou_dist = compute_iou_distribution(self.detections, self.gt_merged)
            summary["iou_mean"] = iou_dist["mean"]
            summary["iou_median"] = iou_dist["median"]

        # Semantic completeness
        if self.completeness:
            for key in ["cir", "scs", "over_crop", "table_iou",
                        "table_coverage", "caption_coverage"]:
                vals = [c[key] for c in self.completeness if key in c]
                if vals:
                    summary[key] = float(np.mean(vals))

            # CUCR (binary)
            cucr_vals = [c.get("cucr", False) for c in self.completeness]
            summary["cucr"] = sum(1 for v in cucr_vals if v) / max(1, len(cucr_vals))

        summary["n_detections"] = len(self.detections)
        summary["n_gt_merged"] = len(self.gt_merged)
        summary["n_gt_table"] = len(self.gt_table_only)
        summary["n_samples"] = len(self.completeness)

        return summary


def run_formulation_a(
    image: Image.Image,
    image_id: str,
    gt_components: list[dict],
    processor, model, device,
    confidence: float,
    result: FormulationResult,
):
    """Formulation A: Table-only detection (pre-trained, no refinement)."""
    dets = detect_tables(image, processor, model, device, confidence)
    dets = suppress_duplicates(dets)

    for gt in gt_components:
        result.add_gt_merged(image_id, gt["merged_box"])
        result.add_gt_table_only(image_id, gt["table_box"])

    for det in dets:
        bbox = tuple(det["bbox"])
        result.add_detection(image_id, bbox, det["score"])

        # Match to closest GT for completeness metrics
        best_iou, best_gt = 0, None
        for gt in gt_components:
            iou = _iou(bbox, gt["table_box"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        if best_gt and best_iou >= 0.3:
            metrics = compute_all_metrics(
                best_gt["table_box"], best_gt["caption_boxes"], bbox
            )
            result.add_completeness(metrics)


def run_formulation_b(
    image: Image.Image,
    image_id: str,
    gt_components: list[dict],
    processor, model, device,
    confidence: float,
    result: FormulationResult,
):
    """Formulation B: Merged semantically complete crop (fine-tuned model)."""
    dets = detect_tables(image, processor, model, device, confidence)
    dets = suppress_duplicates(dets)

    for gt in gt_components:
        result.add_gt_merged(image_id, gt["merged_box"])
        result.add_gt_table_only(image_id, gt["table_box"])

    for det in dets:
        bbox = tuple(det["bbox"])
        result.add_detection(image_id, bbox, det["score"])

        best_iou, best_gt = 0, None
        for gt in gt_components:
            iou = _iou(bbox, gt["merged_box"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        if best_gt and best_iou >= 0.3:
            metrics = compute_all_metrics(
                best_gt["table_box"], best_gt["caption_boxes"], bbox
            )
            result.add_completeness(metrics)


def run_formulation_c(
    image: Image.Image,
    image_id: str,
    gt_components: list[dict],
    processor, model, device,
    confidence: float,
    result: FormulationResult,
    use_heuristic: bool = True,
):
    """Formulation C: Table-only detection + heuristic expansion / linking."""
    dets = detect_tables(image, processor, model, device, confidence)
    dets = suppress_duplicates(dets)

    for gt in gt_components:
        result.add_gt_merged(image_id, gt["merged_box"])
        result.add_gt_table_only(image_id, gt["table_box"])

    for det in dets:
        raw_bbox = tuple(det["bbox"])

        if use_heuristic:
            refined = refine_crop(image, det["bbox"])
            bbox = refined
        else:
            bbox = raw_bbox

        result.add_detection(image_id, bbox, det["score"])

        best_iou, best_gt = 0, None
        for gt in gt_components:
            iou = _iou(raw_bbox, gt["table_box"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        if best_gt and best_iou >= 0.3:
            metrics = compute_all_metrics(
                best_gt["table_box"], best_gt["caption_boxes"], bbox
            )
            result.add_completeness(metrics)


# ---------------------------------------------------------------------------
# Ground truth components builder
# ---------------------------------------------------------------------------


def build_gt_components(
    parsed: dict, img_w: int, img_h: int
) -> list[dict]:
    """Build GT table+caption component pairs in image coordinates."""
    canvas_w, canvas_h = parsed["canvas_size"]
    scale_x = img_w / canvas_w if canvas_w > 0 else 1.0
    scale_y = img_h / canvas_h if canvas_h > 0 else 1.0

    components = []
    for table_info in parsed["tables"]:
        table_box = _xywh_to_xyxy(table_info["bbox"], scale_x, scale_y)

        # Find linked captions
        table_id = table_info["id"]
        caption_boxes = []
        for cap in parsed["captions"]:
            if cap["parent_id"] == table_id:
                caption_boxes.append(_xywh_to_xyxy(cap["bbox"], scale_x, scale_y))

        # Build merged box
        if caption_boxes:
            all_boxes = [table_box] + caption_boxes
            merged_box = (
                min(b[0] for b in all_boxes),
                min(b[1] for b in all_boxes),
                max(b[2] for b in all_boxes),
                max(b[3] for b in all_boxes),
            )
        else:
            merged_box = table_box

        components.append({
            "table_box": table_box,
            "caption_boxes": caption_boxes,
            "merged_box": merged_box,
        })

    return components


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    data_dir: str,
    formulations: list[str],
    finetuned_path: str | None = None,
    max_pages: int | None = None,
    dpi: int = 200,
    confidence: float = 0.5,
    seed: int = 42,
    output_path: str | None = None,
) -> dict[str, dict]:
    """Run the full experiment comparing formulations."""
    from pdf2image import convert_from_path

    print("=" * 70)
    print("EXPERIMENT RUNNER")
    print(f"Formulations: {', '.join(formulations)}")
    print("=" * 70)

    # Load pages
    print("\nLoading pages...")
    pages = load_val_pages(data_dir, max_pages, seed)
    print(f"  {len(pages)} pages loaded")

    # Load models
    device = get_device()
    print(f"  Device: {device}")

    models = {}
    if "A" in formulations or "C" in formulations:
        print("  Loading pre-trained TATR...")
        models["pretrained"] = load_model(device)

    if "B" in formulations:
        if finetuned_path and os.path.isdir(finetuned_path):
            print(f"  Loading fine-tuned model from {finetuned_path}...")
            models["finetuned"] = load_model(device, model_path=finetuned_path)
        else:
            print("  ⚠ No fine-tuned model; using pre-trained for B")
            models["finetuned"] = models.get("pretrained", load_model(device))

    # Initialize results
    results = {}
    for f in formulations:
        results[f] = FormulationResult(f"Formulation {f}")

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

    # Process
    print(f"\nProcessing {len(pdf_pages)} PDFs...")
    for pdf_id, page_list in tqdm(pdf_pages.items(), desc="PDFs"):
        pdf_path = os.path.join(data_dir, "PDFs", f"{pdf_id}.pdf")
        max_page = max(pn for pn, _, _ in page_list)

        try:
            rendered = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=max_page)
        except Exception as e:
            tqdm.write(f"  ⚠ {pdf_id}: {e}")
            continue

        for page_num, parsed, annot_path in page_list:
            if page_num > len(rendered):
                continue

            image = rendered[page_num - 1]
            img_w, img_h = image.size
            image_id = Path(annot_path).stem

            gt_components = build_gt_components(parsed, img_w, img_h)
            if not gt_components:
                continue

            # Only evaluate tables that have captions
            gt_with_captions = [g for g in gt_components if g["caption_boxes"]]
            if not gt_with_captions:
                continue

            if "A" in formulations:
                proc, model = models["pretrained"]
                run_formulation_a(
                    image, image_id, gt_with_captions,
                    proc, model, device, confidence, results["A"]
                )

            if "B" in formulations:
                proc, model = models["finetuned"]
                run_formulation_b(
                    image, image_id, gt_with_captions,
                    proc, model, device, confidence, results["B"]
                )

            if "C" in formulations:
                proc, model = models["pretrained"]
                run_formulation_c(
                    image, image_id, gt_with_captions,
                    proc, model, device, confidence, results["C"]
                )

    # Compute summaries
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    summaries = {}
    for f in formulations:
        summary = results[f].compute_summary()
        summaries[f] = summary

    # Print comparison table
    metric_labels = [
        ("AP50_merged", "AP@50 (merged)"),
        ("AP75_merged", "AP@75 (merged)"),
        ("mAP_merged", "mAP (merged)"),
        ("AP50_table", "AP@50 (table)"),
        ("AP75_table", "AP@75 (table)"),
        ("cir", "Caption Incl. Rate"),
        ("scs", "Semantic Coverage"),
        ("cucr", "Complete Unit Cap."),
        ("over_crop", "Over-crop Ratio"),
        ("table_iou", "Table IoU"),
        ("iou_mean", "IoU Mean (merged)"),
    ]

    header = f"{'Metric':<25}"
    for f in formulations:
        header += f" {'Form. ' + f:>15}"
    print(header)
    print("-" * (25 + 16 * len(formulations)))

    for key, label in metric_labels:
        row = f"{label:<25}"
        for f in formulations:
            val = summaries[f].get(key)
            if val is not None:
                row += f" {val:>15.4f}"
            else:
                row += f" {'N/A':>15}"
        print(row)

    print(f"\n{'Counts':<25}")
    for f in formulations:
        s = summaries[f]
        print(f"  Form. {f}: {s['n_detections']} dets, {s['n_gt_merged']} GT(merged), "
              f"{s['n_samples']} completeness samples")

    # Save results
    if output_path:
        with open(output_path, "w") as fp:
            json.dump(summaries, fp, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    return summaries


def main():
    parser = argparse.ArgumentParser(description="Run formulation comparison experiment")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--finetuned-path", default=DEFAULT_FINETUNED)
    parser.add_argument("--formulation", default="all",
                        help="Which formulations to run: A, B, C, or 'all'")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Save JSON results to this path")
    args = parser.parse_args()

    if args.formulation == "all":
        formulations = ["A", "B", "C"]
    else:
        formulations = [f.strip().upper() for f in args.formulation.split(",")]

    run_experiment(
        data_dir=args.data_dir,
        formulations=formulations,
        finetuned_path=args.finetuned_path,
        max_pages=args.max_pages,
        dpi=args.dpi,
        confidence=args.confidence,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
