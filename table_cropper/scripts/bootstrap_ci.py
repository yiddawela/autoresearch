"""
Bootstrap confidence intervals for WP2 formulation comparison.

Computes 95% CIs for key metrics (CIR, SCS, CUCR, over-crop, IoU)
by resampling per-sample completeness measurements from experiment_runner.

Usage:
    uv run bootstrap_ci.py --results wp2_results_final.json
    uv run bootstrap_ci.py --results-dir . --pattern 'wp2_results_*.json'
"""

import _paths  # noqa: F401
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Returns (mean, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    n = len(arr)

    if n == 0:
        return (float('nan'), float('nan'), float('nan'))

    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
    mean = float(np.mean(arr))

    return (mean, lower, upper)


def compute_bootstrap_from_experiment(
    results_path: str,
    n_bootstrap: int = 10000,
) -> dict:
    """Re-run the experiment to collect per-sample metrics, then bootstrap.

    Since wp2_results_final.json only has aggregates, we need to re-run
    the experiment with per-sample output. This function runs the experiment
    and collects per-sample completeness metrics for bootstrapping.
    """
    # Import experiment runner components
    from experiment_runner import (
        FormulationResult,
        build_gt_components,
        load_val_pages,
        run_formulation_a,
        run_formulation_b,
        run_formulation_c,
    )
    from crop_tables import get_device, load_model

    DEFAULT_DATA_DIR = os.path.join(
        os.path.expanduser("~"), ".cache", "table_cropper", "SCI-3000"
    )
    DEFAULT_FINETUNED = os.path.join(
        os.path.expanduser("~"), ".cache", "table_cropper", "checkpoints",
        "phase5_resumed_final"
    ), 

    from pdf2image import convert_from_path
    from tqdm import tqdm

    data_dir = DEFAULT_DATA_DIR
    formulations = ["A", "B", "C"]
    confidence = 0.5

    print("Loading pages...")
    pages = load_val_pages(data_dir, seed=42)
    print(f"  {len(pages)} pages loaded")

    device = get_device()
    print(f"  Device: {device}")

    print("  Loading pre-trained TATR...")
    pretrained = load_model(device)
    print(f"  Loading fine-tuned model...")
    finetuned = load_model(device, model_path=DEFAULT_FINETUNED[0])

    results = {f: FormulationResult(f"Formulation {f}") for f in formulations}

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

    print(f"\nProcessing {len(pdf_pages)} PDFs...")
    for pdf_id, page_list in tqdm(pdf_pages.items(), desc="PDFs"):
        pdf_path = os.path.join(data_dir, "PDFs", f"{pdf_id}.pdf")
        max_page = max(pn for pn, _, _ in page_list)

        try:
            rendered = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=max_page)
        except Exception as e:
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

            proc_pt, model_pt = pretrained
            run_formulation_a(image, image_id, gt_with_captions,
                              proc_pt, model_pt, device, confidence, results["A"])

            proc_ft, model_ft = finetuned
            run_formulation_b(image, image_id, gt_with_captions,
                              proc_ft, model_ft, device, confidence, results["B"])

            run_formulation_c(image, image_id, gt_with_captions,
                              proc_pt, model_pt, device, confidence, results["C"])

    # Now compute bootstrap CIs from per-sample completeness metrics
    print("\n" + "=" * 70)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
    print("=" * 70)

    metrics_keys = ["cir", "scs", "over_crop", "table_iou"]
    all_ci = {}

    for f in formulations:
        all_ci[f] = {}
        samples = results[f].completeness

        for key in metrics_keys:
            vals = [s[key] for s in samples if key in s]
            mean, lo, hi = bootstrap_ci(vals, n_bootstrap=n_bootstrap)
            all_ci[f][key] = {"mean": mean, "ci_lower": lo, "ci_upper": hi, "n": len(vals)}

        # CUCR (binary)
        cucr_vals = [1.0 if s.get("cucr", False) else 0.0 for s in samples]
        mean, lo, hi = bootstrap_ci(cucr_vals, n_bootstrap=n_bootstrap)
        all_ci[f]["cucr"] = {"mean": mean, "ci_lower": lo, "ci_upper": hi, "n": len(cucr_vals)}

    # Print table
    print(f"\n{'Metric':<25} {'Form. A':>20} {'Form. B':>20} {'Form. C':>20}")
    print("-" * 85)

    labels = {
        "cir": "Caption Incl. Rate",
        "scs": "Semantic Coverage",
        "cucr": "Complete Unit Cap.",
        "over_crop": "Over-crop Ratio",
        "table_iou": "Table IoU",
    }
    for key in ["cir", "scs", "cucr", "over_crop", "table_iou"]:
        row = f"{labels[key]:<25}"
        for f in formulations:
            ci = all_ci[f][key]
            row += f" {ci['mean']:.3f} [{ci['ci_lower']:.3f},{ci['ci_upper']:.3f}]"
        print(row)

    print(f"\nSample counts:")
    for f in formulations:
        n = all_ci[f]["cir"]["n"]
        print(f"  Form. {f}: {n} samples")

    # Save
    output_path = results_path.replace(".json", "_bootstrap.json") if results_path else "wp2_bootstrap_ci.json"
    with open(output_path, "w") as fp:
        json.dump(all_ci, fp, indent=2)
    print(f"\nSaved to {output_path}")

    return all_ci


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CIs for WP2")
    parser.add_argument("--results", default="wp2_results_final.json",
                        help="Path to experiment results JSON")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()

    compute_bootstrap_from_experiment(args.results, args.n_bootstrap)


if __name__ == "__main__":
    main()
