"""
WP3: Downstream extraction experiment.

Tests whether better semantic completeness (Formulation B) measurably improves
scientific data extraction compared to table-only (A) and heuristic (C) crops.

Uses GPT-5.2 via OpenRouter API for structured extraction from table images.

Two tiers:
  Tier 1: Table title extraction + study arm identification (simple)
  Tier 2: Full field extraction — all numeric values with row/column headers

Usage:
    uv run downstream_extraction.py --n-tables 100
    uv run downstream_extraction.py --n-tables 100 --tier both --output wp3_results.json
"""

import _paths  # noqa: F401
import argparse
import base64
import io
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from crop_tables import (
    detect_tables,
    get_device,
    load_model,
    refine_crop,
    suppress_duplicates,
)
from experiment_runner import (
    build_gt_components,
    load_val_pages,
)

from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-5.2"  # Frontier vision model via OpenRouter

DEFAULT_DATA_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "table_cropper", "SCI-3000"
)
DEFAULT_FINETUNED = os.path.join(
    os.path.expanduser("~"), ".cache", "table_cropper", "checkpoints",
    "phase5_resumed_final"
)


def encode_image_base64(image: Image.Image, max_size: int = 1024) -> str:
    """Encode PIL image to base64 data URL, resizing if needed."""
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def call_openrouter(
    prompt: str,
    image_url: str,
    model: str = MODEL,
    max_retries: int = 3,
) -> str | None:
    """Call OpenRouter API with vision prompt."""
    import requests

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": 2000,
        "temperature": 0.0,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
                continue
            else:
                print(f"  API error {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as e:
            print(f"  Request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    return None


# ---------------------------------------------------------------------------
# Tier 1: Title extraction + study arm identification
# ---------------------------------------------------------------------------

TIER1_PROMPT = """Look at this image of a table from a scientific paper. Extract the following information in JSON format:

{
  "table_title": "the full title/caption of the table (if visible)",
  "study_arms": ["list of study groups/arms/conditions compared in the table"],
  "has_caption_visible": true/false
}

If the table title/caption is not visible in the image, set table_title to null and has_caption_visible to false.
Return ONLY the JSON, no other text."""


def extract_tier1(image: Image.Image) -> dict | None:
    """Tier 1: Extract table title and study arms."""
    img_url = encode_image_base64(image)
    response = call_openrouter(TIER1_PROMPT, img_url)
    if not response:
        return None
    try:
        # Strip markdown code fences if present
        response = re.sub(r'^```json\s*', '', response.strip())
        response = re.sub(r'\s*```$', '', response.strip())
        return json.loads(response)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Tier 2: Full field extraction
# ---------------------------------------------------------------------------

TIER2_PROMPT = """Look at this image of a table from a scientific paper. Extract ALL data from the table in structured JSON format:

{
  "table_title": "the full title/caption (if visible)",
  "column_headers": ["list of column headers"],
  "rows": [
    {
      "row_label": "row identifier",
      "values": ["value1", "value2", ...]
    }
  ]
}

Extract every numeric value with its associated row and column context.
If the table title/caption is not visible, set table_title to null.
Return ONLY the JSON, no other text."""


def extract_tier2(image: Image.Image) -> dict | None:
    """Tier 2: Full structured field extraction."""
    img_url = encode_image_base64(image)
    response = call_openrouter(TIER2_PROMPT, img_url)
    if not response:
        return None
    try:
        response = re.sub(r'^```json\s*', '', response.strip())
        response = re.sub(r'\s*```$', '', response.strip())
        return json.loads(response)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_tier1(result_a: dict, result_b: dict, result_c: dict) -> dict:
    """Score Tier 1 results — compare title extraction across formulations."""
    scores = {}
    for name, result in [("A", result_a), ("B", result_b), ("C", result_c)]:
        if result is None or not isinstance(result, dict):
            scores[name] = {"title_extracted": False, "has_caption": False, "n_arms": 0}
        else:
            title = result.get("table_title")
            scores[name] = {
                "title_extracted": title is not None and len(str(title)) > 5,
                "has_caption": result.get("has_caption_visible", False),
                "n_arms": len(result.get("study_arms", [])),
            }
    return scores


def score_tier2(result_a: dict, result_b: dict, result_c: dict) -> dict:
    """Score Tier 2 results — compare extraction completeness."""
    scores = {}
    for name, result in [("A", result_a), ("B", result_b), ("C", result_c)]:
        if result is None or not isinstance(result, dict):
            scores[name] = {"title_extracted": False, "n_columns": 0, "n_rows": 0, "n_values": 0}
        else:
            title = result.get("table_title")
            cols = result.get("column_headers", [])
            if not isinstance(cols, list):
                cols = []
            rows = result.get("rows", [])
            if not isinstance(rows, list):
                rows = []
            n_values = sum(len(r.get("values", [])) for r in rows if isinstance(r, dict))
            scores[name] = {
                "title_extracted": title is not None and len(str(title)) > 5,
                "n_columns": len(cols),
                "n_rows": len(rows),
                "n_values": n_values,
            }
    return scores


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def generate_crops(
    image: Image.Image,
    det: dict,
    proc_ft, model_ft, device: str,
    confidence: float,
) -> dict:
    """Generate three crop types from a single detection."""
    bbox = det["bbox"]

    # Formulation A: table-only (tight box from pre-trained detection)
    x0, y0, x1, y1 = [int(round(c)) for c in bbox]
    pad = 10
    crop_a = image.crop((
        max(0, x0 - pad), max(0, y0 - pad),
        min(image.width, x1 + pad), min(image.height, y1 + pad)
    ))

    # Formulation B: fine-tuned merged detection
    dets_ft = detect_tables(image, proc_ft, model_ft, device, confidence)
    dets_ft = suppress_duplicates(dets_ft)
    # Find best matching fine-tuned detection
    from experiment_runner import _iou
    best_ft_det = None
    best_iou = 0
    for d in dets_ft:
        iou = _iou(tuple(bbox), tuple(d["bbox"]))
        if iou > best_iou:
            best_iou = iou
            best_ft_det = d

    if best_ft_det:
        bx0, by0, bx1, by1 = [int(round(c)) for c in best_ft_det["bbox"]]
        crop_b = image.crop((
            max(0, bx0 - 5), max(0, by0 - 5),
            min(image.width, bx1 + 5), min(image.height, by1 + 5)
        ))
    else:
        crop_b = crop_a  # fallback

    # Formulation C: heuristic expansion
    refined = refine_crop(image, bbox)
    crop_c = image.crop(refined)

    return {"A": crop_a, "B": crop_b, "C": crop_c}


def run_experiment(
    n_tables: int = 100,
    tier: str = "both",
    data_dir: str = DEFAULT_DATA_DIR,
    finetuned_path: str = DEFAULT_FINETUNED,
    output_path: str | None = None,
    seed: int = 42,
    confidence: float = 0.5,
):
    """Run the downstream extraction experiment."""
    from pdf2image import convert_from_path
    
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not set in .env")
        return

    print("=" * 70)
    print("WP3: DOWNSTREAM EXTRACTION EXPERIMENT")
    print(f"Tier: {tier} | Tables: {n_tables} | Model: {MODEL}")
    print("=" * 70)

    # Load models
    device = get_device()
    print(f"\nDevice: {device}")
    print("Loading pre-trained TATR...")
    proc_pt, model_pt = load_model(device)
    print(f"Loading fine-tuned TATR from {finetuned_path}...")
    proc_ft, model_ft = load_model(device, model_path=finetuned_path)

    # Load pages
    print("Loading val pages...")
    pages = load_val_pages(data_dir, seed=seed)
    print(f"  {len(pages)} pages loaded")

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

    # Collect tables across all PDFs
    print(f"\nCollecting tables from {len(pdf_pages)} PDFs...")

    all_tables = []  # (image, detection, gt_component, pdf_id, page_num)

    pdf_ids = list(pdf_pages.keys())
    random.seed(seed)
    random.shuffle(pdf_ids)

    for pdf_id in tqdm(pdf_ids, desc="Scanning PDFs"):
        if len(all_tables) >= n_tables * 2:  # collect extras for filtering
            break

        page_list = pdf_pages[pdf_id]
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

            gt_components = build_gt_components(parsed, img_w, img_h)
            gt_with_captions = [g for g in gt_components if g["caption_boxes"]]

            if not gt_with_captions:
                continue

            dets = detect_tables(image, proc_pt, model_pt, device, confidence)
            dets = suppress_duplicates(dets)

            for det in dets:
                from experiment_runner import _iou
                for gt in gt_with_captions:
                    iou = _iou(tuple(det["bbox"]), gt["table_box"])
                    if iou >= 0.5:
                        all_tables.append((image, det, gt, pdf_id, page_num))
                        break

    random.shuffle(all_tables)
    all_tables = all_tables[:n_tables]
    print(f"  Selected {len(all_tables)} tables for extraction")

    # Run extraction
    results = []
    tier1_scores = {"A": [], "B": [], "C": []}
    tier2_scores = {"A": [], "B": [], "C": []}

    for i, (image, det, gt, pdf_id, page_num) in enumerate(tqdm(all_tables, desc="Extracting")):
        crops = generate_crops(image, det, proc_ft, model_ft, device, confidence)

        table_result = {
            "table_idx": i,
            "pdf_id": pdf_id,
            "page_num": page_num,
        }

        if tier in ("1", "both"):
            r_a = extract_tier1(crops["A"])
            r_b = extract_tier1(crops["B"])
            r_c = extract_tier1(crops["C"])
            table_result["tier1"] = {
                "A": r_a, "B": r_b, "C": r_c,
                "scores": score_tier1(r_a, r_b, r_c),
            }
            for f in ["A", "B", "C"]:
                tier1_scores[f].append(table_result["tier1"]["scores"][f])

        if tier in ("2", "both"):
            r_a = extract_tier2(crops["A"])
            r_b = extract_tier2(crops["B"])
            r_c = extract_tier2(crops["C"])
            table_result["tier2"] = {
                "A": r_a, "B": r_b, "C": r_c,
                "scores": score_tier2(r_a, r_b, r_c),
            }
            for f in ["A", "B", "C"]:
                tier2_scores[f].append(table_result["tier2"]["scores"][f])

        results.append(table_result)

    # Print summary
    print("\n" + "=" * 70)
    print("DOWNSTREAM EXTRACTION RESULTS")
    print("=" * 70)

    if tier in ("1", "both") and tier1_scores["A"]:
        print("\n--- Tier 1: Title Extraction ---")
        print(f"{'Metric':<30} {'Form. A':>10} {'Form. B':>10} {'Form. C':>10}")
        print("-" * 60)

        for f in ["A", "B", "C"]:
            scores = tier1_scores[f]
            title_rate = sum(1 for s in scores if s["title_extracted"]) / len(scores)
            caption_rate = sum(1 for s in scores if s["has_caption"]) / len(scores)
            avg_arms = np.mean([s["n_arms"] for s in scores])
            if f == "A":
                print(f"{'Title extraction rate':<30} {title_rate:>10.1%} ", end="")
            elif f == "B":
                print(f"{title_rate:>10.1%} ", end="")
            else:
                print(f"{title_rate:>10.1%}")

        # Re-print properly
        for metric_name, metric_key in [
            ("Title extraction rate", "title_extracted"),
            ("Caption visible", "has_caption"),
        ]:
            row = f"{metric_name:<30}"
            for f in ["A", "B", "C"]:
                scores = tier1_scores[f]
                rate = sum(1 for s in scores if s[metric_key]) / len(scores)
                row += f" {rate:>10.1%}"
            print(row)

        row = f"{'Mean study arms found':<30}"
        for f in ["A", "B", "C"]:
            avg = np.mean([s["n_arms"] for s in tier1_scores[f]])
            row += f" {avg:>10.1f}"
        print(row)

    if tier in ("2", "both") and tier2_scores["A"]:
        print("\n--- Tier 2: Full Field Extraction ---")
        print(f"{'Metric':<30} {'Form. A':>10} {'Form. B':>10} {'Form. C':>10}")
        print("-" * 60)

        for metric_name, metric_key in [
            ("Title extraction rate", "title_extracted"),
            ("Mean columns extracted", "n_columns"),
            ("Mean rows extracted", "n_rows"),
            ("Mean values extracted", "n_values"),
        ]:
            row = f"{metric_name:<30}"
            for f in ["A", "B", "C"]:
                scores = tier2_scores[f]
                if metric_key == "title_extracted":
                    val = sum(1 for s in scores if s[metric_key]) / len(scores)
                    row += f" {val:>10.1%}"
                else:
                    val = np.mean([s[metric_key] for s in scores])
                    row += f" {val:>10.1f}"
            print(row)

    # Save results
    if output_path:
        summary = {
            "model": MODEL,
            "n_tables": len(all_tables),
            "tier": tier,
        }
        if tier1_scores["A"]:
            summary["tier1"] = {}
            for f in ["A", "B", "C"]:
                scores = tier1_scores[f]
                summary["tier1"][f] = {
                    "title_rate": sum(1 for s in scores if s["title_extracted"]) / max(1, len(scores)),
                    "caption_rate": sum(1 for s in scores if s["has_caption"]) / max(1, len(scores)),
                    "mean_arms": float(np.mean([s["n_arms"] for s in scores])) if scores else 0,
                }
        if tier2_scores["A"]:
            summary["tier2"] = {}
            for f in ["A", "B", "C"]:
                scores = tier2_scores[f]
                summary["tier2"][f] = {
                    "title_rate": sum(1 for s in scores if s["title_extracted"]) / max(1, len(scores)),
                    "mean_columns": float(np.mean([s["n_columns"] for s in scores])) if scores else 0,
                    "mean_rows": float(np.mean([s["n_rows"] for s in scores])) if scores else 0,
                    "mean_values": float(np.mean([s["n_values"] for s in scores])) if scores else 0,
                }

        with open(output_path, "w") as fp:
            json.dump({"summary": summary, "per_table": results}, fp, indent=2, default=str)
        print(f"\nSaved to {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="WP3 downstream extraction")
    parser.add_argument("--n-tables", type=int, default=100)
    parser.add_argument("--tier", choices=["1", "2", "both"], default="both")
    parser.add_argument("--output", default="wp3_results.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finetuned-path", default=DEFAULT_FINETUNED)
    args = parser.parse_args()

    run_experiment(
        n_tables=args.n_tables,
        tier=args.tier,
        output_path=args.output,
        seed=args.seed,
        finetuned_path=args.finetuned_path,
    )


if __name__ == "__main__":
    main()
