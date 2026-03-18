"""
Prepare SCI-3000 data for DETR fine-tuning.

Renders PDF pages, merges table+caption bounding boxes, and saves
as an on-disk dataset in COCO-format for HuggingFace DETR training.

Usage:
    uv run prepare_finetune_data.py                   # full dataset
    uv run prepare_finetune_data.py --max-pages 200   # quick subset
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Import text-scanning helpers from crop_tables for bbox expansion
from crop_tables import (
    _row_has_text,
    _scan_for_content_boundary,
    _col_has_text,
    _scan_columns_for_text,
)


DEFAULT_DATA_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "table_cropper", "SCI-3000"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "table_cropper", "finetune_data"
)


# ---------------------------------------------------------------------------
# Annotation parsing (same as evaluate.py)
# ---------------------------------------------------------------------------

import re


def parse_bbox_value(value: str) -> tuple[float, float, float, float]:
    """Parse 'xywh=pixel:x,y,w,h' → (x, y, w, h)."""
    match = re.match(r"xywh=pixel:([\d.]+),([\d.]+),([\d.]+),([\d.]+)", value)
    if not match:
        raise ValueError(f"Cannot parse bbox: {value}")
    return tuple(float(match.group(i)) for i in range(1, 5))


def parse_page_annotations(json_path: str) -> dict:
    """Parse a SCI-3000 per-page annotation JSON."""
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


def merge_table_with_captions(
    table_info: dict,
    captions: list[dict],
    canvas_w: int,
    canvas_h: int,
) -> tuple[float, float, float, float]:
    """
    Merge table bbox with its linked captions.
    Returns (x_center, y_center, width, height) normalized to [0, 1].
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

    # Clamp
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(canvas_w, x1)
    y1 = min(canvas_h, y1)

    # Normalize to [0, 1] (COCO format for DETR)
    cx = (x0 + x1) / 2 / canvas_w
    cy = (y0 + y1) / 2 / canvas_h
    w = (x1 - x0) / canvas_w
    h = (y1 - y0) / canvas_h

    return (cx, cy, w, h)


def expand_bbox_on_image(
    page_image: Image.Image,
    norm_bbox: tuple[float, float, float, float],
    canvas_w: int,
    canvas_h: int,
    pad_up_max: int = 80,
    pad_down_max: int = 100,
    pad_right_max: int = 200,
    gap_tolerance: int = 15,
) -> tuple[float, float, float, float]:
    """
    Expand a normalized COCO bbox (cx, cy, w, h) by scanning the rendered
    page image for nearby footnote/caption text.

    Scans:
    - Upward (up to pad_up_max px) for missed title text
    - Downward (up to pad_down_max px) for footnotes
    - Rightward (up to pad_right_max px) for side annotations (wide tables only)

    Returns expanded (cx, cy, w, h) normalized to [0, 1].
    """
    img_w, img_h = page_image.size

    # Undo normalization: COCO (cx, cy, w, h) → pixel (x0, y0, x1, y1)
    # Note: canvas coords != pixel coords when DPI differs from PDF native
    # Scale factor: pixel / canvas
    sx = img_w / canvas_w if canvas_w > 0 else 1
    sy = img_h / canvas_h if canvas_h > 0 else 1

    cx_n, cy_n, w_n, h_n = norm_bbox
    px0 = int((cx_n - w_n / 2) * img_w)
    py0 = int((cy_n - h_n / 2) * img_h)
    px1 = int((cx_n + w_n / 2) * img_w)
    py1 = int((cy_n + h_n / 2) * img_h)

    # Clamp
    px0 = max(0, px0)
    py0 = max(0, py0)
    px1 = min(img_w, px1)
    py1 = min(img_h, py1)

    new_top = py0
    new_bottom = py1
    new_right = px1 + 10  # small side padding
    new_left = max(0, px0 - 10)

    # --- Scan upward for title ---
    scan_top = max(0, py0 - pad_up_max)
    if scan_top < py0:
        scan_x0 = max(0, px0 - 20)
        scan_x1 = min(img_w, px1 + 20)
        region_above = page_image.crop((scan_x0, scan_top, scan_x1, py0))
        gray_above = np.array(region_above.convert("L"))
        result = _scan_for_content_boundary(gray_above, "up", gap_tolerance)
        if result is not None:
            new_top = max(0, scan_top + result - 6)

    # --- Scan downward for footnotes ---
    scan_bottom = min(img_h, py1 + pad_down_max)
    if scan_bottom > py1:
        scan_x0 = max(0, px0 - 20)
        scan_x1 = min(img_w, px1 + 20)
        region_below = page_image.crop((scan_x0, py1, scan_x1, scan_bottom))
        gray_below = np.array(region_below.convert("L"))
        result = _scan_for_content_boundary(gray_below, "down", gap_tolerance)
        if result is not None:
            new_bottom = min(img_h, py1 + result + 6)

    # --- Scan rightward for side annotations (wide tables only) ---
    table_width = px1 - px0
    is_wide = img_w > 0 and table_width / img_w >= 0.55
    if is_wide:
        scan_right = min(img_w, px1 + pad_right_max)
        if scan_right > px1:
            region_right = page_image.crop((px1, new_top, scan_right, new_bottom))
            gray_right = np.array(region_right.convert("L"))
            if gray_right.size > 0 and gray_right.shape[1] > 0:
                # Tight scan first
                r1 = _scan_columns_for_text(gray_right, large_gap=15)
                pass1_right = (r1 + 6) if r1 is not None else 10
                # Check for side footnote block beyond tight scan
                start_col = pass1_right + 10
                if start_col < gray_right.shape[1]:
                    remaining = gray_right[:, start_col:]
                    n_rows = remaining.shape[0]
                    text_rows = sum(1 for r in range(n_rows) if _row_has_text(remaining[r]))
                    table_h = new_bottom - new_top
                    if table_h > 0 and text_rows / table_h > 0.30:
                        r2 = _scan_columns_for_text(remaining, large_gap=40)
                        if r2 is not None:
                            pass1_right = start_col + r2 + 6
                new_right = min(img_w, px1 + pass1_right)

    new_right = min(img_w, new_right)

    # Convert back to normalized COCO (cx, cy, w, h)
    out_cx = (new_left + new_right) / 2 / img_w
    out_cy = (new_top + new_bottom) / 2 / img_h
    out_w = (new_right - new_left) / img_w
    out_h = (new_bottom - new_top) / img_h

    # Clamp to valid range
    out_cx = max(0, min(1, out_cx))
    out_cy = max(0, min(1, out_cy))
    out_w = max(0, min(1, out_w))
    out_h = max(0, min(1, out_h))

    return (out_cx, out_cy, out_w, out_h)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def prepare_data(
    data_dir: str,
    output_dir: str,
    max_pages: int | None = None,
    dpi: int = 200,
    val_ratio: float = 0.2,
    seed: int = 42,
    include_negatives: bool = False,
    negative_ratio: float = 0.3,
):
    """Prepare fine-tuning data from SCI-3000.
    
    Args:
        include_negatives: If True, also include pages without tables
            as negative examples (empty annotation targets).
        negative_ratio: Fraction of negative pages relative to positive
            pages to include (e.g. 0.3 = 30% as many negatives as positives).
    """
    from pdf2image import convert_from_path

    annot_dir = os.path.join(data_dir, "Annotations")
    pdf_dir = os.path.join(data_dir, "PDFs")

    if not os.path.isdir(annot_dir):
        print(f"Error: {annot_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Collect pages with tables AND pages without tables
    print("Scanning annotations...")
    annot_files = sorted(Path(annot_dir).glob("*.json"))

    pages_with_tables = []
    pages_without_tables = []
    for af in annot_files:
        parsed = parse_page_annotations(str(af))
        if parsed["tables"]:
            pages_with_tables.append((af, parsed))
        elif include_negatives:
            pages_without_tables.append((af, parsed))

    print(f"  Found {len(pages_with_tables)} pages with tables")
    if include_negatives:
        print(f"  Found {len(pages_without_tables)} pages without tables")

    if max_pages:
        random.seed(seed)
        random.shuffle(pages_with_tables)
        pages_with_tables = pages_with_tables[:max_pages]
        print(f"  Limited to {max_pages} positive pages")

    # Sample negative pages — prioritize pages with figures (hard negatives)
    negative_pages = []
    if include_negatives and pages_without_tables:
        n_neg = int(len(pages_with_tables) * negative_ratio)
        # Split into figure-containing (hard) and plain (easy) negatives
        hard_negatives = [p for p in pages_without_tables if p[1]["figures"]]
        easy_negatives = [p for p in pages_without_tables if not p[1]["figures"]]
        random.seed(seed + 1)
        random.shuffle(hard_negatives)
        random.shuffle(easy_negatives)
        # Take all hard negatives first, fill remainder with easy
        if len(hard_negatives) >= n_neg:
            negative_pages = hard_negatives[:n_neg]
        else:
            negative_pages = hard_negatives + easy_negatives[:n_neg - len(hard_negatives)]
        print(f"  Including {len(negative_pages)} negative examples "
              f"({len([p for p in negative_pages if p[1]['figures']])} with figures, "
              f"{negative_ratio:.0%} ratio)")

    # Combine all pages for processing
    all_pages = [(af, parsed, True) for af, parsed in pages_with_tables] + \
                [(af, parsed, False) for af, parsed in negative_pages]

    # Group by PDF for efficient rendering
    pdf_pages = {}
    for af, parsed, has_tables in all_pages:
        basename = af.stem
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
        pdf_pages[pdf_id].append((page_num, parsed, str(af), has_tables))

    print(f"  Spanning {len(pdf_pages)} PDFs")

    # Create output dirs
    train_img_dir = os.path.join(output_dir, "train", "images")
    train_ann_dir = os.path.join(output_dir, "train", "annotations")
    val_img_dir = os.path.join(output_dir, "val", "images")
    val_ann_dir = os.path.join(output_dir, "val", "annotations")
    for d in [train_img_dir, train_ann_dir, val_img_dir, val_ann_dir]:
        os.makedirs(d, exist_ok=True)

    # Split PDFs into train/val (split by PDF, not page, to avoid data leakage)
    pdf_ids = sorted(pdf_pages.keys())
    random.seed(seed)
    random.shuffle(pdf_ids)
    val_count = max(1, int(len(pdf_ids) * val_ratio))
    val_pdf_ids = set(pdf_ids[:val_count])
    train_pdf_ids = set(pdf_ids[val_count:])

    print(f"  Train PDFs: {len(train_pdf_ids)}, Val PDFs: {len(val_pdf_ids)}")

    # Process each PDF
    train_samples = 0
    val_samples = 0
    total_tables = 0
    negative_samples = 0
    skipped = 0

    for pdf_id in tqdm(pdf_ids, desc="Processing PDFs"):
        pdf_path = os.path.join(pdf_dir, f"{pdf_id}.pdf")
        page_list = pdf_pages[pdf_id]
        max_page = max(pn for pn, _, _, _ in page_list)
        is_val = pdf_id in val_pdf_ids

        try:
            pages = convert_from_path(
                pdf_path, dpi=dpi, first_page=1, last_page=max_page
            )
        except Exception as e:
            tqdm.write(f"  ⚠ Failed to render {pdf_id}: {e}")
            skipped += 1
            continue

        for page_num, parsed, annot_path, has_tables in page_list:
            if page_num > len(pages):
                continue

            page_image = pages[page_num - 1]
            img_w, img_h = page_image.size
            canvas_w, canvas_h = parsed["canvas_size"]

            if has_tables:
                # Build merged table+caption targets, then expand with pixel scanning
                targets = []
                for table_info in parsed["tables"]:
                    merged_bbox = merge_table_with_captions(
                        table_info, parsed["captions"], canvas_w, canvas_h
                    )
                    # Validate bbox
                    cx, cy, w, h = merged_bbox
                    if w <= 0 or h <= 0 or w > 1 or h > 1:
                        continue
                    # Expand bbox by scanning rendered image for footnotes
                    expanded_bbox = expand_bbox_on_image(
                        page_image, merged_bbox, canvas_w, canvas_h
                    )
                    ex, ey, ew, eh = expanded_bbox
                    if ew <= 0 or eh <= 0 or ew > 1 or eh > 1:
                        expanded_bbox = merged_bbox  # fallback
                    targets.append({
                        "bbox": list(expanded_bbox),  # [cx, cy, w, h] normalized
                        "category_id": 0,  # 0 = table_with_caption
                    })

                if not targets:
                    continue

                total_tables += len(targets)
            else:
                # Negative example: no tables on this page
                targets = []
                negative_samples += 1

            # Determine split
            if is_val:
                img_dir = val_img_dir
                ann_dir = val_ann_dir
                val_samples += 1
            else:
                img_dir = train_img_dir
                ann_dir = train_ann_dir
                train_samples += 1

            # Save image
            page_id = Path(annot_path).stem
            img_path = os.path.join(img_dir, f"{page_id}.png")
            page_image.save(img_path, "PNG")

            # Save annotation
            ann_data = {
                "image_id": page_id,
                "image_size": [img_w, img_h],
                "canvas_size": [canvas_w, canvas_h],
                "annotations": targets,
            }
            ann_path = os.path.join(ann_dir, f"{page_id}.json")
            with open(ann_path, "w") as f:
                json.dump(ann_data, f)

    # Save metadata
    metadata = {
        "num_classes": 1,
        "class_names": ["table_with_caption"],
        "train_samples": train_samples,
        "val_samples": val_samples,
        "total_tables": total_tables,
        "negative_samples": negative_samples,
        "dpi": dpi,
        "skipped_pdfs": skipped,
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Dataset prepared!")
    print(f"  Train samples: {train_samples} ({negative_samples} negative)")
    print(f"  Val samples:   {val_samples}")
    print(f"  Total tables:  {total_tables}")
    print(f"  Skipped PDFs:  {skipped}")
    print(f"  Output: {output_dir}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Prepare SCI-3000 for fine-tuning")
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help=f"SCI-3000 directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-pages", type=int, default=None,
        help="Limit to N pages (for testing)",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="DPI for rendering (default: 200)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--include-negatives", action="store_true",
        help="Include pages without tables as negative examples",
    )
    parser.add_argument(
        "--negative-ratio", type=float, default=0.3,
        help="Ratio of negative pages to positive pages (default: 0.3)",
    )
    args = parser.parse_args()

    prepare_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_pages=args.max_pages,
        dpi=args.dpi,
        val_ratio=args.val_ratio,
        include_negatives=args.include_negatives,
        negative_ratio=args.negative_ratio,
    )


if __name__ == "__main__":
    main()
