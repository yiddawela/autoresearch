"""
Crop tables (with titles and captions) from scientific journal PDFs.

Uses Microsoft Table Transformer (TATR) pre-trained on PubTables-1M for
table detection, with layout-aware crop refinement to capture titles and
footnotes/captions.

Usage:
    uv run crop_tables.py --input-dir /path/to/pdfs --output-dir ./output
    uv run crop_tables.py --input-dir /path/to/pdfs --output-dir ./output --dpi 300 --confidence 0.7
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    DetrImageProcessor,
    TableTransformerForObjectDetection,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_NAME = "microsoft/table-transformer-detection"


def load_model(device: torch.device, model_path: str | None = None):
    """Load TATR detection model and image processor.
    
    Args:
        device: Target device.
        model_path: Path to fine-tuned checkpoint directory. If None, uses
                    pre-trained microsoft/table-transformer-detection.
    """
    source = model_path or MODEL_NAME
    processor = DetrImageProcessor.from_pretrained(source if not model_path else MODEL_NAME)
    model = TableTransformerForObjectDetection.from_pretrained(source)
    model.to(device)
    model.eval()
    if model_path:
        # Also try loading processor from checkpoint if available
        try:
            processor = DetrImageProcessor.from_pretrained(model_path)
        except Exception:
            pass
    return processor, model


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Table detection
# ---------------------------------------------------------------------------

# TATR detection labels: 0 = "table", 1 = "table rotated"
TABLE_LABELS = {0, 1}  # include both standard and rotated


def detect_tables(
    image: Image.Image,
    processor: DetrImageProcessor,
    model: TableTransformerForObjectDetection,
    device: torch.device,
    confidence_threshold: float = 0.5,
) -> list[dict]:
    """
    Run table detection on a single page image.
    Returns list of dicts with keys: bbox (x0, y0, x1, y1), score, label.
    """
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Convert outputs to absolute coordinates
    target_sizes = torch.tensor([image.size[::-1]], device=device)  # (H, W)
    results = processor.post_process_object_detection(
        outputs, threshold=confidence_threshold, target_sizes=target_sizes
    )[0]

    detections = []
    for score, label, box in zip(
        results["scores"].cpu().tolist(),
        results["labels"].cpu().tolist(),
        results["boxes"].cpu().tolist(),
    ):
        if label in TABLE_LABELS:
            detections.append(
                {
                    "bbox": box,  # [x0, y0, x1, y1]
                    "score": score,
                    "label": label,
                    "label_name": "table" if label == 0 else "table_rotated",
                }
            )

    return detections


# ---------------------------------------------------------------------------
# Duplicate suppression (IoU-based NMS)
# ---------------------------------------------------------------------------


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two [x0, y0, x1, y1] boxes."""
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])

    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def suppress_duplicates(
    detections: list[dict], iou_threshold: float = 0.5
) -> list[dict]:
    """Remove overlapping detections, keeping highest confidence.
    Also suppresses fragments that are contained within or adjacent to larger detections."""
    if len(detections) <= 1:
        return detections

    # Sort by area descending (largest first), then by score
    def sort_key(d):
        b = d["bbox"]
        return -((b[2] - b[0]) * (b[3] - b[1])), -d["score"]

    detections = sorted(detections, key=sort_key)
    kept = []

    for det in detections:
        bbox = det["bbox"]
        det_w = bbox[2] - bbox[0]
        det_h = bbox[3] - bbox[1]
        det_area = det_w * det_h
        is_duplicate = False
        for ki, kept_det in enumerate(kept):
            kb = kept_det["bbox"]
            kept_w = kb[2] - kb[0]
            kept_h = kb[3] - kb[1]
            kept_area = kept_w * kept_h

            # Standard IoU suppression
            iou = compute_iou(bbox, kb)
            if iou > iou_threshold:
                is_duplicate = True
                break

            # Containment: this detection is >60% inside a kept one
            ix0 = max(bbox[0], kb[0])
            iy0 = max(bbox[1], kb[1])
            ix1 = min(bbox[2], kb[2])
            iy1 = min(bbox[3], kb[3])
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            if det_area > 0 and inter / det_area > 0.6:
                is_duplicate = True
                break

            # Spatial adjacency: smaller detection adjacent to (within 5% gap of)
            # a larger one, sharing significant vertical overlap (>50%)
            if det_area < kept_area * 0.8:  # smaller than the kept one
                gap_threshold = max(kept_w, det_w) * 0.05
                # Horizontal adjacency (side-by-side)
                h_gap = max(0, bbox[0] - kb[2], kb[0] - bbox[2])
                # Vertical overlap
                v_overlap = max(0, min(bbox[3], kb[3]) - max(bbox[1], kb[1]))
                v_span = min(det_h, kept_h)
                if h_gap < gap_threshold and v_span > 0 and v_overlap / v_span > 0.5:
                    # Merge: expand the kept detection to include this one
                    kept[ki] = {
                        **kept_det,
                        "bbox": [
                            min(bbox[0], kb[0]),
                            min(bbox[1], kb[1]),
                            max(bbox[2], kb[2]),
                            max(bbox[3], kb[3]),
                        ],
                    }
                    is_duplicate = True
                    break

        if not is_duplicate:
            kept.append(det)

    return kept


# ---------------------------------------------------------------------------
# Layout-aware crop refinement
# ---------------------------------------------------------------------------


def _row_has_text(row: np.ndarray, dark_threshold: int = 150) -> bool:
    """
    Check if a grayscale row contains text.
    A row has text if it contains enough dark pixels (below threshold).
    Uses a count-based approach: at least 3 pixels must be dark,
    which handles both dense table text and sparse footnote text.
    """
    dark_pixels = np.sum(row < dark_threshold)
    return dark_pixels >= 3


def _scan_for_content_boundary(
    gray_region: np.ndarray,
    direction: str,
    large_gap: int = 60,
) -> int:
    """
    Scan a grayscale region for content boundaries using pixel-level
    dark pixel detection.

    A row is considered to have content if it has >= 3 pixels darker
    than 150/255. This catches even sparse footnote text reliably.

    Uses a single-tier gap strategy:
    - Tolerates gaps smaller than large_gap (handles gaps between
      table body and footnotes, between footnote lines, etc.)
    - Stops at gaps >= large_gap (transition to next section)

    For 'down': scans top-to-bottom, returns offset of last content row.
    For 'up': scans bottom-to-top, returns offset of first content row.
    """
    n_rows = gray_region.shape[0]

    if direction == "down":
        indices = range(n_rows)
    else:  # up
        indices = range(n_rows - 1, -1, -1)

    last_content_pos = None
    gap_count = 0
    found_any_text = False

    for i in indices:
        if _row_has_text(gray_region[i]):
            found_any_text = True
            gap_count = 0
            last_content_pos = i
        else:
            if found_any_text:
                gap_count += 1
                if gap_count >= large_gap:
                    break

    return last_content_pos


def refine_crop(
    image: Image.Image,
    bbox: list[float],
    pad_top_max: int = 250,
    pad_bottom_max: int = 400,
    pad_side: int = 20,
    large_gap: int = 60,
) -> tuple[int, int, int, int]:
    """
    Expand bbox to include title (above) and captions/footnotes (below)
    using dark-pixel scanning within the table's column bounds.

    Detects text rows by checking for dark pixels (>= 3 pixels below 150/255).
    Tolerates gaps up to large_gap pixels (handles whitespace between table
    body and footnotes). Stops at gaps >= large_gap.

    Returns (x0, y0, x1, y1) in pixel coordinates, clamped to image bounds.
    """
    img_w, img_h = image.size
    bx0, by0, bx1, by1 = [int(round(c)) for c in bbox]

    # Horizontal scan bounds: use table's column extent with margin
    col_margin = 40
    scan_x0 = max(0, bx0 - col_margin)
    scan_x1 = min(img_w, bx1 + col_margin)

    # --- Upward scan for title ---
    scan_top = max(0, by0 - pad_top_max)
    new_top = by0

    if scan_top < by0:
        region_above = image.crop((scan_x0, scan_top, scan_x1, by0))
        gray_above = np.array(region_above.convert("L"))

        result = _scan_for_content_boundary(gray_above, "up", large_gap)
        if result is not None:
            new_top = max(0, scan_top + result - 8)  # 8px margin

    # --- Downward scan for captions/footnotes ---
    scan_bottom = min(img_h, by1 + pad_bottom_max)
    new_bottom = by1

    if scan_bottom > by1:
        region_below = image.crop((scan_x0, by1, scan_x1, scan_bottom))
        gray_below = np.array(region_below.convert("L"))

        result = _scan_for_content_boundary(gray_below, "down", large_gap)
        if result is not None:
            new_bottom = min(img_h, by1 + result + 8)  # 8px margin

    # --- Side padding ---
    x0 = max(0, bx0 - pad_side)
    x1 = min(img_w, bx1 + pad_side)

    return (x0, new_top, x1, new_bottom)


def _col_has_text(col: np.ndarray, dark_threshold: int = 150) -> bool:
    """Check if a grayscale column contains text (≥3 dark pixels)."""
    return np.sum(col < dark_threshold) >= 3


def _scan_columns_for_text(
    gray_region: np.ndarray, large_gap: int = 40
) -> int | None:
    """Scan left-to-right for text columns. Returns offset of last text column."""
    n_cols = gray_region.shape[1]
    last_content_col = None
    gap_count = 0
    found_any = False

    for c in range(n_cols):
        if _col_has_text(gray_region[:, c]):
            found_any = True
            gap_count = 0
            last_content_col = c
        elif found_any:
            gap_count += 1
            if gap_count >= large_gap:
                break

    return last_content_col


def extend_to_captions(
    image: Image.Image,
    bbox: list[float],
    pad_up_max: int = 60,
    pad_down_max: int = 40,
    pad_right_max: int = 200,
    pad_side: int = 10,
    gap_tolerance: int = 25,
) -> tuple[int, int, int, int]:
    """
    Extend bbox outward to include nearby caption/footnote text.

    For fine-tuned models that detect tight table bodies, this scans:
    - Upward (up to pad_up_max px) for table titles
    - Downward (up to pad_down_max px) for footnotes/captions below
    - Rightward (up to pad_right_max px) for side captions/footnotes

    Uses dark-pixel scanning (same as refine_crop) with a conservative
    gap tolerance to avoid pulling in unrelated content.

    Returns (x0, y0, x1, y1) in pixel coordinates.
    """
    img_w, img_h = image.size
    bx0, by0, bx1, by1 = [int(round(c)) for c in bbox]

    # Small left padding
    new_left = max(0, bx0 - pad_side)

    # --- Scan upward for title ---
    scan_top = max(0, by0 - pad_up_max)
    new_top = by0
    if scan_top < by0:
        scan_x0 = max(0, bx0 - 20)
        scan_x1 = min(img_w, bx1 + 20)
        region_above = image.crop((scan_x0, scan_top, scan_x1, by0))
        gray_above = np.array(region_above.convert("L"))
        result = _scan_for_content_boundary(gray_above, "up", gap_tolerance)
        if result is not None:
            new_top = max(0, scan_top + result - 6)

    # --- Scan downward for captions/footnotes ---
    scan_bottom = min(img_h, by1 + pad_down_max)
    new_bottom = by1
    if scan_bottom > by1:
        scan_x0 = max(0, bx0 - 20)
        scan_x1 = min(img_w, bx1 + 20)
        region_below = image.crop((scan_x0, by1, scan_x1, scan_bottom))
        gray_below = np.array(region_below.convert("L"))
        result = _scan_for_content_boundary(gray_below, "down", gap_tolerance)
        if result is not None:
            new_bottom = min(img_h, by1 + result + 6)

    # --- Scan rightward for side captions ---
    # Two-pass approach:
    # Pass 1: Tight gap (15px) captures immediately adjacent text
    # Pass 2: If a gap is found, look for a text block further right
    #   that spans ≥30% of the table height (side footnote block)
    scan_right = min(img_w, bx1 + pad_right_max)
    new_right = bx1 + pad_side
    if scan_right > bx1:
        table_height = new_bottom - new_top
        region_right = image.crop((bx1, new_top, scan_right, new_bottom))
        gray_right = np.array(region_right.convert("L"))
        if gray_right.size > 0 and gray_right.shape[1] > 0:
            # Pass 1: tight scan (15px gap tolerance)
            result = _scan_columns_for_text(gray_right, large_gap=15)
            if result is not None:
                pass1_right = result + 6
            else:
                pass1_right = pad_side

            # Pass 2: look for side footnote block beyond the tight scan
            # Only for wide tables (≥55% page width) — narrow tables in
            # multi-column layouts have body text in adjacent columns
            table_width = bx1 - bx0
            is_wide_table = img_w > 0 and table_width / img_w >= 0.55
            start_col = pass1_right + 10  # skip past any gap
            if is_wide_table and start_col < gray_right.shape[1]:
                remaining = gray_right[:, start_col:]
                # Count text rows in the remaining region
                n_rows = remaining.shape[0]
                text_row_count = 0
                for r in range(n_rows):
                    if _row_has_text(remaining[r]):
                        text_row_count += 1
                # If ≥30% of the table height has text in this region,
                # it's likely a side footnote block
                if table_height > 0 and text_row_count / table_height > 0.30:
                    # Find the rightmost text column in this block
                    result2 = _scan_columns_for_text(remaining, large_gap=40)
                    if result2 is not None:
                        pass1_right = start_col + result2 + 6

            new_right = min(img_w, bx1 + pass1_right)
    new_right = min(img_w, new_right)

    return (new_left, new_top, new_right, new_bottom)


def process_pdf(
    pdf_path: str,
    output_dir: str,
    processor: DetrImageProcessor,
    model: TableTransformerForObjectDetection,
    device: torch.device,
    dpi: int = 200,
    confidence: float = 0.5,
    use_refine: bool = True,
) -> list[str]:
    """
    Process a single PDF: convert pages to images, detect tables,
    optionally refine crops, save results.
    Returns list of saved file paths.
    """
    from pdf2image import convert_from_path

    pdf_name = Path(pdf_path).stem
    saved_files = []

    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        print(f"  ⚠ Failed to convert {pdf_path}: {e}", file=sys.stderr)
        return saved_files

    table_counter = 0

    for page_idx, page_image in enumerate(pages):
        img_w, img_h = page_image.size
        page_area = img_w * img_h

        # Detect tables on this page
        detections = detect_tables(
            page_image, processor, model, device, confidence
        )
        detections = suppress_duplicates(detections)

        for det in detections:
            bbox = det["bbox"]

            # Post-detection filters
            det_w = bbox[2] - bbox[0]
            det_h = bbox[3] - bbox[1]
            det_area = det_w * det_h

            # Skip if detection covers >80% of page (whole-page false positive)
            if det_area > 0.80 * page_area:
                continue
            # Skip if detection is tiny (<1% of page)
            if det_area < 0.01 * page_area:
                continue
            # Skip extreme aspect ratios (>4:1 width:height — likely a header bar)
            if det_h > 0 and det_w / det_h > 4.0:
                continue
            # ── False-positive filters (fine-tuned model only) ──
            if not use_refine:
                crop_region = page_image.crop(
                    (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                )
                gray_crop = np.array(crop_region.convert("L"))
                crop_h, crop_w = gray_crop.shape[:2]

                if crop_h > 50:
                    # 1. Text density
                    text_rows = sum(
                        1 for r in range(crop_h)
                        if _row_has_text(gray_crop[r])
                    )
                    text_density = text_rows / crop_h

                    # 2. Count horizontal rules (dark lines ≥40% width)
                    min_rule_span = int(crop_w * 0.40)
                    rule_count = 0
                    prev_rule_row = -10
                    for r in range(crop_h):
                        dark_px = int(np.sum(gray_crop[r] < 80))
                        # Must be thin (rule) not thick (image bar):
                        # a rule spans wide but row above/below is NOT dark
                        if dark_px >= min_rule_span:
                            if r - prev_rule_row > 3:  # don't double-count
                                rule_count += 1
                                prev_rule_row = r
                    has_table_rules = rule_count >= 2  # need ≥2 rules

                    # 3. Detect flow diagrams (vertical lines / box borders)
                    has_vertical_lines = False
                    if crop_h > 100:
                        vert_threshold = int(crop_h * 0.15)
                        for c in range(10, crop_w - 10):
                            col = gray_crop[:, c]
                            dark_col_px = int(np.sum(col < 80))
                            if dark_col_px >= vert_threshold:
                                has_vertical_lines = True
                                break

                    # 4. Detect structured abstracts (bold section headers
                    #    like "METHODS", "RESULTS" in all-caps)
                    is_abstract_like = False
                    if bbox[1] < img_h * 0.35:  # top third
                        # Check if there are many all-caps bold headings
                        # These show as short but very dark text rows
                        # surrounded by dense text (~abstract structure)
                        if text_density > 0.55:
                            # Additional check: if no table rules, and
                            # detection is in first page top area, likely abstract
                            if not has_table_rules and bbox[1] < img_h * 0.15:
                                is_abstract_like = True

                    # Apply filters
                    # a) Dense text without table rules → reject
                    if not has_table_rules and text_density > 0.70:
                        continue
                    # b) Flow diagram (vertical lines + boxes) → reject
                    if has_vertical_lines and not has_table_rules and text_density < 0.35:
                        continue
                    # c) Abstract at top of page → reject
                    if is_abstract_like:
                        continue

            table_counter += 1

            if use_refine:
                # Refine crop to include title + captions (for pre-trained model)
                crop_box = refine_crop(page_image, bbox)
            else:
                # Fine-tuned model: extend crop to include nearby captions/footnotes
                # Scan outward from detected bbox for text content
                crop_box = extend_to_captions(page_image, bbox)

            # Crop and save
            cropped = page_image.crop(crop_box)
            filename = f"{pdf_name}_page{page_idx + 1:03d}_table{table_counter:02d}.png"
            filepath = os.path.join(output_dir, filename)
            cropped.save(filepath, "PNG")
            saved_files.append(filepath)

    return saved_files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Crop tables from scientific journal PDFs"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory to save cropped table images (default: ./output)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF-to-image conversion (default: 200)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for table detection (default: 0.5)",
    )
    parser.add_argument(
        "--single-pdf",
        default=None,
        help="Process only this single PDF file (overrides --input-dir)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to fine-tuned model checkpoint (default: pre-trained TATR)",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        default=None,
        help="Disable heuristic crop refinement (auto-enabled with --model-path)",
    )
    args = parser.parse_args()

    # Auto-disable refine when using fine-tuned model
    use_refine = True
    if args.no_refine is not None:
        use_refine = not args.no_refine
    elif args.model_path:
        use_refine = False

    # Collect PDF files
    if args.single_pdf:
        pdf_files = [args.single_pdf]
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"Error: {args.input_dir} is not a directory", file=sys.stderr)
            sys.exit(1)
        pdf_files = sorted(str(p) for p in input_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s)")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading Table Transformer model...")
    device = get_device()
    print(f"  Device: {device}")
    processor, model = load_model(device, model_path=args.model_path)
    print(f"  Model loaded ✓ ({'fine-tuned' if args.model_path else 'pre-trained'})")

    # Process each PDF
    total_tables = 0
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        saved = process_pdf(
            pdf_path,
            args.output_dir,
            processor,
            model,
            device,
            dpi=args.dpi,
            confidence=args.confidence,
            use_refine=use_refine,
        )
        total_tables += len(saved)
        if saved:
            tqdm.write(f"  {Path(pdf_path).name}: {len(saved)} table(s)")
        else:
            tqdm.write(f"  {Path(pdf_path).name}: no tables detected")

    print(f"\nDone! {total_tables} table(s) saved to {args.output_dir}")


if __name__ == "__main__":
    main()
