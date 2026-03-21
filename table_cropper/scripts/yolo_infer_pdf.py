"""Run YOLO inference on PDFs and save cropped table images.

Usage:
    uv run python scripts/yolo_infer_pdf.py \\
        --input-dir data/72_included_articles \\
        --output-dir data/72_included_articles/yolo_crops \\
        --model-path ~/.cache/table_cropper/yolo_data/runs/formulation_B/weights/best.pt
"""
import argparse
import os
import sys
from pathlib import Path

from pdf2image import convert_from_path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO table crop inference on PDFs")
    parser.add_argument("--input-dir", required=True, help="Directory with PDFs")
    parser.add_argument("--output-dir", required=True, help="Directory to save crops")
    parser.add_argument("--model-path", required=True, help="Path to YOLO best.pt")
    parser.add_argument("--dpi", type=int, default=200, help="PDF rendering DPI")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading YOLO model from {args.model_path}...")
    model = YOLO(args.model_path)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s)")
    total = 0

    for pdf_path in pdf_files:
        print(f"  Processing {pdf_path.name}...")
        try:
            pages = convert_from_path(pdf_path, dpi=args.dpi)
        except Exception as e:
            print(f"    Error: {e}")
            continue

        for i, page in enumerate(pages):
            results = model(page, conf=args.conf, verbose=False)
            boxes = results[0].boxes
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cropped = page.crop((x1, y1, x2, y2))
                out_name = f"{pdf_path.stem}_page{i+1:03d}_table{j+1:02d}.png"
                cropped.save(output_dir / out_name)
                total += 1

        print(f"    {sum(len(model(p, conf=args.conf, verbose=False)[0].boxes) for p in convert_from_path(pdf_path, dpi=args.dpi))} tables" if False else "")

    print(f"\nDone! {total} table(s) saved to {output_dir}")


if __name__ == "__main__":
    main()
