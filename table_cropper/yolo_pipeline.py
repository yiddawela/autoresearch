"""
YOLO26 training pipeline for table detection experiments.

Converts SCI-3000 finetune data from COCO format to YOLO format,
trains YOLO26 models for each formulation, and exports results.

Formulations:
  A: Table-only (single class: table body)
  B: Merged crop (single class: table+caption merged box)
  C: Multi-class (two classes: table, caption)

Usage:
    uv run yolo_pipeline.py prepare --formulation A
    uv run yolo_pipeline.py train --formulation A --epochs 50
    uv run yolo_pipeline.py evaluate --formulation A
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

DEFAULT_DATA_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "table_cropper", "SCI-3000"
)
FINETUNE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "table_cropper", "finetune_data"
)
YOLO_DATA_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "table_cropper", "yolo_data"
)


# ---------------------------------------------------------------------------
# SCI-3000 annotation parsing
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
    tables, captions = [], []
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
        elif ann_type == "Caption":
            captions.append(info)
    return {"canvas_size": (canvas_w, canvas_h), "tables": tables, "captions": captions}


def _xywh_to_yolo_norm(bbox_xywh: tuple, canvas_w: int, canvas_h: int,
                        img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert (x, y, w, h) in canvas coords to YOLO normalized (cx, cy, w, h)."""
    x, y, w, h = bbox_xywh
    # Scale to image coordinates
    sx = img_w / canvas_w if canvas_w > 0 else 1.0
    sy = img_h / canvas_h if canvas_h > 0 else 1.0
    px, py, pw, ph = x * sx, y * sy, w * sx, h * sy
    # Convert to normalized center format
    cx = (px + pw / 2) / img_w
    cy = (py + ph / 2) / img_h
    nw = pw / img_w
    nh = ph / img_h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, nw)),
        max(0.0, min(1.0, nh)),
    )


def _merge_boxes_xywh(boxes: list[tuple]) -> tuple:
    """Merge multiple (x, y, w, h) boxes into one encompassing box."""
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


# ---------------------------------------------------------------------------
# Data preparation: convert to YOLO format
# ---------------------------------------------------------------------------


def prepare_yolo_data(
    formulation: str,
    data_dir: str = DEFAULT_DATA_DIR,
    output_dir: str = YOLO_DATA_DIR,
):
    """Convert SCI-3000 annotations to YOLO format for a given formulation."""
    annot_dir = os.path.join(data_dir, "Annotations")

    # Use pre-rendered images from finetune data
    for split in ["train", "val"]:
        img_dir = os.path.join(FINETUNE_DIR, split, "images")
        if not os.path.isdir(img_dir):
            print(f"  ⚠ Missing {split} images at {img_dir}")
            continue

        # Create output dirs
        form_dir = os.path.join(output_dir, formulation)
        out_img_dir = os.path.join(form_dir, split, "images")
        out_lbl_dir = os.path.join(form_dir, split, "labels")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        page_ids = [Path(f).stem for f in os.listdir(img_dir) if f.endswith(".png")]
        converted = 0

        for pid in tqdm(page_ids, desc=f"Converting {split} ({formulation})"):
            img_path = os.path.join(img_dir, f"{pid}.png")
            ann_path = os.path.join(annot_dir, f"{pid}.json")

            if not os.path.exists(ann_path):
                continue

            parsed = parse_page_annotations(ann_path)
            if not parsed["tables"]:
                continue

            # Get image dimensions
            with Image.open(img_path) as img:
                img_w, img_h = img.size

            canvas_w, canvas_h = parsed["canvas_size"]
            labels = []

            if formulation == "A":
                # Table-only: class 0 = table body
                for table in parsed["tables"]:
                    cx, cy, w, h = _xywh_to_yolo_norm(
                        table["bbox"], canvas_w, canvas_h, img_w, img_h
                    )
                    labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            elif formulation == "B":
                # Merged: class 0 = table+caption merged box
                for table in parsed["tables"]:
                    boxes_to_merge = [table["bbox"]]
                    table_id = table["id"]
                    for cap in parsed["captions"]:
                        if cap["parent_id"] == table_id:
                            boxes_to_merge.append(cap["bbox"])
                    merged = _merge_boxes_xywh(boxes_to_merge)
                    cx, cy, w, h = _xywh_to_yolo_norm(
                        merged, canvas_w, canvas_h, img_w, img_h
                    )
                    labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            elif formulation == "C":
                # Multi-class: class 0 = table, class 1 = caption
                for table in parsed["tables"]:
                    cx, cy, w, h = _xywh_to_yolo_norm(
                        table["bbox"], canvas_w, canvas_h, img_w, img_h
                    )
                    labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                for cap in parsed["captions"]:
                    # Only include captions linked to tables
                    if cap["parent_id"] and any(
                        t["id"] == cap["parent_id"] for t in parsed["tables"]
                    ):
                        cx, cy, w, h = _xywh_to_yolo_norm(
                            cap["bbox"], canvas_w, canvas_h, img_w, img_h
                        )
                        labels.append(f"1 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if labels:
                # Hardlink image (robust against source file replacement)
                out_img = os.path.join(out_img_dir, f"{pid}.png")
                if not os.path.exists(out_img):
                    if os.path.exists(img_path):
                        os.link(img_path, out_img)
                    else:
                        continue  # Skip if source image missing

                # Write label file
                out_lbl = os.path.join(out_lbl_dir, f"{pid}.txt")
                with open(out_lbl, "w") as f:
                    f.write("\n".join(labels) + "\n")
                converted += 1

        print(f"  {split}: converted {converted}/{len(page_ids)} pages")

    # Write YOLO dataset YAML
    form_dir = os.path.join(output_dir, formulation)
    nc = 1 if formulation in ("A", "B") else 2
    names = ["table"] if formulation in ("A", "B") else ["table", "caption"]

    yaml_content = f"""# YOLO26 dataset config for Formulation {formulation}
path: {os.path.abspath(form_dir)}
train: train/images
val: val/images

nc: {nc}
names: {names}
"""
    yaml_path = os.path.join(form_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n  Dataset YAML: {yaml_path}")
    return yaml_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_yolo(
    formulation: str,
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    model_size: str = "m",
    output_dir: str = YOLO_DATA_DIR,
):
    """Train YOLO26 model for a given formulation."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: uv add ultralytics")
        return None

    yaml_path = os.path.join(output_dir, formulation, "dataset.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found. Run 'prepare' first.")
        return None

    # Load pre-trained YOLO26
    model_name = f"yolo26{model_size}.pt"
    print(f"Loading {model_name}...")
    model = YOLO(model_name)

    # Train
    project_dir = os.path.join(output_dir, "runs")
    run_name = f"formulation_{formulation}"

    print(f"\nTraining Formulation {formulation}:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Model: {model_name}")

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        device="mps",  # Apple Silicon
        patience=10,
        save=True,
        plots=True,
    )

    print(f"\nTraining complete. Results saved to {project_dir}/{run_name}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="YOLO26 table detection pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # Prepare
    p_prepare = sub.add_parser("prepare", help="Convert SCI-3000 to YOLO format")
    p_prepare.add_argument("--formulation", required=True, choices=["A", "B", "C"])
    p_prepare.add_argument("--data-dir", default=DEFAULT_DATA_DIR)

    # Train
    p_train = sub.add_parser("train", help="Train YOLO26 model")
    p_train.add_argument("--formulation", required=True, choices=["A", "B", "C"])
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--img-size", type=int, default=640)
    p_train.add_argument("--model-size", default="m", choices=["n", "s", "m", "l", "x"])

    args = parser.parse_args()

    if args.command == "prepare":
        print(f"Preparing YOLO data for Formulation {args.formulation}...")
        prepare_yolo_data(args.formulation, args.data_dir)

    elif args.command == "train":
        print(f"Training YOLO26 for Formulation {args.formulation}...")
        train_yolo(
            args.formulation,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            model_size=args.model_size,
        )


if __name__ == "__main__":
    main()
