"""
PubTables-1M dataset adapter for cross-dataset transfer experiments.

Downloads a subset of PubTables-1M from HuggingFace and converts
PASCAL VOC XML annotations to our evaluation format.

PubTables-1M has:
  - 575,305 table detection images (from PubMed PDFs)
  - PASCAL VOC XML annotations with classes: 'table', 'table rotated'
  - NO caption annotations (table-body only)

For our experiments, we use PubTables-1M ONLY for cross-dataset
localisation transfer — testing whether SCI-3000-trained models
generalise to a different document distribution.

Usage:
    uv run pubtables_adapter.py download --n-samples 500
    uv run pubtables_adapter.py convert --format yolo
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

PUBTABLES_CACHE = os.path.join(
    os.path.expanduser("~"), ".cache", "table_cropper", "pubtables1m"
)


# ---------------------------------------------------------------------------
# Download from HuggingFace
# ---------------------------------------------------------------------------


def download_pubtables(
    n_samples: int = 500,
    output_dir: str = PUBTABLES_CACHE,
    split: str = "val",
):
    """Download a subset of PubTables-1M detection data from HuggingFace.

    The HF dataset provides PASCAL VOC XML annotations as bytes via
    webdataset. We need a different approach: use the Microsoft Research
    direct download for the test set (much smaller than full train).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' not installed. Run: uv add datasets")
        return

    img_dir = os.path.join(output_dir, split, "images")
    ann_dir = os.path.join(output_dir, split, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    print(f"Loading {n_samples} PubTables-1M annotations from HuggingFace...")

    # HF webdataset: provides xml bytes + key per sample
    ds = load_dataset("bsmock/pubtables-1m", split="train", streaming=True)

    count = 0
    for sample in tqdm(ds, total=n_samples, desc="Processing"):
        if count >= n_samples:
            break

        xml_bytes = sample.get("xml", b"")
        key = sample.get("__key__", f"sample_{count}")
        image_id = key.replace("/", "_")

        if not xml_bytes:
            continue

        # Parse PASCAL VOC XML
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError:
            continue

        size_el = root.find("size")
        if size_el is None:
            continue
        width = int(size_el.find("width").text)
        height = int(size_el.find("height").text)

        tables = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            bbox_el = obj.find("bndbox")
            x0 = float(bbox_el.find("xmin").text)
            y0 = float(bbox_el.find("ymin").text)
            x1 = float(bbox_el.find("xmax").text)
            y1 = float(bbox_el.find("ymax").text)
            tables.append({
                "bbox": [x0, y0, x1, y1],
                "category": name,
            })

        if not tables:
            continue

        # Save annotation
        ann = {
            "image_id": image_id,
            "image_size": [width, height],
            "tables": tables,
            "source_filename": root.find("filename").text if root.find("filename") is not None else "",
        }
        ann_path = os.path.join(ann_dir, f"{image_id}.json")
        with open(ann_path, "w") as fp:
            json.dump(ann, fp)

        count += 1

    print(f"\n  Saved {count} annotations to {ann_dir}/")
    print(f"  NOTE: Images require separate download from PubTables-1M.")
    print(f"  For cross-dataset evaluation, we evaluate using annotations only")
    print(f"  against models that produce predictions on SCI-3000-style inputs.")
    return output_dir


# ---------------------------------------------------------------------------
# Convert to YOLO format for training/evaluation
# ---------------------------------------------------------------------------


def convert_to_yolo(
    input_dir: str = PUBTABLES_CACHE,
    split: str = "val",
):
    """Convert downloaded PubTables-1M annotations to YOLO format."""
    ann_dir = os.path.join(input_dir, split, "annotations")
    img_dir = os.path.join(input_dir, split, "images")
    yolo_lbl_dir = os.path.join(input_dir, split, "labels")
    os.makedirs(yolo_lbl_dir, exist_ok=True)

    ann_files = sorted(Path(ann_dir).glob("*.json"))
    converted = 0

    for af in tqdm(ann_files, desc=f"Converting {split} to YOLO"):
        with open(af) as f:
            ann = json.load(f)

        img_w, img_h = ann["image_size"]
        labels = []

        for table in ann["tables"]:
            x0, y0, x1, y1 = table["bbox"]
            cx = ((x0 + x1) / 2) / img_w
            cy = ((y0 + y1) / 2) / img_h
            w = (x1 - x0) / img_w
            h = (y1 - y0) / img_h

            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w = max(0, min(1, w))
            h = max(0, min(1, h))

            labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if labels:
            lbl_path = os.path.join(yolo_lbl_dir, f"{af.stem}.txt")
            with open(lbl_path, "w") as f:
                f.write("\n".join(labels) + "\n")
            converted += 1

    print(f"  Converted {converted}/{len(ann_files)} to YOLO format")

    # Write dataset YAML
    yaml_path = os.path.join(input_dir, "dataset.yaml")
    yaml_content = f"""# PubTables-1M transfer evaluation dataset
path: {os.path.abspath(input_dir)}
train: {split}/images
val: {split}/images

nc: 1
names: ['table']
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"  YAML: {yaml_path}")


# ---------------------------------------------------------------------------
# Load for experiment_runner evaluation
# ---------------------------------------------------------------------------


def load_pubtables_for_eval(
    input_dir: str = PUBTABLES_CACHE,
    split: str = "val",
    max_samples: int | None = None,
) -> list[dict]:
    """Load PubTables-1M samples for evaluation with experiment_runner.

    Returns list of dicts with:
        image_path: path to PNG
        image_id: unique identifier
        gt_tables: list of (x0, y0, x1, y1) table boxes
    """
    ann_dir = os.path.join(input_dir, split, "annotations")
    img_dir = os.path.join(input_dir, split, "images")

    ann_files = sorted(Path(ann_dir).glob("*.json"))
    if max_samples:
        ann_files = ann_files[:max_samples]

    samples = []
    for af in ann_files:
        with open(af) as f:
            ann = json.load(f)

        image_id = ann["image_id"]
        img_path = os.path.join(img_dir, f"{image_id}.png")
        if not os.path.exists(img_path):
            continue

        gt_tables = [tuple(t["bbox"]) for t in ann["tables"]]
        samples.append({
            "image_path": img_path,
            "image_id": image_id,
            "gt_tables": gt_tables,
            "gt_captions": [],  # PubTables has no caption annotations
        })

    return samples


# ---------------------------------------------------------------------------
# PASCAL VOC XML parsing (fallback for raw downloads)
# ---------------------------------------------------------------------------


def parse_voc_xml(xml_path: str) -> dict:
    """Parse PASCAL VOC XML annotation."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    tables = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        x0 = float(bbox.find("xmin").text)
        y0 = float(bbox.find("ymin").text)
        x1 = float(bbox.find("xmax").text)
        y1 = float(bbox.find("ymax").text)
        tables.append({
            "bbox": [x0, y0, x1, y1],
            "category": name,
        })

    return {
        "image_size": [width, height],
        "tables": tables,
    }


def extract_images(
    input_dir: str = PUBTABLES_CACHE,
    split: str = "val",
):
    """Extract only the images matching our downloaded annotations from the tar.gz.

    Expects PubTables-1M-Detection_Images_Test.tar.gz to be in input_dir.
    """
    import tarfile

    tar_path = os.path.join(input_dir, "PubTables-1M-Detection_Images_Test.tar.gz")
    if not os.path.exists(tar_path):
        print(f"Error: {tar_path} not found. Download it first.")
        return

    ann_dir = os.path.join(input_dir, split, "annotations")
    img_dir = os.path.join(input_dir, split, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Build set of filenames we need
    needed = set()
    for af in Path(ann_dir).glob("*.json"):
        with open(af) as f:
            ann = json.load(f)
        source = ann.get("source_filename", "")
        if source:
            needed.add(source)

    print(f"Extracting {len(needed)} images from tar.gz...")

    extracted = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tqdm(tar, desc="Scanning tar"):
            basename = os.path.basename(member.name)
            if basename in needed:
                # Extract to images dir with our naming convention
                # Map source_filename back to annotation image_id
                image_id = basename.replace(".jpg", "").replace(".png", "")
                out_path = os.path.join(img_dir, f"{image_id}.jpg")
                if not os.path.exists(out_path):
                    f = tar.extractfile(member)
                    if f:
                        with open(out_path, "wb") as out_f:
                            out_f.write(f.read())
                        extracted += 1

    print(f"  Extracted {extracted} images to {img_dir}/")


def main():
    parser = argparse.ArgumentParser(description="PubTables-1M dataset adapter")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Download annotations from HuggingFace")
    p_dl.add_argument("--n-samples", type=int, default=500)
    p_dl.add_argument("--split", default="val")

    p_extract = sub.add_parser("extract", help="Extract images from downloaded tar.gz")
    p_extract.add_argument("--split", default="val")

    p_conv = sub.add_parser("convert", help="Convert to YOLO format")
    p_conv.add_argument("--split", default="val")

    args = parser.parse_args()

    if args.command == "download":
        download_pubtables(args.n_samples, split=args.split)
    elif args.command == "extract":
        extract_images(split=args.split)
    elif args.command == "convert":
        convert_to_yolo(split=args.split)


if __name__ == "__main__":
    main()
