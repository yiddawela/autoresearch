"""
Download and extract the SCI-3000 dataset from Zenodo.

SCI-3000 contains bounding boxes of figures, tables, and captions
in 34,791 pages from 3,000 scientific publications.

Usage:
    uv run download_data.py              # download + extract
    uv run download_data.py --data-dir /path/to/store
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ZENODO_URL = "https://zenodo.org/records/6564971/files/SCI-3000-full.zip?download=1"
FILENAME = "SCI-3000-full.zip"
DEFAULT_DATA_DIR = os.path.join(os.path.expanduser("~"), ".cache", "table_cropper")


def download_file(url: str, dest: str, chunk_size: int = 1024 * 1024):
    """Download a file with progress bar."""
    if os.path.exists(dest):
        print(f"  Already downloaded: {dest}")
        return

    print(f"  Downloading from {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    temp_path = dest + ".tmp"

    with open(temp_path, "wb") as f:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="  Downloading",
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    os.rename(temp_path, dest)
    print(f"  Saved to {dest}")


def extract_zip(zip_path: str, extract_to: str):
    """Extract a zip file with progress."""
    print(f"  Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="  Extracting", unit="files"):
            zf.extract(member, extract_to)
    print(f"  Extracted {len(members)} files to {extract_to}")


def summarize_dataset(data_dir: str):
    """Print a summary of the dataset structure."""
    sci_dir = os.path.join(data_dir, "SCI-3000-full")
    if not os.path.isdir(sci_dir):
        # Try to find the actual extracted directory
        for item in os.listdir(data_dir):
            candidate = os.path.join(data_dir, item)
            if os.path.isdir(candidate) and item != "__MACOSX":
                sci_dir = candidate
                break

    print(f"\nDataset directory: {sci_dir}")
    if not os.path.isdir(sci_dir):
        print("  Warning: could not find extracted directory")
        return sci_dir

    # Count files by type
    pdf_count = 0
    annotation_count = 0
    image_count = 0
    for root, dirs, files in os.walk(sci_dir):
        for f in files:
            if f.endswith(".pdf"):
                pdf_count += 1
            elif f.endswith((".json", ".xml", ".csv", ".txt")):
                annotation_count += 1
            elif f.endswith((".png", ".jpg", ".jpeg")):
                image_count += 1

    print(f"  PDFs: {pdf_count}")
    print(f"  Annotations: {annotation_count}")
    print(f"  Images: {image_count}")

    # Show top-level structure
    print(f"\n  Top-level structure:")
    try:
        for item in sorted(os.listdir(sci_dir))[:20]:
            item_path = os.path.join(sci_dir, item)
            if os.path.isdir(item_path):
                child_count = sum(1 for _ in os.scandir(item_path))
                print(f"    📁 {item}/ ({child_count} items)")
            else:
                size_mb = os.path.getsize(item_path) / (1024 * 1024)
                print(f"    📄 {item} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"    Error listing: {e}")

    return sci_dir


def main():
    parser = argparse.ArgumentParser(description="Download SCI-3000 dataset")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directory to store dataset (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction (if already extracted)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, FILENAME)

    print(f"Data directory: {data_dir}")
    print()

    # Step 1: Download
    print("Step 1: Download SCI-3000 from Zenodo")
    download_file(ZENODO_URL, zip_path)
    print()

    # Step 2: Extract
    if not args.skip_extract:
        print("Step 2: Extract")
        extract_zip(zip_path, data_dir)
        print()

    # Step 3: Summarize
    print("Step 3: Dataset summary")
    summarize_dataset(data_dir)
    print()
    print("Done! Dataset ready for evaluation.")


if __name__ == "__main__":
    main()
