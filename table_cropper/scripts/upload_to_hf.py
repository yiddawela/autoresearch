"""Upload model checkpoints to Hugging Face Hub.

Prerequisites:
    1. Create a free account at https://huggingface.co/join
    2. Create an access token at https://huggingface.co/settings/tokens
       (select "Write" permission)
    3. pip install huggingface_hub

Usage:
    # First time: log in (stores token locally)
    huggingface-cli login

    # Then run this script
    python scripts/upload_to_hf.py --hf-user YOUR_USERNAME

    # To make repos public (e.g. after paper acceptance)
    python scripts/upload_to_hf.py --hf-user YOUR_USERNAME --public
"""

import argparse
import os
import sys
from pathlib import Path

CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "table_cropper"

TATR_CHECKPOINT = CACHE_DIR / "checkpoints" / "phase5_resumed_final"

YOLO_WEIGHTS = {
    "formulation_A": CACHE_DIR / "yolo_data" / "runs" / "formulation_A" / "weights" / "best.pt",
    "formulation_B": CACHE_DIR / "yolo_data" / "runs" / "formulation_B" / "weights" / "best.pt",
    "formulation_C": CACHE_DIR / "yolo_data" / "runs" / "formulation_C" / "weights" / "best.pt",
}

TATR_MODEL_CARD = """---
license: apache-2.0
tags:
  - table-detection
  - document-analysis
  - object-detection
  - TATR
datasets:
  - SCI-3000
pipeline_tag: object-detection
---

# TATR Fine-Tuned for Semantically Complete Table Extraction (Formulation B)

Fine-tuned [Microsoft Table Transformer (TATR)](https://huggingface.co/microsoft/table-transformer-detection)
for detecting merged table-body-plus-caption regions in scientific documents.

## Training

- **Base model:** microsoft/table-transformer-detection
- **Dataset:** SCI-3000 (2003 training pages, 997 validation pages)
- **Target:** Merged bounding box enclosing table body + linked captions
- **Resolution:** 800px max dimension at 200 dpi
- **Validation AP@50:** 0.989

## Usage

```python
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from PIL import Image

model = AutoModelForObjectDetection.from_pretrained("{hf_user}/tatr-formulation-b")
processor = AutoImageProcessor.from_pretrained("{hf_user}/tatr-formulation-b")

image = Image.open("page.png")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
```

## Paper

*Beyond Table Bodies: Semantically Complete Table Extraction from Scientific Documents*
(submitted to IJDAR)
"""

YOLO_MODEL_CARD = """---
license: apache-2.0
tags:
  - table-detection
  - document-analysis
  - object-detection
  - YOLO
datasets:
  - SCI-3000
---

# YOLOv26m Table Detection Weights (Formulations A, B, C)

YOLOv26m checkpoints fine-tuned on SCI-3000 for three table detection formulations:

| File | Formulation | Target |
|------|-------------|--------|
| `formulation_A.pt` | A (Table-only) | Table body bounding box |
| `formulation_B.pt` | B (Merged crop) | Merged table body + caption bounding box |
| `formulation_C.pt` | C (Multi-class) | Separate table and caption class detections |

## Training

- **Architecture:** YOLOv26m (21.8M parameters)
- **Dataset:** SCI-3000 (same split as TATR experiments)
- **Epochs:** 50
- **Batch size:** 16
- **Optimiser:** AdamW

## Usage

```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

path = hf_hub_download("{hf_user}/yolo-table-detection", "formulation_B.pt")
model = YOLO(path)
results = model("page.png")
```

## Paper

*Beyond Table Bodies: Semantically Complete Table Extraction from Scientific Documents*
(submitted to IJDAR)
"""


def upload_tatr(hf_user: str, private: bool = True):
    """Upload TATR checkpoint as a HuggingFace model repo."""
    from huggingface_hub import HfApi

    repo_id = f"{hf_user}/tatr-formulation-b"
    api = HfApi()

    if not TATR_CHECKPOINT.exists():
        print(f"TATR checkpoint not found at {TATR_CHECKPOINT}")
        return False

    print(f"Creating repo {repo_id} (private={private})...")
    api.create_repo(repo_id, private=private, exist_ok=True)

    # Write model card
    card_path = TATR_CHECKPOINT / "README.md"
    card_path.write_text(TATR_MODEL_CARD.replace("{hf_user}", hf_user))

    print(f"Uploading TATR checkpoint from {TATR_CHECKPOINT}...")
    api.upload_folder(
        folder_path=str(TATR_CHECKPOINT),
        repo_id=repo_id,
        commit_message="Upload fine-tuned TATR (Formulation B) checkpoint",
    )
    print(f"Done: https://huggingface.co/{repo_id}")
    return True


def upload_yolo(hf_user: str, private: bool = True):
    """Upload YOLO weights to a single HuggingFace repo."""
    from huggingface_hub import HfApi
    import tempfile

    repo_id = f"{hf_user}/yolo-table-detection"
    api = HfApi()

    print(f"Creating repo {repo_id} (private={private})...")
    api.create_repo(repo_id, private=private, exist_ok=True)

    # Upload model card
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(YOLO_MODEL_CARD.replace("{hf_user}", hf_user))
        card_tmp = f.name

    api.upload_file(
        path_or_fileobj=card_tmp,
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add model card",
    )
    os.unlink(card_tmp)

    # Upload each weight file
    uploaded = 0
    for name, path in YOLO_WEIGHTS.items():
        if not path.exists():
            print(f"  Skipping {name}: {path} not found")
            continue
        dest = f"{name}.pt"
        print(f"  Uploading {name} ({path.stat().st_size / 1e6:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=dest,
            repo_id=repo_id,
            commit_message=f"Upload {name} weights",
        )
        uploaded += 1

    print(f"Done: https://huggingface.co/{repo_id} ({uploaded} weights uploaded)")
    return uploaded > 0


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoints to Hugging Face Hub")
    parser.add_argument("--hf-user", required=True, help="Your HuggingFace username")
    parser.add_argument("--public", action="store_true", help="Make repos public (default: private)")
    parser.add_argument("--tatr-only", action="store_true", help="Upload TATR only")
    parser.add_argument("--yolo-only", action="store_true", help="Upload YOLO only")
    args = parser.parse_args()

    private = not args.public

    try:
        from huggingface_hub import HfApi  # noqa: F401
    except ImportError:
        print("Install huggingface_hub first:  pip install huggingface_hub")
        sys.exit(1)

    if not args.yolo_only:
        upload_tatr(args.hf_user, private=private)

    if not args.tatr_only:
        upload_yolo(args.hf_user, private=private)

    print("\nTo make repos public later, run:")
    print(f"  python scripts/upload_to_hf.py --hf-user {args.hf_user} --public")


if __name__ == "__main__":
    main()
