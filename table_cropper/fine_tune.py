"""
Agentic fine-tuning loop for Table Transformer on SCI-3000.

Trains the model to detect table+caption regions as a single target.
Runs an autonomous experiment loop: train → evaluate → adjust → repeat.

Usage:
    uv run fine_tune.py                           # full run
    uv run fine_tune.py --epochs 3 --lr 1e-5      # quick test
    uv run fine_tune.py --resume checkpoint.pt    # resume from checkpoint
"""

import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    TableTransformerForObjectDetection,
)

DEFAULT_DATA_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "table_cropper", "finetune_data"
)
DEFAULT_CHECKPOINT_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "table_cropper", "checkpoints"
)
RESULTS_FILE = os.path.join(
    os.path.dirname(__file__), "results.tsv"
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TableCaptionDataset(Dataset):
    """Dataset for fine-tuning on pre-rendered page images with merged table+caption boxes."""

    def __init__(self, data_dir: str, split: str, processor, augment: bool = False):
        self.split_dir = os.path.join(data_dir, split)
        self.img_dir = os.path.join(self.split_dir, "images")
        self.ann_dir = os.path.join(self.split_dir, "annotations")
        self.processor = processor
        self.augment = augment

        # Collect all annotation files
        self.samples = sorted(Path(self.ann_dir).glob("*.json"))
        if not self.samples:
            raise ValueError(f"No annotations found in {self.ann_dir}")

    def __len__(self):
        return len(self.samples)

    def _apply_augmentation(self, image, boxes):
        """Apply training-time augmentations to image and adjust bboxes.
        
        Augmentations:
        - Color jitter (brightness, contrast, saturation ±20%)
        - Random horizontal flip (50%) with bbox mirroring
        - Random scale (85%–115%) with bbox adjustment
        """
        import random

        # Color jitter (doesn't affect bboxes)
        for enhancer_cls in [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]:
            factor = random.uniform(0.8, 1.2)
            image = enhancer_cls(image).enhance(factor)

        # Random horizontal flip
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            # Mirror bboxes: new_cx = 1 - cx
            boxes = [[1.0 - cx, cy, w, h] for cx, cy, w, h in boxes]

        # Random scale (resize image, adjust nothing since bboxes are normalized)
        scale = random.uniform(0.85, 1.15)
        new_w = int(image.width * scale)
        new_h = int(image.height * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)

        return image, boxes

    def __getitem__(self, idx):
        ann_path = self.samples[idx]
        page_id = ann_path.stem
        img_path = os.path.join(self.img_dir, f"{page_id}.png")

        # Load image
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # Load annotation
        with open(ann_path) as f:
            ann_data = json.load(f)

        # Convert COCO (cx, cy, w, h) normalized → format for processor
        boxes = []
        labels = []
        for ann in ann_data["annotations"]:
            cx, cy, w, h = ann["bbox"]
            boxes.append([cx, cy, w, h])
            labels.append(ann["category_id"])

        # Apply augmentation (training only)
        if self.augment and boxes:
            image, boxes = self._apply_augmentation(image, boxes)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "class_labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([image.height, image.width]),
        }

        # Process image
        encoding = self.processor(
            images=image,
            return_tensors="pt",
        )

        pixel_values = encoding["pixel_values"].squeeze(0)

        return pixel_values, target


def collate_fn(batch):
    """Custom collate: pad variable-size images to uniform batch dimensions."""
    pixel_values_list = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Find max height and width in batch
    max_h = max(pv.shape[1] for pv in pixel_values_list)
    max_w = max(pv.shape[2] for pv in pixel_values_list)

    # Pad each image and create pixel mask
    padded = []
    pixel_masks = []
    for pv in pixel_values_list:
        c, h, w = pv.shape
        pad_h = max_h - h
        pad_w = max_w - w
        # Pad right and bottom with zeros
        padded_pv = torch.nn.functional.pad(pv, (0, pad_w, 0, pad_h), value=0)
        padded.append(padded_pv)
        # Pixel mask: 1 where there's real content, 0 where padded
        mask = torch.zeros(max_h, max_w, dtype=torch.long)
        mask[:h, :w] = 1
        pixel_masks.append(mask)

    pixel_values = torch.stack(padded)
    pixel_mask = torch.stack(pixel_masks)
    return pixel_values, pixel_mask, targets


# ---------------------------------------------------------------------------
# Evaluation on pre-rendered val set (fast, no PDF rendering)
# ---------------------------------------------------------------------------


def evaluate_fast(
    model,
    processor,
    data_dir: str,
    device: torch.device,
    confidence: float = 0.5,
    iou_threshold: float = 0.3,
) -> dict:
    """
    Fast evaluation on pre-rendered val images.
    Tracks: precision, recall, F1, mean IoU, and false-positive rate
    on negative pages (pages with no ground-truth tables).
    """
    val_dir = os.path.join(data_dir, "val")
    img_dir = os.path.join(val_dir, "images")
    ann_dir = os.path.join(val_dir, "annotations")

    ann_files = sorted(Path(ann_dir).glob("*.json"))
    if not ann_files:
        return {"error": "No val annotations found"}

    model.eval()

    total_gt = 0
    total_det = 0
    true_pos = 0
    false_pos = 0
    false_neg = 0
    iou_sum = 0.0
    matched_count = 0
    # Track false positives on negative pages
    negative_pages = 0
    negative_pages_with_fp = 0

    for ann_path in tqdm(ann_files, desc="  Evaluating", leave=False):
        page_id = ann_path.stem
        img_path = os.path.join(img_dir, f"{page_id}.png")

        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # Load GT
        with open(ann_path) as f:
            ann_data = json.load(f)

        gt_boxes = []
        for ann in ann_data["annotations"]:
            cx, cy, w, h = ann["bbox"]
            x0 = (cx - w / 2) * img_w
            y0 = (cy - h / 2) * img_h
            x1 = (cx + w / 2) * img_w
            y1 = (cy + h / 2) * img_h
            gt_boxes.append((x0, y0, x1, y1))

        is_negative = len(gt_boxes) == 0
        if is_negative:
            negative_pages += 1

        total_gt += len(gt_boxes)

        # Run detection
        encoding = processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values)

        target_sizes = torch.tensor([[img_h, img_w]], device=device)
        results = processor.post_process_object_detection(
            outputs, threshold=confidence, target_sizes=target_sizes
        )[0]

        det_boxes = results["boxes"].cpu().numpy()
        total_det += len(det_boxes)

        if is_negative:
            if len(det_boxes) > 0:
                negative_pages_with_fp += 1
                false_pos += len(det_boxes)
            continue

        # Match detections to GT (positive pages)
        gt_matched = [False] * len(gt_boxes)

        for det_box in det_boxes:
            det_xyxy = (det_box[0], det_box[1], det_box[2], det_box[3])
            best_iou = 0.0
            best_idx = -1

            for gi, gt_box in enumerate(gt_boxes):
                iou = _compute_iou(det_xyxy, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi

            if best_iou >= iou_threshold and best_idx >= 0 and not gt_matched[best_idx]:
                true_pos += 1
                gt_matched[best_idx] = True
                iou_sum += best_iou
                matched_count += 1
            else:
                false_pos += 1

        false_neg += sum(1 for m in gt_matched if not m)

    precision = true_pos / total_det if total_det > 0 else 0
    recall = true_pos / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = iou_sum / matched_count if matched_count > 0 else 0
    fp_rate_neg = negative_pages_with_fp / negative_pages if negative_pages > 0 else 0

    return {
        "gt_tables": total_gt,
        "detections": total_det,
        "true_positives": true_pos,
        "false_positives": false_pos,
        "false_negatives": false_neg,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "negative_pages": negative_pages,
        "negative_pages_with_fp": negative_pages_with_fp,
        "fp_rate_on_negatives": round(fp_rate_neg, 4),
    }


def _compute_iou(a, b):
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_grad_norm: float = 0.1,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"  Epoch {epoch}", leave=False)
    for pixel_values, pixel_mask, targets in pbar:
        pixel_values = pixel_values.to(device)
        pixel_mask = pixel_mask.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=targets)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------


def log_result(
    experiment: str,
    metrics: dict,
    status: str,
    description: str,
):
    """Append a row to results.tsv."""
    header = "experiment\tcaption_recall\ttable_recall\tprecision\tmean_iou\tf1\tstatus\tdescription\n"

    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write(header)

    row = (
        f"{experiment}\t"
        f"{metrics.get('caption_recall', '-')}\t"
        f"{metrics.get('recall', '-')}\t"
        f"{metrics.get('precision', '-')}\t"
        f"{metrics.get('mean_iou', '-')}\t"
        f"{metrics.get('f1', '-')}\t"
        f"{status}\t"
        f"{description}\n"
    )
    with open(RESULTS_FILE, "a") as f:
        f.write(row)


# ---------------------------------------------------------------------------
# Target checking
# ---------------------------------------------------------------------------

TARGETS = {
    "recall": 0.95,         # Must find ≥95% of tables
    "precision": 0.97,     # ≥97% of detections must be real tables (tightened)
    "mean_iou": 0.93,      # Tighter bboxes with expanded GT (Phase 4)
    "fp_rate_on_negatives": -0.03,  # ≤3% of negative pages get FP (tightened from 0.05)
}


def check_targets(metrics: dict) -> tuple[bool, list[str]]:
    """Check if all target metrics are met. Returns (all_met, list_of_failures)."""
    failures = []
    for key, target in TARGETS.items():
        val = metrics.get(key, 0)
        if target < 0:
            # Upper-bound metric (e.g. fp_rate — smaller is better)
            if val > abs(target):
                failures.append(f"{key}={val:.4f} > {abs(target)}")
        else:
            if val < target:
                failures.append(f"{key}={val:.4f} < {target}")
    return len(failures) == 0, failures


def is_improvement(new: dict, best: dict) -> bool:
    """Check if new metrics are better than best (any metric improved, none regressed badly)."""
    improved = False
    for key in ["recall", "precision", "mean_iou", "f1"]:
        new_val = new.get(key, 0)
        best_val = best.get(key, 0)
        if new_val > best_val + 0.001:
            improved = True
        if new_val < best_val - 0.02:  # Allow small regression
            return False
    return improved


# ---------------------------------------------------------------------------
# Main agentic loop
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Table Transformer")
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help="Path to prepared dataset",
    )
    parser.add_argument(
        "--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR,
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Resume from a checkpoint",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Epochs per training round",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size (default 8 for M4 Max, reduce if OOM)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Detection confidence threshold for evaluation",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=10,
        help="Maximum training rounds before stopping",
    )
    parser.add_argument(
        "--experiment-name", default=None,
        help="Name for this experiment (auto-generated if not set)",
    )
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load processor
    processor = AutoImageProcessor.from_pretrained(
        "microsoft/table-transformer-detection"
    )

    # Load model
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = TableTransformerForObjectDetection.from_pretrained(args.resume)
    else:
        print("Loading pre-trained Table Transformer...")
        model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection"
        )

    # The pre-trained model has 2 classes (table, table rotated).
    # We need 1 class (table_with_caption). Reinit the class head.
    # But skip reinit if resuming from a fine-tuned checkpoint.
    num_labels = 1
    if model.config.num_labels != num_labels and not args.resume:
        print(f"  Reinitializing classification head: {model.config.num_labels} → {num_labels}")
        model.config.num_labels = num_labels
        model.class_labels_classifier = nn.Linear(
            model.config.hidden_size, num_labels + 1
        )
        nn.init.xavier_uniform_(model.class_labels_classifier.weight)
        nn.init.zeros_(model.class_labels_classifier.bias)

    model = model.to(device)

    # Dataset
    print(f"\nLoading dataset from {args.data_dir}")
    train_dataset = TableCaptionDataset(args.data_dir, "train", processor, augment=True)
    print(f"  Train samples: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # MPS doesn't play well with multiprocess
        pin_memory=False,
    )

    # Baseline evaluation
    print("\nBaseline evaluation (before fine-tuning)...")
    baseline_metrics = evaluate_fast(
        model, processor, args.data_dir, device, args.confidence
    )
    print(f"  Baseline: {baseline_metrics}")
    log_result(
        "baseline",
        baseline_metrics,
        "keep",
        "Pre-trained TATR (class head reinit)"
    )

    best_metrics = baseline_metrics
    best_checkpoint = None
    no_improve_count = 0
    round_num = 0

    # Agentic loop
    print(f"\n{'='*60}")
    print("STARTING AGENTIC FINE-TUNING LOOP")
    print(f"  Max rounds: {args.max_rounds}")
    print(f"  Targets: {TARGETS}")
    print(f"{'='*60}\n")

    current_lr = args.lr
    current_epochs = args.epochs

    while round_num < args.max_rounds:
        round_num += 1
        exp_name = args.experiment_name or f"round{round_num:02d}"

        print(f"\n{'─'*50}")
        print(f"ROUND {round_num}/{args.max_rounds}")
        print(f"  LR: {current_lr:.2e}, Epochs: {current_epochs}, Batch: {args.batch_size}")
        print(f"{'─'*50}")

        # Save model state before training (for rollback)
        pre_train_state = copy.deepcopy(model.state_dict())

        # Optimizer (re-create each round for fresh state)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.class_labels_classifier.parameters(), "lr": current_lr * 10},
                {"params": model.bbox_predictor.parameters(), "lr": current_lr},
                {"params": [p for n, p in model.named_parameters()
                            if "class_labels_classifier" not in n and "bbox_predictor" not in n],
                 "lr": current_lr * 0.1},
            ],
            weight_decay=1e-4,
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=current_epochs * len(train_loader)
        )

        # Train
        t0 = time.time()
        for epoch in range(1, current_epochs + 1):
            avg_loss = train_one_epoch(
                model, train_loader, optimizer, device, epoch
            )
            scheduler.step()
            print(f"    Epoch {epoch}: avg_loss={avg_loss:.4f}")

        train_time = time.time() - t0
        print(f"  Training time: {train_time:.1f}s")

        # Evaluate
        print(f"  Evaluating...")
        metrics = evaluate_fast(
            model, processor, args.data_dir, device, args.confidence
        )
        print(f"  Results: {metrics}")

        # Check targets
        all_met, failures = check_targets(metrics)
        if all_met:
            print(f"\n  ✅ ALL TARGETS MET!")
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{exp_name}_final")
            model.save_pretrained(checkpoint_path)
            processor.save_pretrained(checkpoint_path)
            log_result(exp_name, metrics, "final", f"LR={current_lr:.2e} E={current_epochs} ALL_TARGETS_MET")
            print(f"  Saved final model to {checkpoint_path}")
            break

        # Check improvement
        improved = is_improvement(metrics, best_metrics)

        if improved:
            print(f"  📈 Improvement! Keeping checkpoint.")
            no_improve_count = 0
            best_metrics = metrics
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{exp_name}_best")
            model.save_pretrained(checkpoint_path)
            processor.save_pretrained(checkpoint_path)
            best_checkpoint = checkpoint_path
            log_result(exp_name, metrics, "keep", f"LR={current_lr:.2e} E={current_epochs}")
        else:
            print(f"  📉 No improvement. Rolling back.")
            no_improve_count += 1
            model.load_state_dict(pre_train_state)
            log_result(exp_name, metrics, "discard", f"LR={current_lr:.2e} E={current_epochs} no_improvement")

        # Remaining failures
        print(f"  Remaining targets: {failures}")

        # Adaptive strategy
        if no_improve_count >= 2:
            # Shift strategy
            if current_lr < 5e-5:
                current_lr *= 3
                print(f"  🔧 Increasing LR to {current_lr:.2e}")
            elif current_lr < 1e-4:
                current_lr *= 2
                current_epochs = min(current_epochs + 2, 15)
                print(f"  🔧 LR={current_lr:.2e}, Epochs={current_epochs}")
            else:
                current_lr = args.lr
                current_epochs += 3
                print(f"  🔧 Reset LR={current_lr:.2e}, Epochs={current_epochs}")
            no_improve_count = 0

        # Check for stagnation
        if round_num >= 3 and no_improve_count >= 3:
            print(f"\n  ⚠ Stagnation detected after {round_num} rounds. Stopping.")
            break

    # Final summary
    print(f"\n{'='*60}")
    print("FINE-TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"  Rounds: {round_num}")
    print(f"  Best metrics: {best_metrics}")
    all_met, failures = check_targets(best_metrics)
    if all_met:
        print(f"  ✅ All targets met!")
    else:
        print(f"  ⚠ Remaining targets: {failures}")
    if best_checkpoint:
        print(f"  Best checkpoint: {best_checkpoint}")
    print(f"  Results log: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
