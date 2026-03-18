#!/usr/bin/env bash
# Resume B from last checkpoint + train C from scratch.
# This script runs detached from the terminal via nohup.
set -e
cd "$(dirname "$0")"

echo "=== Resume B + Train C ==="
echo "Start: $(date)"

# Resume B
echo "--- Resuming Formulation B ---"
uv run python -c "
from ultralytics import YOLO
import os
model = YOLO(os.path.expanduser('~/.cache/table_cropper/yolo_data/runs/formulation_B/weights/last.pt'))
model.train(
    data=os.path.expanduser('~/.cache/table_cropper/yolo_data/B/dataset.yaml'),
    epochs=50, batch=16, imgsz=640,
    project=os.path.expanduser('~/.cache/table_cropper/yolo_data/runs'),
    name='formulation_B', exist_ok=True, device='mps',
    patience=10, save=True, plots=True, resume=True,
)
print('B DONE')
"

# Train C from scratch
echo "--- Training Formulation C ---"
uv run python yolo_pipeline.py train --formulation C --epochs 50 --model-size m

echo "=== ALL DONE: $(date) ==="
