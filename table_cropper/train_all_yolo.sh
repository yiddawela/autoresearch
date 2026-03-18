#!/usr/bin/env bash
# Train YOLO26 for all three formulations sequentially.
# Run this overnight:  nohup bash train_all_yolo.sh > yolo_training.log 2>&1 &
#
# Monitor progress:  tail -f yolo_training.log

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "YOLO26 Training: All Formulations"
echo "Started: $(date)"
echo "=========================================="

for FORM in A B C; do
    echo ""
    echo "--- Formulation ${FORM} ---"
    echo "Start: $(date)"
    uv run python yolo_pipeline.py train --formulation "${FORM}" --epochs 50 --model-size m
    echo "End: $(date)"
done

echo ""
echo "=========================================="
echo "All training complete: $(date)"
echo "=========================================="
echo "Results at: ~/.cache/table_cropper/yolo_data/runs/"
ls -la ~/.cache/table_cropper/yolo_data/runs/
