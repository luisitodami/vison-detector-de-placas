# VISON — License Plate Detector (YOLOv8 + OCR)

This repository contains the code and instructions to reproduce the ALPR pipeline (LPD + LPR), the training-size study, and the deployment benchmarks used in the paper.

## Quick start (CPU)
```bash
pip install -r requirements.txt
python scripts/bench_cpu.py
```

## Train / Validate (Ultralytics YOLOv8)
```bash
# Train (example)
yolo detect train model=yolov8s.pt data=data.yaml imgsz=640 epochs=50 patience=50
# Validate and save PR/F1 curves
yolo detect val model=runs/detect/exp/weights/best.pt plots=True
```

## Notes
- Datasets (`train/`, `valid/`, `test/`, `subsets_series/`) and run artifacts (`runs/`) are **not** in the repo; see your local paths.
- You can export a YOLOv8 model to ONNX and then run `scripts/bench_onnx_cpu.py --onnx yolov8s.onnx`.
