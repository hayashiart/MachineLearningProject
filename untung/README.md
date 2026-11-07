# Untung Tanujaya — MobileNetV3-Large on MPS (Apple GPU)

Training run for the LeafScan project: **MobileNetV3-Large** with transfer
learning on **PlantVillage** (38 classes), trained on a MacBook Pro
(Apple M5 Pro, 24 GB) through the **MPS** backend.

Compared against:
- Mustafa — EfficientNet-B0, MPS
- Sebastien — EfficientNet-B0, CUDA

## Result (run of Nov 7, 2026)

| Test accuracy | Macro F1 | Training (12 ep) | Inference (batch 1, MPS) | Params / size |
|---|---|---|---|---|
| **99.67 %** | 0.9953 | **19.7 min** | **4.73 ms/img** | 4.25 M / 16.4 MB |

Artifacts (curves, confusion matrix, per-class report) in
`results/mobilenetv3_mps_full/`; comparison table + MPS performance gotchas in
`COMPARISON.md`. `runs/` (checkpoints) stays local, not in git.

## Setup

```bash
# from the repo root
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r untung/requirements.txt
```

## Data

```bash
git clone --depth 1 https://github.com/spMohanty/PlantVillage-Dataset \
    data/raw/PlantVillage-Dataset
```

The split (70/15/15, stratified, seed 42) is **committed** in `ml/splits/` so
all three members train and evaluate on identical data. To regenerate:

```bash
cd untung/ml && python prepare_data.py
```

## Train

```bash
cd untung/ml
python train.py --smoke    # 500 images, 1 epoch — pipeline bug check
python train.py            # full run: freeze 2 epochs + fine-tune 10 epochs
```

Outputs land in `untung/runs/<run_name>/`: `best.pt`, `history.json`,
`curves.png`, `train_summary.json`.

## Evaluate (test set, once)

```bash
python evaluate.py                       # latest run
python evaluate.py --checkpoint ../runs/<run>/best.pt
```

Outputs: `report.txt` (accuracy + per-class P/R/F1), `confusion_matrix.png`,
`results.json` (numbers for the comparison table, incl. batch-1 inference ms).

## Training recipe (per project plan §6)

| Item | Value |
|---|---|
| Backbone | MobileNetV3-Large, ImageNet-1K V2 weights |
| Head | `classifier[3]` replaced by `Linear(1280, 38)` |
| Input | 224×224, ImageNet mean/std |
| Augmentation (train only) | RandomResizedCrop, H/V flip, rotation 15°, colour jitter, Gaussian blur |
| Stage 1 | backbone frozen, AdamW lr 1e-3, 2 epochs |
| Stage 2 | unfrozen, AdamW lr 1e-4, cosine decay, label smoothing 0.1, 10 epochs |
| Batch size | 64 |
| Precision | float32 (no AMP), `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| Device order | cuda → mps → cpu |
| Seed | 42 |
