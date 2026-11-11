# Untung Tanujaya — MobileNetV3-Large on MPS (Apple GPU)

Training run for the LeafScan project: **MobileNetV3-Large** with transfer
learning on **PlantVillage** (38 classes), trained on a MacBook Pro
(Apple M5 Pro, 24 GB) through the **MPS** backend.

Compared against:
- Mustafa — EfficientNet-B0, MPS
- Sebastien — EfficientNet-B0, CUDA

## Reported result: lab → field (run of Nov 11, 2026)

In-domain (lab → lab) accuracy is deliberately **not** reported: on
PlantVillage it saturates near 100 % for any modern backbone, so it cannot
discriminate models and says nothing about real-world use. Those numbers live
in a local appendix only (`report/experiment_log_lab_appendix.pdf`, not in
git). The yardstick is field data the models never saw. Comparison table +
MPS performance gotchas in `COMPARISON.md`. `runs/` (checkpoints) stays local.

Train on **lab data** (PlantVillage, plain background, controlled light),
test on **field data** (PlantDoc, 2,572 internet photos of leaves on real
plants, 28 classes mapped to PlantVillage names in `ml/field_mapping.json`).
Field data is never used for training or model selection. Same recipe, seed 42.

| Run | Train data | **Field top-1** (95 % CI) | Field top-5 | Macro F1 | PlantDoc official test (n=236) |
|---|---|---|---|---|---|
| A `lab_mobilenetv3_mps` | PlantVillage train split (70 %, 38,013) | **25.04 %** [23.40, 26.75] | 54.94 % | 0.2225 | 31.78 % |
| B `fulllab_mobilenetv3_mps` | full PlantVillage (100 %, 54,305) | **25.43 %** [23.78, 27.15] | 53.89 % | 0.2255 | 33.05 % |

- The gap between saturated in-domain accuracy (local appendix) and field
  accuracy is **domain shift**, not a training bug: run A reproduces the
  Nov 7 in-domain run exactly.
- 43 % more lab data does **not** help on field data: paired exact McNemar test
  on the same 2,572 images, 104 images only A gets right vs 114 only B,
  **p = 0.54** (`results/field_A_vs_B_mcnemar.json`).
- Errors collapse onto a few lab classes (Tomato Late blight, Strawberry Leaf
  scorch — the latter has *no* field images at all).
- No lab↔field leakage: dHash + pixel-correlation check found 0 duplicates
  (max correlation 0.816, `results/field_overlap.json`).
- Expected range: Mohanty et al. 2016 report 31.4 % on internet images;
  PlantDoc paper reports 15.08 % for PlantVillage-trained models.

Artifacts: `results/lab_mobilenetv3_mps/`, `results/fulllab_mobilenetv3_mps/`
(`field_report.txt`, `field_results.json`, `field_confusion.png`,
`field_predictions.csv`), `results/lab_vs_field.jpg`. Write-ups:
`report/experiment_log.pdf` (reported, field-only) and
`report/experiment_log_lab_appendix.pdf` (local, in-domain numbers).

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

## Field evaluation (lab → field)

```bash
# field data (from repo root; github.com blocked? use: gh api repos/pratikkayal/PlantDoc-Dataset/tarball/master > plantdoc.tgz)
git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset data/raw/PlantDoc-Dataset

cd untung/ml
python train.py --run-name lab_mobilenetv3_mps                 # A: 70 % lab split
python train.py --full-lab --run-name fulllab_mobilenetv3_mps  # B: 100 % lab, no val, last epoch
python evaluate_field.py --checkpoint ../runs/<run>/best.pt    # PlantDoc train/ + test/
python check_overlap.py                                        # lab<->field near-duplicate check
```

For the EfficientNet-B0 runs to be comparable, score them on the identical
field set: reuse `ml/field_mapping.json` + `collect_field()` from
`evaluate_field.py` and swap `M.build_model(...)` for your own model builder.

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
