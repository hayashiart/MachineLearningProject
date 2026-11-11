"""Central configuration for the MobileNetV3-Large (MPS) training run.

Member: Untung Tanujaya
Model : MobileNetV3-Large, transfer learning on PlantVillage (38 classes)
Device: Apple GPU through the MPS backend (cuda -> mps -> cpu fallback)
"""

import os
from pathlib import Path

# Must be set before torch is imported anywhere in the pipeline.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ---------------------------------------------------------------- paths
ML_DIR = Path(__file__).resolve().parent          # untung/ml
UNTUNG_DIR = ML_DIR.parent                        # untung/
REPO_ROOT = UNTUNG_DIR.parent                     # MachineLearningProject/

DATA_ROOT = REPO_ROOT / "data" / "raw" / "PlantVillage-Dataset" / "raw" / "color"
SPLITS_DIR = ML_DIR / "splits"                    # committed so all members share one split
# Field (real-world) test set: never used for training or model selection
FIELD_ROOT = REPO_ROOT / "data" / "raw" / "PlantDoc-Dataset"
FIELD_MAPPING = ML_DIR / "field_mapping.json"
RUNS_DIR = UNTUNG_DIR / "runs"                    # gitignored: checkpoints, curves, reports

# ---------------------------------------------------------------- reproducibility
SEED = 42

# ---------------------------------------------------------------- data
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

NUM_WORKERS = 4

# ---------------------------------------------------------------- training
MODEL_NAME = "mobilenet_v3_large"
BATCH_SIZE = 64

# Stage 1 - frozen backbone, head only
FREEZE_EPOCHS = 2
LR_HEAD = 1e-3

# Stage 2 - full fine-tuning
FINETUNE_EPOCHS = 10
LR_FINETUNE = 1e-4
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 1e-4

# Smoke test: tiny subset, catches bugs in minutes
SMOKE_IMAGES = 500
SMOKE_EPOCHS = 1


def pick_device() -> str:
    """cuda -> mps -> cpu, same order on every member's machine."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
