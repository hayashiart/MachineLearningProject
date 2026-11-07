"""Stratified 70/15/15 train/val/test split of PlantVillage (color).

Writes plain-text file lists so the exact same split can be reused by every
team member (fair backbone/device comparison) and never changes between runs.

Usage:
    python prepare_data.py
"""

import json
import random
from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split

import config


def collect_samples():
    """Return (relpaths, labels) for every image under DATA_ROOT."""
    if not config.DATA_ROOT.is_dir():
        raise SystemExit(
            f"Dataset not found at {config.DATA_ROOT}\n"
            "Download it first:  git clone --depth 1 "
            "https://github.com/spMohanty/PlantVillage-Dataset "
            f"{config.DATA_ROOT.parent / 'PlantVillage-Dataset'}"
        )

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    relpaths, labels = [], []
    for class_dir in sorted(p for p in config.DATA_ROOT.iterdir() if p.is_dir()):
        for img in sorted(class_dir.iterdir()):
            if img.suffix in exts:
                relpaths.append(f"{class_dir.name}/{img.name}")
                labels.append(class_dir.name)
    return relpaths, labels


def main():
    random.seed(config.SEED)

    relpaths, labels = collect_samples()
    classes = sorted(set(labels))
    counts = Counter(labels)

    print(f"Found {len(relpaths)} images in {len(classes)} classes")
    print(f"Min class size: {min(counts.values())} | Max: {max(counts.values())}")

    # Stratified split: 70 / 15 / 15
    train_p, tmp_p, train_l, tmp_l = train_test_split(
        relpaths, labels, test_size=1 - config.TRAIN_FRAC,
        random_state=config.SEED, stratify=labels,
    )
    val_frac_adj = config.VAL_FRAC / (config.VAL_FRAC + config.TEST_FRAC)
    val_p, test_p, val_l, test_l = train_test_split(
        tmp_p, tmp_l, test_size=1 - val_frac_adj,
        random_state=config.SEED, stratify=tmp_l,
    )

    splits = {"train": train_p, "val": val_p, "test": test_p}
    split_labels = {"train": train_l, "val": val_l, "test": test_l}

    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, paths in splits.items():
        out = config.SPLITS_DIR / f"{name}.txt"
        out.write_text("\n".join(paths) + "\n")
        print(f"{name:5s}: {len(paths):6d} images -> {out}")
        # sanity: stratification kept proportions
        c = Counter(split_labels[name])
        assert min(c.values()) >= 1, f"class missing in {name}"

    meta = {
        "seed": config.SEED,
        "num_classes": len(classes),
        "classes": classes,
        "class_counts_total": dict(sorted(counts.items())),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "source": "spMohanty/PlantVillage-Dataset (raw/color)",
    }
    (config.SPLITS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote meta.json with {len(classes)} class names")
    print("Split is fixed - commit splits/ so all members train on identical data.")


if __name__ == "__main__":
    main()
