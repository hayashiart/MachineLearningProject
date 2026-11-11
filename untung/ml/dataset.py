"""Dataset, augmentation and DataLoader factories.

Preprocessing is identical to what the inference server must apply:
resize 224x224 + ImageNet normalisation. Augmentation is train-only.
"""

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config


def class_list():
    meta = json.loads((config.SPLITS_DIR / "meta.json").read_text())
    return meta["classes"]


def build_transforms(train: bool):
    if train:
        # Plan: random crop, flips, rotation, colour jitter, light blur
        return transforms.Compose([
            transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


class LeafDataset(Dataset):
    """Reads split file lists (one relative path per line).

    `split` may join several splits with "+", e.g. "train+val+test" = all lab data.
    """

    def __init__(self, split: str, transform=None, max_images: int | None = None):
        relpaths = []
        for name in split.split("+"):
            split_file = config.SPLITS_DIR / f"{name}.txt"
            if not split_file.is_file():
                raise SystemExit(f"{split_file} missing - run prepare_data.py first")
            relpaths += [p for p in split_file.read_text().splitlines() if p]

        self.classes = class_list()
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.root = config.DATA_ROOT

        if max_images is not None:
            # Deterministic subset for the smoke test: spread across classes
            by_class = {}
            for rp in relpaths:
                by_class.setdefault(rp.split("/", 1)[0], []).append(rp)
            per_class = max(1, max_images // len(by_class))
            relpaths = sorted(
                rp for paths in by_class.values() for rp in paths[:per_class]
            )[:max_images]

        self.samples = [(rp, self.class_to_idx[rp.split("/", 1)[0]])
                        for rp in relpaths]
        self.transform = transform or build_transforms(train=split.startswith("train"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rp, label = self.samples[idx]
        img = Image.open(self.root / rp).convert("RGB")
        return self.transform(img), label


def make_loader(split: str, batch_size: int = config.BATCH_SIZE,
                shuffle: bool | None = None, max_images: int | None = None):
    ds = LeafDataset(split, max_images=max_images)
    if shuffle is None:
        shuffle = split.startswith("train")
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=config.NUM_WORKERS, persistent_workers=True,
        pin_memory=False,  # MPS does not benefit from pinned host memory
    )
