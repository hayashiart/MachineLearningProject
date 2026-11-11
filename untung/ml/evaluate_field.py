"""Evaluate a lab-trained checkpoint on field images (PlantDoc).

PlantDoc folders are mapped to PlantVillage classes (field_mapping.json).
The model still predicts over all 38 lab classes; field data is never used
for training or checkpoint selection.

Produces in the run directory:
  field_report.txt, field_results.json, field_confusion.png,
  field_predictions.csv

Usage:
    python evaluate_field.py --checkpoint ../runs/<run>/best.pt
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

import config
import dataset as ds
import model as M

IMG_EXTS = {".jpg", ".jpeg", ".png"}
CONF_THRESHOLD = 0.6  # project plan section 7: "uncertain" below this


def wilson_ci(k: int, n: int, z: float = 1.96):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def collect_field(class_to_idx):
    mapping = json.loads(config.FIELD_MAPPING.read_text())["mapping"]
    if not config.FIELD_ROOT.is_dir():
        raise SystemExit(
            f"Field data not found at {config.FIELD_ROOT}\n"
            "Download: gh api repos/pratikkayal/PlantDoc-Dataset/tarball/master "
            "> plantdoc.tar.gz && tar -xzf plantdoc.tar.gz --strip-components=1 "
            f"-C {config.FIELD_ROOT}")
    samples, skipped = [], []
    for subset in ("train", "test"):
        for class_dir in sorted((config.FIELD_ROOT / subset).iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_to_idx[mapping[class_dir.name]]
            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() not in IMG_EXTS:
                    continue
                try:
                    with Image.open(f) as im:
                        im.verify()
                except Exception:
                    skipped.append(str(f.relative_to(config.FIELD_ROOT)))
                    continue
                samples.append((f, label, subset, class_dir.name))
    return samples, skipped


class FieldDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.transform = ds.build_transforms(train=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _, _ = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


@torch.no_grad()
def predict(net, loader, device):
    net.eval()
    probs = [net(x.to(device)).softmax(1).cpu() for x, _ in loader]
    return torch.cat(probs).numpy()


def plot_confusion(cm, row_names, col_names, title, out_path):
    fig, ax = plt.subplots(figsize=(18, 11))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(col_names)), col_names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(row_names)), row_names, fontsize=7)
    ax.set_xlabel("Predicted (38 lab classes)")
    ax.set_ylabel("True (28 field classes)")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    run_dir = ckpt_path.parent
    device = torch.device(args.device or config.pick_device())
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    net = M.build_model(len(classes), pretrained=False).to(device)
    net.load_state_dict(ckpt["model_state_dict"])

    samples, skipped = collect_field(class_to_idx)
    loader = DataLoader(FieldDataset(samples), batch_size=config.BATCH_SIZE,
                        shuffle=False, num_workers=config.NUM_WORKERS)
    probs = predict(net, loader, device)

    y = np.array([s[1] for s in samples])
    subset = np.array([s[2] for s in samples])
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = pred == y
    n = len(y)

    present = sorted(set(y.tolist()))
    restricted = np.array(present)[probs[:, present].argmax(1)]
    top5 = np.argsort(-probs, 1)[:, :5]
    crop = np.array([c.split("___")[0] for c in classes])
    healthy = np.array([c.endswith("healthy") for c in classes])

    k = int(correct.sum())
    lo, hi = wilson_ci(k, n)
    test_mask = subset == "test"
    accepted = conf >= CONF_THRESHOLD
    pred_counts = np.bincount(pred, minlength=len(classes))
    top_pred = [(classes[i], int(pred_counts[i]))
                for i in np.argsort(-pred_counts)[:5]]

    results = {
        "checkpoint": str(ckpt_path),
        "model": ckpt.get("model_name"),
        "train_data": ckpt.get("train_data", "lab train split (70%)"),
        "field_images": n,
        "field_skipped_unreadable": len(skipped),
        "field_classes": len(present),
        "top1_accuracy": round(float(correct.mean()), 4),
        "top1_correct": k,
        "top1_ci95_wilson": [round(lo, 4), round(hi, 4)],
        "top3_accuracy": round(float((top5[:, :3] == y[:, None]).any(1).mean()), 4),
        "top5_accuracy": round(float((top5 == y[:, None]).any(1).mean()), 4),
        "macro_f1_field_classes": round(float(
            f1_score(y, pred, labels=present, average="macro", zero_division=0)), 4),
        "restricted_28way_accuracy": round(float((restricted == y).mean()), 4),
        "crop_accuracy": round(float((crop[pred] == crop[y]).mean()), 4),
        "healthy_vs_diseased_accuracy": round(float((healthy[pred] == healthy[y]).mean()), 4),
        "plantdoc_official_test_top1": round(float(correct[test_mask].mean()), 4),
        "plantdoc_official_test_n": int(test_mask.sum()),
        "chance_top1_38way": round(1 / len(classes), 4),
        "mean_conf_correct": round(float(conf[correct].mean()), 4),
        "mean_conf_wrong": round(float(conf[~correct].mean()), 4),
        "conf_threshold": CONF_THRESHOLD,
        "coverage_at_threshold": round(float(accepted.mean()), 4),
        "accuracy_at_threshold": round(float(correct[accepted].mean()), 4) if accepted.any() else None,
        "wrong_but_confident": int((~correct & accepted).sum()),
        "most_predicted_classes": top_pred,
        "lab_val_acc_at_save": ckpt.get("val_acc"),
    }

    names = [classes[i] for i in present]
    report = classification_report(y, pred, labels=present, target_names=names,
                                   digits=4, zero_division=0)
    summary = "\n".join(f"{k_}: {v}" for k_, v in results.items())
    (run_dir / "field_report.txt").write_text(summary + "\n\n" + report)
    (run_dir / "field_results.json").write_text(json.dumps(results, indent=2))

    cm = confusion_matrix(y, pred, labels=list(range(len(classes))))[present]
    plot_confusion(cm, names, classes,
                   f"Field confusion - trained on {results['train_data']}, "
                   f"tested on PlantDoc (n={n})", run_dir / "field_confusion.png")

    with open(run_dir / "field_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "subset", "true", "pred", "confidence", "correct"])
        for (p, lbl, sub, _), pr, cf in zip(samples, pred, conf):
            w.writerow([p.relative_to(config.FIELD_ROOT), sub, classes[lbl],
                        classes[pr], f"{cf:.4f}", int(pr == lbl)])

    print(summary + "\n\n" + report)
    print(f"Saved field_* artifacts -> {run_dir}")


if __name__ == "__main__":
    main()
