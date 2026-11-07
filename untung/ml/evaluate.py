"""Evaluate best.pt on the held-out test set (used once, at the end).

Produces:
  report.txt          accuracy, per-class precision/recall/F1
  confusion_matrix.png
  results.json        numbers for the backbone/device comparison table
  inference benchmark (ms per image, batch=1, on the training device)

Usage:
    python evaluate.py --checkpoint ../runs/<run>/best.pt
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import config
import dataset as ds
import model as M


def latest_run() -> Path:
    runs = sorted(p for p in config.RUNS_DIR.iterdir()
                  if (p / "best.pt").exists())
    if not runs:
        raise SystemExit("No run with best.pt found - train first")
    return runs[-1]


@torch.no_grad()
def predict_all(net, loader, device):
    net.eval()
    ys, preds, probs = [], [], []
    for x, y in loader:
        out = net(x.to(device))
        p = out.softmax(1)
        preds.append(p.argmax(1).cpu())
        probs.append(p.cpu())
        ys.append(y)
    return (torch.cat(ys).numpy(), torch.cat(preds).numpy(),
            torch.cat(probs).numpy())


def benchmark_inference(net, device, n_warmup=10, n_iter=50):
    """Batch-1 latency, the number the server will actually see."""
    net.eval()
    x = torch.rand(1, 3, config.IMG_SIZE, config.IMG_SIZE, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            net(x)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            net(x)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n_iter * 1000
    return dt


def plot_confusion(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix - MobileNetV3-Large (MPS), test set")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="path to best.pt (default: latest run)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else latest_run() / "best.pt"
    run_dir = ckpt_path.parent
    device = torch.device(args.device or config.pick_device())
    print(f"checkpoint={ckpt_path}\ndevice={device}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    net = M.build_model(len(classes), pretrained=False).to(device)
    net.load_state_dict(ckpt["model_state_dict"])

    loader = ds.make_loader("test", config.BATCH_SIZE, shuffle=False)
    y_true, y_pred, probs = predict_all(net, loader, device)

    acc = (y_true == y_pred).mean()
    report = classification_report(y_true, y_pred, target_names=classes,
                                   digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # mean top-1 confidence and per-image latency
    conf = probs.max(1)
    ms_per_img = benchmark_inference(net, device)
    total_params, _ = M.count_params(net)
    size_mb = ckpt_path.stat().st_size / 2**20
    print(f"\ntest images : {len(y_true)}")
    print(f"accuracy    : {acc:.4f}")
    print(f"mean top-1 confidence: {conf.mean():.4f}")
    print(f"inference   : {ms_per_img:.2f} ms/image (batch=1, {device})")
    print(f"params      : {total_params:,} | checkpoint {size_mb:.1f} MB")
    print("\n" + report)

    (run_dir / "report.txt").write_text(
        f"checkpoint: {ckpt_path}\ndevice: {device}\n"
        f"test images: {len(y_true)}\naccuracy: {acc:.4f}\n"
        f"mean top-1 confidence: {conf.mean():.4f}\n"
        f"inference ms/image (batch=1): {ms_per_img:.2f}\n"
        f"params: {total_params}\ncheckpoint MB: {size_mb:.1f}\n\n" + report)

    plot_confusion(cm, classes, run_dir / "confusion_matrix.png")

    results = {
        "model": config.MODEL_NAME,
        "device": str(device),
        "checkpoint": str(ckpt_path),
        "test_images": int(len(y_true)),
        "test_accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred,
                                           average="weighted")), 4),
        "mean_top1_confidence": round(float(conf.mean()), 4),
        "inference_ms_per_image": round(ms_per_img, 2),
        "params": total_params,
        "checkpoint_mb": round(size_mb, 1),
        "val_acc_at_save": ckpt.get("val_acc"),
    }

    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved report.txt, confusion_matrix.png, results.json -> {run_dir}")


if __name__ == "__main__":
    main()
