"""Training loop: MobileNetV3-Large on PlantVillage, Apple GPU via MPS.

Two stages (project plan, section 6):
  1. freeze backbone, train the new head       (2 epochs, lr 1e-3)
  2. unfreeze everything, fine-tune            (10 epochs, lr 1e-4,
     cosine schedule, label smoothing 0.1)

Saves best.pt whenever validation accuracy improves, plus history.json and
training curves.

Usage:
    python train.py                 # full run
    python train.py --smoke         # 500 images, 1 epoch - bug check
    python train.py --full-lab      # 100% PlantVillage, no val, keep last epoch
    python train.py --device cpu    # force a device
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone

if sys.platform == "darwin":
    # Detached/background processes on macOS get a low QoS class, which
    # deprioritises their Metal command queues on the GPU scheduler.
    # Raise this process to USER_INTERACTIVE before torch initialises MPS.
    import ctypes
    try:
        ctypes.CDLL("libc.dylib").pthread_set_qos_class_self_np(0x21, 0)
    except OSError:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import config
import dataset as ds
import model as M


def set_seed(seed: int):
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool,
              tag: str = ""):
    """One pass over loader. Metrics accumulate on-device; a single sync at
    the end. Per-batch .item() would flush+wait the MPS queue every batch
    (~1000x per epoch) and serialise the CPU/GPU pipeline."""
    model.train() if train else model.eval()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.detach() * x.size(0)
            correct += (out.argmax(1) == y).sum()
            n += x.size(0)
            if (i + 1) % 100 == 0:
                print(f"  {tag} {i + 1}/{len(loader)} batches", flush=True)
    return (total_loss / n).item(), (correct / n).item()


def plot_curves(history, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ep = [h["epoch"] for h in history]
    has_val = history[0]["val_loss"] is not None

    axes[0].plot(ep, [h["train_loss"] for h in history], label="train")
    if has_val:
        axes[0].plot(ep, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

    key = "val_acc" if has_val else "train_acc"
    axes[1].plot(ep, [h[key] for h in history], color="green")
    axes[1].set_title("Validation accuracy" if has_val else "Train accuracy")
    axes[1].set_xlabel("epoch")

    axes[2].plot(ep, [h["lr"] for h in history], color="red")
    axes[2].set_title("Learning rate"); axes[2].set_xlabel("epoch")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help=f"{config.SMOKE_IMAGES} images, {config.SMOKE_EPOCHS} epoch")
    ap.add_argument("--full-lab", action="store_true",
                    help="train on train+val+test (all lab data); no validation, "
                         "fixed recipe, last-epoch checkpoint")
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    set_seed(config.SEED)
    device = torch.device(args.device or config.pick_device())
    prefix = ("smoke_" if args.smoke else
              "fulllab_mobilenetv3_mps_" if args.full_lab else "mobilenetv3_mps_")
    run_name = args.run_name or prefix + \
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = config.RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device} | run={run_name} | batch={args.batch_size}")
    if device.type == "mps":
        print("MPS backend:", torch.backends.mps.is_available(),
              "| recommended_max_memory:",
              torch.mps.recommended_max_memory() // 2**20, "MiB")

    # ---------------------------------------------------------------- data
    limit = config.SMOKE_IMAGES if args.smoke else None
    train_split = "train+val+test" if args.full_lab else "train"
    train_data = ("full PlantVillage (100%)" if args.full_lab
                  else "PlantVillage train split (70%)")
    train_loader = ds.make_loader(train_split, args.batch_size, max_images=limit)
    val_loader = (None if args.full_lab else
                  ds.make_loader("val", args.batch_size, max_images=limit))
    classes = train_loader.dataset.classes
    num_classes = len(classes)
    print(f"train={train_split} ({len(train_loader.dataset)} images, "
          f"{len(train_loader)} batches) | "
          f"val batches={len(val_loader) if val_loader else 0} "
          f"| classes={num_classes}")

    # ---------------------------------------------------------------- model
    net = M.build_model(num_classes, pretrained=not args.smoke).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    history = []
    best_acc = 0.0
    t_start = time.time()

    def stage(name, epochs, lr, sched_factory=None):
        nonlocal best_acc
        trainable = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr,
                                      weight_decay=config.WEIGHT_DECAY)
        scheduler = (sched_factory(optimizer) if sched_factory
                     else torch.optim.lr_scheduler.ConstantLR(optimizer, 1.0))
        total, trainable_n = M.count_params(net)
        print(f"\n=== {name}: {epochs} epochs, lr={lr}, "
              f"params {trainable_n:,}/{total:,} trainable ===")

        for ep in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(net, train_loader, criterion,
                                        optimizer, device, train=True,
                                        tag=f"{name} {ep}/{epochs} train")
            if val_loader is not None:
                va_loss, va_acc = run_epoch(net, val_loader, criterion,
                                            optimizer, device, train=False,
                                            tag=f"{name} {ep}/{epochs} val")
            else:
                va_loss = va_acc = None
            cur_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            dt = time.time() - t0

            history.append({
                "epoch": len(history) + 1, "stage": name,
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": va_loss, "val_acc": va_acc,
                "lr": cur_lr, "epoch_seconds": round(dt, 1),
            })
            star = ""
            # Without validation the recipe is fixed up front, so the last
            # epoch is the model: no selection on any held-out data.
            if va_acc is None or va_acc > best_acc:
                if va_acc is not None:
                    best_acc = va_acc
                torch.save({
                    "model_state_dict": net.state_dict(),
                    "classes": classes,
                    "img_size": config.IMG_SIZE,
                    "mean": config.IMAGENET_MEAN,
                    "std": config.IMAGENET_STD,
                    "model_name": config.MODEL_NAME,
                    "val_acc": va_acc,
                    "epoch": len(history),
                    "device_trained_on": str(device),
                    "train_data": train_data,
                    "selection": "best lab val acc" if val_loader else "last epoch",
                }, run_dir / "best.pt")
                star = " *saved best.pt"
            va_txt = (f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
                      if va_acc is not None else "no val")
            print(f"[{name} {ep}/{epochs}] train_loss={tr_loss:.4f} "
                  f"train_acc={tr_acc:.4f} | {va_txt} | {dt:.0f}s{star}")

    if args.smoke:
        stage("smoke", config.SMOKE_EPOCHS, config.LR_HEAD)
    else:
        # Stage 1: frozen backbone, head only
        M.freeze_backbone(net)
        stage("freeze", config.FREEZE_EPOCHS, config.LR_HEAD)

        # Stage 2: full fine-tune with cosine decay
        M.unfreeze_backbone(net)
        stage("finetune", config.FINETUNE_EPOCHS, config.LR_FINETUNE,
              lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
                  opt, T_max=config.FINETUNE_EPOCHS))

    total_s = time.time() - t_start

    # ---------------------------------------------------------------- save
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_curves(history, run_dir / "curves.png")

    summary = {
        "run_name": run_name,
        "model": config.MODEL_NAME,
        "device": str(device),
        "smoke": args.smoke,
        "train_data": train_data,
        "train_images": len(train_loader.dataset),
        "seed": config.SEED,
        "batch_size": args.batch_size,
        "best_val_acc": best_acc if val_loader else None,
        "total_train_seconds": round(total_s, 1),
        "epochs": len(history),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nDone. best_val_acc={best_acc if val_loader else 'n/a'} "
          f"| total {total_s/60:.1f} min")
    print(f"Artifacts in {run_dir}")


if __name__ == "__main__":
    main()
