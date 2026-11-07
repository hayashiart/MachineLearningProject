# Model / Device Comparison — LeafScan

Same dataset (PlantVillage, 38 classes), **same fixed split** (`untung/ml/splits/`,
seed 42, stratified 70/15/15), same recipe (freeze head 2 epochs lr 1e-3 →
fine-tune 10 epochs lr 1e-4 cosine, label smoothing 0.1, AdamW, batch 64,
float32, 224×224, ImageNet normalisation).

| Metric | **MobileNetV3-Large**<br>MPS — Untung (M5 Pro, 24 GB) | **EfficientNet-B0**<br>MPS — Mustafa (M1 Pro) | **EfficientNet-B0**<br>CUDA — Sebastien |
|---|---|---|---|
| Test accuracy | **99.67 %** | TBD | TBD |
| Macro F1 | 0.9953 | TBD | TBD |
| Best val accuracy | 99.74 % | TBD | TBD |
| Parameters | 4.25 M | ~5.3 M | ~5.3 M |
| Checkpoint size | 16.4 MB | TBD | TBD |
| Total training time | **19.7 min** (12 ep) | TBD | TBD |
| Train time / epoch | 34 s (frozen) → 110 s (fine-tune) | TBD | TBD |
| Inference (batch 1) | **4.73 ms/img** | TBD | TBD |
| Mean top-1 confidence | 0.899 | TBD | TBD |
| torch version | 2.15.0 | TBD | TBD |

Raw numbers: `results/mobilenetv3_mps_full/results.json`.

## Notes for the report

- PlantVillage is photographed on plain backgrounds in controlled conditions, so
  ~99.7 % test accuracy overstates real-world performance (project plan §5).
  The real-world set (PlantDoc / phone photos) is evaluated separately by Member B.
- MobileNetV3-Large is ~20 % smaller in parameters and its batch-1 latency on
  MPS (4.7 ms) is comfortably inside the server's budget.
- Efficiency: full training in under 20 minutes on an M5 Pro means the whole
  comparison + ablation matrix is cheap to re-run.

## MPS performance gotchas (learned the hard way — read before training on a Mac)

1. **Do not launch training with `nohup ... &` from an interactive zsh.**
   zsh's `BG_NICE` backgrounds jobs at lower priority, and macOS then assigns
   the process a Background QoS class — Metal command queues get deprioritised
   on the GPU scheduler. Measured: **14+ min/epoch instead of 34 s (~25×)**.
   Fixes: run in the foreground, inside `screen`/`tmux`, `setopt nobgnice`, or
   set QoS explicitly (done in `train.py`):
   ```python
   ctypes.CDLL("libc.dylib").pthread_set_qos_class_self_np(0x21, 0)  # USER_INTERACTIVE
   ```
2. **Don't call `.item()` every batch.** Each call is a full MPS
   commit-and-wait (GPU pipeline flush). Accumulate metrics on-device and sync
   once per epoch (done in `train.py:run_epoch`).
3. `channels_last` was **slower** on M5 Pro / torch 2.15 for this model
   (413 vs 166 ms/batch) — plain NCHW is correct here.
4. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` (done in `config.py`).
5. Pretrained weights download through torch hub can stall; download
   `mobilenet_v3_large-5c1a4163.pth` with curl into
   `~/.cache/torch/hub/checkpoints/` if it hangs.
