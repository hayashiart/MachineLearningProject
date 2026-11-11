# Model / Device Comparison — LeafScan

Same dataset (PlantVillage, 38 classes), **same fixed split** (`untung/ml/splits/`,
seed 42, stratified 70/15/15), same recipe (freeze head 2 epochs lr 1e-3 →
fine-tune 10 epochs lr 1e-4 cosine, label smoothing 0.1, AdamW, batch 64,
float32, 224×224, ImageNet normalisation).

| Metric (field yardstick: PlantDoc, n = 2,572) | **MobileNetV3-Large**<br>MPS — Untung (M5 Pro, 24 GB) | **EfficientNet-B0**<br>MPS — Mustafa (M1 Pro) | **EfficientNet-B0**<br>CUDA — Sebastien |
|---|---|---|---|
| **Field top-1** | **25.04 %** [23.40–26.75] | TBD | TBD |
| Field top-5 | 54.94 % | TBD | TBD |
| Field macro F1 (28 classes) | 0.2225 | TBD | TBD |
| Crop correct | 48.06 % | TBD | TBD |
| Healthy vs diseased | 77.45 % | TBD | TBD |
| Parameters | 4.25 M | ~5.3 M | ~5.3 M |
| Checkpoint size | 16.4 MB | TBD | TBD |
| Training time (lab) | 18.1 min (12 ep) | TBD | TBD |
| Inference (batch 1) | **4.57 ms/img** | TBD | TBD |
| torch version | 2.15.0 | TBD | TBD |

In-domain (lab → lab) accuracy is **not** a comparison metric: on PlantVillage
it saturates near 100 % for any modern backbone, so it cannot discriminate
between models and rewards fitting lab artifacts. Each member keeps in-domain
numbers in a local appendix only (mine:
`report/experiment_log_lab_appendix.pdf`, not in git).

Raw numbers: `results/lab_mobilenetv3_mps/field_results.json`.

## Lab → field (train PlantVillage, test PlantDoc 2,572 images)

Fill in with your own model, evaluated on the **same** field set and mapping
(`ml/field_mapping.json`, PlantDoc `train/` + `test/`, all 28 classes).

| Metric | **MobileNetV3-L** A<br>70 % lab | **MobileNetV3-L** B<br>100 % lab | **EfficientNet-B0**<br>MPS — Mustafa | **EfficientNet-B0**<br>CUDA — Sebastien |
|---|---|---|---|---|
| **Field top-1** | **25.04 %** | **25.43 %** | TBD | TBD |
| 95 % CI (Wilson) | 23.40–26.75 | 23.78–27.15 | TBD | TBD |
| Field top-5 | 54.94 % | 53.89 % | TBD | TBD |
| Field macro F1 | 0.2225 | 0.2255 | TBD | TBD |
| Crop correct | 48.06 % | 47.28 % | TBD | TBD |
| Healthy vs diseased | 77.45 % | 78.97 % | TBD | TBD |
| PlantDoc official test (n=236) | 31.78 % | 33.05 % | TBD | TBD |
| Training time | 18.1 min | 31.5 min | TBD | TBD |

A vs B: paired exact McNemar p = 0.54 → no significant difference; more lab
data does not close the domain gap.

## Notes for the report

- PlantVillage is photographed on plain backgrounds in controlled conditions,
  so in-domain test accuracy saturates and overstates real-world performance
  (project plan §5); it is therefore not reported — lab → field accuracy is
  ~25 % (tables above).
- MobileNetV3-Large is ~20 % smaller in parameters and its batch-1 latency on
  MPS (4.6 ms) is comfortably inside the server's budget.
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
