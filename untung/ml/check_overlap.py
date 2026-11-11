"""Near-duplicate check between field (PlantDoc) and lab (PlantVillage) images.

64-bit difference hash (dHash); a pair with Hamming distance <= THRESHOLD is
reported as a candidate duplicate. Candidates are then re-checked with a
64x64 grayscale pixel correlation: only corr >= CONFIRM_CORR counts as a
confirmed duplicate (dHash alone also fires on shared plain backgrounds).

Writes ../results/field_overlap.json and ../results/field_overlap_top.jpg
(the closest candidate pairs, field left / lab right).

Usage:
    python check_overlap.py
"""

import json
from multiprocessing import Pool

import numpy as np
from PIL import Image

import config

THRESHOLD = 6
CORR_SIZE = 64
CONFIRM_CORR = 0.95
TOP_PAIRS = 3


def dhash(path):
    try:
        with Image.open(path) as im:
            g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR),
                           dtype=np.int16)
    except Exception:
        return None
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def hash_all(paths):
    with Pool(4) as pool:
        return pool.map(dhash, paths, chunksize=256)


def popcount64(x):
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return (x * 0x0101010101010101) >> 56


def gray64(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((CORR_SIZE, CORR_SIZE),
                                                 Image.BICUBIC),
                          dtype=np.float64)


def pixel_corr(pa, pb):
    a, b = gray64(pa), gray64(pb)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def field_class_of(rel_path, mapping):
    folder = rel_path.split("/")[1]
    return mapping.get(folder)


def plot_pairs(matches, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = matches[:TOP_PAIRS]
    fig, axes = plt.subplots(len(rows), 2, figsize=(8, 4 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for ax, m in zip(axes, rows):
        for a, key in zip(ax, ("field", "lab")):
            base = (config.FIELD_ROOT if key == "field" else config.DATA_ROOT)
            a.imshow(Image.open(base / m[key]))
            a.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    lab = sorted(p for p in config.DATA_ROOT.rglob("*") if p.is_file()
                 and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    field = sorted(p for s in ("train", "test")
                   for p in (config.FIELD_ROOT / s).rglob("*") if p.is_file()
                   and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"hashing {len(lab)} lab + {len(field)} field images")
    lab_h = hash_all(lab)
    field_h = hash_all(field)

    lab_ok = [(p, h) for p, h in zip(lab, lab_h) if h is not None]
    lab_arr = np.array([h for _, h in lab_ok], dtype=np.uint64)

    mapping = json.loads(config.FIELD_MAPPING.read_text())["mapping"]

    matches = []
    for fp, fh in zip(field, field_h):
        if fh is None:
            continue
        d = popcount64(lab_arr ^ np.uint64(fh)).astype(np.int64)
        j = int(d.argmin())
        if d[j] <= THRESHOLD:
            lp = lab_ok[j][0]
            rel_f = str(fp.relative_to(config.FIELD_ROOT))
            rel_l = str(lp.relative_to(config.DATA_ROOT))
            matches.append({
                "field": rel_f,
                "lab": rel_l,
                "hamming": int(d[j]),
                "pixel_corr": round(pixel_corr(fp, lp), 4),
                "same_class": field_class_of(rel_f, mapping) == rel_l.split("/")[0],
            })

    matches.sort(key=lambda m: (m["hamming"], -m["pixel_corr"]))
    confirmed = [m for m in matches if m["pixel_corr"] >= CONFIRM_CORR]

    out = {
        "method": f"64-bit dHash, nearest lab image, Hamming <= {THRESHOLD}",
        "lab_images": len(lab_ok),
        "field_images": sum(h is not None for h in field_h),
        "candidate_duplicates": len(matches),
        "verification": (f"candidates re-checked with {CORR_SIZE}x{CORR_SIZE} "
                         f"grayscale pixel correlation; confirmed = corr >= "
                         f"{CONFIRM_CORR}"),
        "confirmed_duplicates": confirmed,
        "matches": matches,
    }
    dest = config.UNTUNG_DIR / "results" / "field_overlap.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"{len(matches)} candidates, {len(confirmed)} confirmed -> {dest}")
    if matches:
        print(f"max pixel_corr among candidates: "
              f"{max(m['pixel_corr'] for m in matches):.3f}")
        fig_path = config.UNTUNG_DIR / "results" / "field_overlap_top.jpg"
        plot_pairs(matches, fig_path)
        print(f"top pairs -> {fig_path}")
    for m in matches[:15]:
        print(m)


if __name__ == "__main__":
    main()
