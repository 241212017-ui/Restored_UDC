"""Pre-cut aligned 256x256 LQ/HQ crops into one uint8 memmap.

Random PNG decoding of 1024x2048 frames is the training bottleneck on this
laptop.  Cutting a fixed pool of crops once makes each training sample a
~400 KB memmap read, which is effectively free, and removes all
multiprocessing/pickling pressure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path("D:/UDC_AMFM_Project/data")
OUT = ROOT / "cache"
CROP = 256
PER_IMAGE = 6


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> None:
    rng = np.random.default_rng(2025)
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    for ds in ("Poled", "Toled"):
        for p in sorted((ROOT / "Train" / ds / "HQ").glob("*.png")):
            items.append((ds, p.stem, ROOT / "Train" / ds / "LQ" / p.name, p))
    print(f"{len(items)} pairs -> {len(items) * PER_IMAGE} crops")

    out_path = OUT / f"crops_{CROP}.npy"
    arr = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.uint8, shape=(len(items) * PER_IMAGE, 2, CROP, CROP, 3)
    )
    meta = []
    k = 0
    for ds, stem, lq_path, hq_path in items:
        lq = read_rgb(lq_path)
        hq = read_rgb(hq_path)
        h, w = hq.shape[:2]
        for _ in range(PER_IMAGE):
            y = int(rng.integers(0, h - CROP + 1))
            x = int(rng.integers(0, w - CROP + 1))
            arr[k, 0] = lq[y : y + CROP, x : x + CROP]
            arr[k, 1] = hq[y : y + CROP, x : x + CROP]
            meta.append({"dataset": ds.lower(), "name": f"{ds}_{stem}_{k}", "y": y, "x": x})
            k += 1
        if k % 240 == 0:
            print(f"  {k}/{len(items) * PER_IMAGE}")
    arr.flush()
    del arr
    with open(OUT / f"crops_{CROP}_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    print(f"saved {out_path} ({out_path.stat().st_size / 1024**3:.2f} GB)")


if __name__ == "__main__":
    main()
