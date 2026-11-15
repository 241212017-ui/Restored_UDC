"""Convert the validation/test .mat tensors to memory-mapped .npy files.

This keeps host RAM flat: each image is paged in on demand instead of
loading four ~190 MB MATLAB arrays into every process.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scipy.io as sio


def main() -> None:
    data_dir = Path("D:/UDC_AMFM_Project/data")
    for ds in ("poled", "toled"):
        for split in ("val", "test"):
            for kind in ("display", "gt"):
                mat_path = data_dir / f"{ds}_{split}_{kind}.mat"
                npy_path = data_dir / f"{ds}_{split}_{kind}.npy"
                key = f"{split}_{kind}"
                arr = sio.loadmat(mat_path)[key]
                assert arr.shape == (30, 1024, 2048, 3), arr.shape
                np.save(npy_path, np.ascontiguousarray(arr))
                print(f"{mat_path.name:28s} -> {npy_path.name:28s} {arr.shape} {arr.dtype}")

    # Quick sanity report
    print("\nDataset inventory")
    for ds in ("Poled", "Toled"):
        for kind in ("LQ", "HQ"):
            n = len(list((data_dir / "Train" / ds / kind).glob("*.png")))
            print(f"  train {ds}/{kind}: {n} images")
    for ds in ("poled", "toled"):
        for split in ("val", "test"):
            a = np.load(data_dir / f"{ds}_{split}_display.npy", mmap_mode="r")
            print(f"  {ds} {split}: {a.shape[0]} images @ {a.shape[1]}x{a.shape[2]}")


if __name__ == "__main__":
    main()
