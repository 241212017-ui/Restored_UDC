"""Datasets for POLED and TOLED under-display-camera restoration.

Training data are real paired PNGs (``LQ`` / ``HQ``).  Validation and test
splits are distributed as MATLAB ``.mat`` tensors of shape (N, 1024, 2048, 3).
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

cv2.setNumThreads(0)

DATASET_NAMES = ("poled", "toled")


def _read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float() / 255.0


# --------------------------------------------------------------------------- #
# Training set (real PNG pairs)
# --------------------------------------------------------------------------- #
class UDCTrainDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        datasets=("Poled", "Toled"),
        crop: int | None = 256,
        augment: bool = True,
        repeats: int = 1,
    ) -> None:
        self.root = Path(root)
        self.crop = crop
        self.augment = augment
        self.items: list[tuple[str, str, str, str]] = []
        for ds in datasets:
            lq_dir = self.root / ds / "LQ"
            hq_dir = self.root / ds / "HQ"
            stems = sorted(
                (p.stem for p in hq_dir.glob("*.png")),
                key=lambda s: (len(s), s),
            )
            for stem in stems:
                lq = lq_dir / f"{stem}.png"
                hq = hq_dir / f"{stem}.png"
                if lq.exists() and hq.exists():
                    self.items.append((str(lq), str(hq), stem, ds.lower()))
        if not self.items:
            raise RuntimeError(f"No training pairs found under {self.root}")
        self.repeats = max(1, int(repeats))

    def __len__(self) -> int:
        return len(self.items) * self.repeats

    def _random_crop(self, lq: np.ndarray, hq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = hq.shape[:2]
        if isinstance(self.crop, (tuple, list)):
            ch, cw = int(self.crop[0]), int(self.crop[1])
        else:
            ch = cw = int(self.crop)
        if h < ch or w < cw:
            lq = cv2.resize(lq, (max(cw, w), max(ch, h)), interpolation=cv2.INTER_CUBIC)
            hq = cv2.resize(hq, (max(cw, w), max(ch, h)), interpolation=cv2.INTER_CUBIC)
            h, w = hq.shape[:2]
        y = np.random.randint(0, h - ch + 1)
        x = np.random.randint(0, w - cw + 1)
        return lq[y : y + ch, x : x + cw], hq[y : y + ch, x : x + cw]

    def __getitem__(self, index: int):
        lq_path, hq_path, name, ds = self.items[index % len(self.items)]
        lq = _read_rgb(lq_path)
        hq = _read_rgb(hq_path)

        if self.crop:
            lq, hq = self._random_crop(lq, hq)

        if self.augment:
            if np.random.rand() < 0.5:
                lq, hq = lq[:, ::-1], hq[:, ::-1]
            if np.random.rand() < 0.5:
                lq, hq = lq[::-1], hq[::-1]
            k = np.random.randint(0, 4)
            if k:
                lq = np.ascontiguousarray(np.rot90(lq, k))
                hq = np.ascontiguousarray(np.rot90(hq, k))

        lq, hq = np.ascontiguousarray(lq), np.ascontiguousarray(hq)
        return {"lq": _to_tensor(lq), "gt": _to_tensor(hq), "name": name, "dataset": ds}


# --------------------------------------------------------------------------- #
# Validation / test sets (.mat tensors)
# --------------------------------------------------------------------------- #
_MEMMAP_CACHE: dict[str, np.ndarray] = {}


def _open_memmap(path: str | Path) -> np.ndarray:
    """Open (and cache per process) a memory-mapped .npy array.

    Crucially the dataset stores only *paths*, never arrays, so DataLoader
    workers do not try to pickle hundreds of megabytes of image data.
    """
    key = str(path)
    if key not in _MEMMAP_CACHE:
        _MEMMAP_CACHE[key] = np.load(key, mmap_mode="r")
    return _MEMMAP_CACHE[key]


class CachedCropDataset(Dataset):
    """Training set backed by a pre-cut uint8 crop memmap (shape N,2,H,W,3)."""

    def __init__(self, cache_path: str | Path, augment: bool = True) -> None:
        self.cache_path = str(cache_path)
        self.augment = augment
        self.n = int(np.load(self.cache_path, mmap_mode="r").shape[0])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        arr = _open_memmap(self.cache_path)
        lq = np.array(arr[index, 0], copy=True)
        hq = np.array(arr[index, 1], copy=True)
        if self.augment:
            if np.random.rand() < 0.5:
                lq, hq = lq[:, ::-1], hq[:, ::-1]
            if np.random.rand() < 0.5:
                lq, hq = lq[::-1], hq[::-1]
            k = np.random.randint(0, 4)
            if k:
                lq = np.ascontiguousarray(np.rot90(lq, k))
                hq = np.ascontiguousarray(np.rot90(hq, k))
        return {
            "lq": _to_tensor(np.ascontiguousarray(lq)),
            "gt": _to_tensor(np.ascontiguousarray(hq)),
            "name": str(index),
            "dataset": "cache",
        }


class UDCMatDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str = "test",
        datasets=("poled", "toled"),
        crop: int | None = None,
        crop_size: tuple[int, int] | None = None,
        center: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.crop = crop
        self.crop_size = crop_size
        self.center = center
        self.items: list[tuple[str, str, int, str, str]] = []

        for ds in datasets:
            npy_display = self.data_dir / f"{ds}_{split}_display.npy"
            npy_gt = self.data_dir / f"{ds}_{split}_gt.npy"
            if npy_display.exists() and npy_gt.exists():
                n = np.load(npy_display, mmap_mode="r").shape[0]
                for i in range(n):
                    self.items.append((str(npy_display), str(npy_gt), i, f"{ds}_{split}_{i:03d}", ds))
            else:
                import scipy.io as sio

                display = sio.loadmat(self.data_dir / f"{ds}_{split}_display.mat")[f"{split}_display"]
                gt = sio.loadmat(self.data_dir / f"{ds}_{split}_gt.mat")[f"{split}_gt"]
                for i in range(display.shape[0]):
                    arr_d = self.data_dir / f"{ds}_{split}_display{i:03d}.npy"
                    arr_g = self.data_dir / f"{ds}_{split}_gt{i:03d}.npy"
                    if not arr_d.exists():
                        np.save(arr_d, np.ascontiguousarray(display[i]))
                        np.save(arr_g, np.ascontiguousarray(gt[i]))
                    self.items.append((str(arr_d), str(arr_g), 0, f"{ds}_{split}_{i:03d}", ds))

    def __len__(self) -> int:
        return len(self.items)

    def _crop_pair(self, lq: np.ndarray, gt: np.ndarray):
        h, w = lq.shape[:2]
        if self.crop_size is not None:
            ch, cw = self.crop_size
            if self.center:
                y, x = (h - ch) // 2, (w - cw) // 2
            else:
                y, x = np.random.randint(0, h - ch + 1), np.random.randint(0, w - cw + 1)
            return lq[y : y + ch, x : x + cw], gt[y : y + ch, x : x + cw]
        if self.crop is not None:
            ch = cw = self.crop
            if self.center:
                y, x = (h - ch) // 2, (w - cw) // 2
            else:
                y, x = np.random.randint(0, h - ch + 1), np.random.randint(0, w - cw + 1)
            return lq[y : y + ch, x : x + cw], gt[y : y + ch, x : x + cw]
        return lq, gt

    def __getitem__(self, index: int):
        display_path, gt_path, item_index, name, ds = self.items[index]
        lq = np.asarray(_open_memmap(display_path)[item_index])
        gt = np.asarray(_open_memmap(gt_path)[item_index])
        lq, gt = self._crop_pair(lq, gt)
        return {"lq": _to_tensor(np.ascontiguousarray(lq)),
                "gt": _to_tensor(np.ascontiguousarray(gt)),
                "name": name, "dataset": ds}


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def build_train_loader(config: dict, num_workers: int = 6):
    from torch.utils.data import DataLoader

    cache_path = config["data"].get("crop_cache")
    if cache_path and Path(cache_path).exists():
        ds = CachedCropDataset(cache_path, augment=config["data"].get("augment", True))
    else:
        ds = UDCTrainDataset(
            root=config["data"]["train_root"],
            datasets=config["data"].get("train_datasets", ("Poled", "Toled")),
            crop=config["data"].get("crop", 256),
            augment=config["data"].get("augment", True),
        )
    loader = DataLoader(
        ds,
        batch_size=config["train"].get("batch_size", 4),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader, ds


def build_val_loader(config: dict, split: str = "val", full: bool = False, num_workers: int = 2):
    from torch.utils.data import DataLoader

    crop_size = None if full else tuple(config["data"].get("val_crop", (512, 1024)))
    ds = UDCMatDataset(
        data_dir=config["data"]["mat_root"],
        split=split,
        datasets=config["data"].get("eval_datasets", ("poled", "toled")),
        crop_size=crop_size,
        center=True,
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return loader, ds


def build_test_loaders(config: dict, num_workers: int = 2):
    """One loader per dataset so per-dataset metrics can be reported."""
    from torch.utils.data import DataLoader

    loaders = {}
    for ds_name in config["data"].get("eval_datasets", ("poled", "toled")):
        ds = UDCMatDataset(
            data_dir=config["data"]["mat_root"],
            split="test",
            datasets=(ds_name,),
            crop_size=None,
        )
        loaders[ds_name] = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=False)
    return loaders
