"""Measure parameters, VRAM and throughput for each fusion mode."""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import LPUDC
from src.utils import count_parameters_m, describe_device


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {describe_device()}")
    for fusion in ["none", "concat", "dgf", "amfm"]:
        torch.cuda.empty_cache() if device.type == "cuda" else None
        model = LPUDC(width=56, levels=3, fusion=fusion).to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        x = torch.rand(4, 3, 256, 256, device=device)
        y = torch.rand(4, 3, 256, 256, device=device)

        # warm-up
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = (model(x) - y).abs().mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        n = 12
        for _ in range(n):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = (model(x) - y).abs().mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n
        peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0
        print(f"{fusion:7s} params={count_parameters_m(model):6.3f}M  {dt*1000:7.1f} ms/it  "
              f"{1/dt:6.2f} it/s  peak={peak:5.2f} GB")
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
