"""Full-reference image quality metrics: PSNR, SSIM, LPIPS, DISTS."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# PSNR
# --------------------------------------------------------------------------- #
def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / (mse + eps))


# --------------------------------------------------------------------------- #
# SSIM (11x11 Gaussian window, sigma=1.5, per-channel then averaged)
# --------------------------------------------------------------------------- #
def _gaussian_window(window_size: int = 11, sigma: float = 1.5, channels: int = 3, device=None, dtype=None):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(0)
    window = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return window.expand(channels, 1, window_size, window_size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    c = pred.shape[1]
    win = _gaussian_window(window_size, sigma, c, pred.device, pred.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(pred, win, padding=pad, groups=c)
    mu2 = F.conv2d(target, win, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, win, padding=pad, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, win, padding=pad, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, win, padding=pad, groups=c) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    )
    return ssim_map.mean(dim=(1, 2, 3))


# --------------------------------------------------------------------------- #
# LPIPS / DISTS (lazy-loaded, optional)
# --------------------------------------------------------------------------- #
class LPIPSMetric:
    def __init__(self, device: torch.device) -> None:
        import lpips

        self.model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.clamp(0, 1) * 2 - 1
        t = target.clamp(0, 1) * 2 - 1
        return self.model(p, t).flatten()


class DISTSMetric:
    def __init__(self, device: torch.device) -> None:
        try:
            from DISTS_pytorch import DISTS  # pip install dists-pytorch
        except ImportError:  # pragma: no cover - alternate package layout
            from dists_pytorch import DISTS

        self.model = DISTS().to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.model(pred.clamp(0, 1), target.clamp(0, 1)).flatten()


class MetricSuite:
    """Accumulates per-image metrics and returns dataset means."""

    def __init__(self, device: torch.device, use_lpips: bool = True, use_dists: bool = True) -> None:
        self.device = device
        self.lpips = None
        self.dists = None
        if use_lpips:
            try:
                self.lpips = LPIPSMetric(device)
            except Exception as exc:  # pragma: no cover - optional dependency
                print(f"[warn] LPIPS disabled: {exc}")
        if use_dists:
            try:
                self.dists = DISTSMetric(device)
            except Exception as exc:  # pragma: no cover - optional dependency
                print(f"[warn] DISTS disabled: {exc}")
        self.reset()

    def reset(self) -> None:
        self.values: dict[str, list[float]] = {k: [] for k in ("PSNR", "SSIM", "LPIPS", "DISTS")}

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        pred = pred.to(self.device).clamp(0, 1)
        target = target.to(self.device).clamp(0, 1)
        out = {
            "PSNR": psnr(pred, target).mean().item(),
            "SSIM": ssim(pred, target).mean().item(),
        }
        if self.lpips is not None:
            out["LPIPS"] = self.lpips(pred, target).mean().item()
        if self.dists is not None:
            out["DISTS"] = self.dists(pred, target).mean().item()
        for k, v in out.items():
            self.values[k].append(v)
        return out

    def summary(self) -> dict[str, float]:
        return {k: float(sum(v) / len(v)) for k, v in self.values.items() if len(v) > 0}
