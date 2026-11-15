"""Classical Laplacian pyramid (Burt & Adelson, 1987).

The pyramid uses a fixed 5x5 separable binomial kernel [1,4,6,4,1]/16,
which is the standard choice for Laplacian pyramid decomposition and is
invertible up to floating point error.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _binomial_kernel(device, dtype, channels: int = 3) -> torch.Tensor:
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype)
    k = k / k.sum()
    kernel = k[:, None] * k[None, :]
    return kernel.expand(channels, 1, 5, 5).contiguous()


def gaussian_blur(x: torch.Tensor) -> torch.Tensor:
    c = x.shape[1]
    kernel = _binomial_kernel(x.device, x.dtype, c)
    return F.conv2d(x, kernel, padding=2, groups=c)


def downsample(x: torch.Tensor) -> torch.Tensor:
    return gaussian_blur(F.avg_pool2d(x, 2))


def upsample(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class LaplacianPyramid(nn.Module):
    """Decompose an image into ``levels`` Laplacian residuals + one low band.

    For ``levels=3`` the returned list is ``[L0, L1, L2]`` where
    ``L0`` is full resolution, ``L1`` is 1/2, and ``L2`` (the low band)
    is 1/4 resolution.
    """

    def __init__(self, levels: int = 3) -> None:
        super().__init__()
        self.levels = levels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        gaussians = [x]
        for _ in range(self.levels - 1):
            gaussians.append(downsample(gaussians[-1]))

        bands: list[torch.Tensor] = []
        for i in range(self.levels - 1):
            up = upsample(gaussians[i + 1], gaussians[i].shape[-2:])
            bands.append(gaussians[i] - up)
        bands.append(gaussians[-1])
        return bands

    def reconstruct(self, bands: list[torch.Tensor]) -> torch.Tensor:
        out = bands[-1]
        for i in range(len(bands) - 2, -1, -1):
            out = upsample(out, bands[i].shape[-2:]) + bands[i]
        return out


def laplacian_loss(pred: torch.Tensor, target: torch.Tensor, levels: int = 3) -> torch.Tensor:
    """L1 distance between Laplacian pyramids of prediction and target."""
    pyramid = LaplacianPyramid(levels)
    p_bands = pyramid(pred)
    t_bands = pyramid(target)
    loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
    for p, t in zip(p_bands, t_bands):
        loss = loss + F.l1_loss(p, t)
    return loss / len(p_bands)
