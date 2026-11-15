"""LPUDC-style Laplacian pyramid network with pluggable multi-scale fusion.

The network follows the base paper (Ding et al., Neurocomputing 2025):

    Low-frequency branch  -> encoder/decoder on the coarsest pyramid band
    High-frequency branch -> dilated residual stack on each Laplacian band
    Fusion module         -> cross-scale interaction between adjacent levels
    Pyramid sum           -> reconstruction

The only architectural variable is the fusion module, which makes the
DGF-vs-AMFM comparison a strictly controlled ablation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import HighFrequencyBranch, LowFrequencyBranch, build_fusion
from .pyramid import LaplacianPyramid


class LPUDC(nn.Module):
    def __init__(
        self,
        width: int = 56,
        levels: int = 3,
        fusion: str = "amfm",
        num_drb: int = 2,
    ) -> None:
        super().__init__()
        if levels < 2:
            raise ValueError("levels must be >= 2")
        self.levels = levels
        self.fusion_mode = fusion
        self.pyramid = LaplacianPyramid(levels)

        self.low_branch = LowFrequencyBranch(width)
        self.low_head = nn.Conv2d(width, 3, 3, padding=1)

        n_high = levels - 1
        self.high_branches = nn.ModuleList(
            [HighFrequencyBranch(width, num_drb=num_drb) for _ in range(n_high)]
        )
        self.fusions = nn.ModuleList([build_fusion(fusion, width) for _ in range(n_high)])
        self.guide_proj = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_high)])
        self.high_heads = nn.ModuleList([nn.Conv2d(width, 3, 3, padding=1) for _ in range(n_high)])

        # Small identity-initialised refinement of the reconstructed image.
        self.final = nn.Conv2d(3, 3, 3, padding=1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands = self.pyramid(x)                 # [fine ... coarse]
        low = self.low_branch(bands[-1])        # coarsest features
        low_img = self.low_head(low)            # coarsest restored image

        guide = low
        level_outputs: list[tuple[int, torch.Tensor]] = []
        for idx in reversed(range(self.levels - 1)):
            target_size = bands[idx].shape[-2:]
            g = self.guide_proj[idx](
                F.interpolate(guide, size=target_size, mode="bilinear", align_corners=False)
            )
            h = self.high_branches[idx](bands[idx])
            fused = self.fusions[idx](g, h)
            level_outputs.append((idx, self.high_heads[idx](fused)))
            guide = fused

        out = low_img
        for idx, img in level_outputs:          # coarse -> fine
            out = F.interpolate(out, size=img.shape[-2:], mode="bilinear", align_corners=False) + img

        return out + self.final(out)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        self.eval()
        y = self.forward(x)
        return y.clamp(0, 1) if clamp else y


def build_model(config: dict) -> LPUDC:
    return LPUDC(
        width=config.get("width", 56),
        levels=config.get("levels", 3),
        fusion=config.get("fusion", "amfm"),
        num_drb=config.get("num_drb", 2),
    )
