"""Building blocks and the four fusion strategies compared in this project.

Fusion strategies (all share the same surrounding network):
    * ``none``   : each pyramid level is processed independently.
    * ``concat`` : low-resolution features are upsampled and concatenated.
    * ``dgf``    : Deep Guided Filtering (the base paper's module).
    * ``amfm``   : Attention-based Multi-Scale Fusion Module (our contribution).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(channels: int) -> nn.Module:
    groups = 8 if channels % 8 == 0 else 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    """Conv -> GroupNorm -> GELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1) -> None:
        super().__init__()
        pad = dilation * (kernel // 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False),
            norm(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DRB(nn.Module):
    """Dilated Residual Block (used by both branches of LPUDC)."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            norm(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            norm(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


# --------------------------------------------------------------------------- #
# Attention primitives (used by AMFM)
# --------------------------------------------------------------------------- #
class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Global average pooling -> bottleneck MLP -> sigmoid gate.  The network
    learns *which feature channels* carry useful structure versus diffraction
    noise.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mlp(self.pool(x))


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention.

    Concatenates channel-wise mean and max maps, then a 7x7 convolution
    predicts a single attention map.  The network learns *which pixels*
    (e.g. diffraction fringes) need more weight.
    """

    def __init__(self, kernel: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class AMFM(nn.Module):
    """Attention-based Multi-Scale Fusion Module  (our proposed module).

    This module is the base paper's Deep Guided Filtering (DGF) **plus** an
    attention-driven correction branch:

    * core   : the identical guided-filtering formulation as DGF
               (``predict`` -> ``to_ab`` -> ``a * G + b`` -> ``refine``), using
               the *same parameter names*, so trained DGF weights can be loaded
               directly;
    * added  : channel attention (which feature channels matter) then spatial
               attention (which pixel locations matter, e.g. diffraction
               fringes), projected by a **zero-initialised** convolution.

    Because the correction branch is zero-initialised, the module is *exactly*
    DGF at initialisation.  Warm-started from a trained DGF it therefore begins
    at the baseline's accuracy, and the attention can only add to it -- the
    baseline solution is always retained.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        # ---- core: Deep Guided Filtering (base paper) ----
        self.predict = nn.Sequential(
            ConvBlock(channels * 2, channels),
            ConvBlock(channels, channels),
        )
        self.to_ab = nn.Conv2d(channels, channels * 2, 1, bias=True)
        self.refine = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)
        # ---- proposed: attention correction branch ----
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()
        self.attn_out = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.attn_out.weight)
        nn.init.zeros_(self.attn_out.bias)

    def forward(self, guide: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        feat = self.predict(torch.cat([guide, high], dim=1))
        ab = self.to_ab(feat)
        a, b = torch.chunk(ab, 2, dim=1)
        a = torch.sigmoid(a) * 2.0            # bounded, positive guidance scaling
        filtered = a * guide + b               # guided-filtering core (DGF)
        core = self.refine(filtered)
        # attention-driven correction (zero at init -> module == DGF)
        correction = self.attn_out(self.spatial_attn(self.channel_attn(feat)))
        return core + correction + high


# --------------------------------------------------------------------------- #
# Baseline fusion modules
# --------------------------------------------------------------------------- #
class NoFusion(nn.Module):
    """No cross-scale interaction: keep the high-frequency features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, guide: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        return high


class ConcatFusion(nn.Module):
    """Direct feature concatenation + 1x1 projection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.project = ConvBlock(channels * 2, channels)

    def forward(self, guide: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([guide, high], dim=1))


class DeepGuidedFilter(nn.Module):
    """Learnable Deep Guided Filtering module (base paper's design).

    Instead of predicting the filtered image directly, the module estimates
    per-channel linear coefficients ``a`` and ``b`` between the guidance and
    the input, mirroring the classical guided-filter formulation

        q = a * G + b

    but with the local statistics replaced by learned convolutions.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.predict = nn.Sequential(
            ConvBlock(channels * 2, channels),
            ConvBlock(channels, channels),
        )
        self.to_ab = nn.Conv2d(channels, channels * 2, 1, bias=True)
        # Zero-initialised residual head (see AMFM for the rationale).
        self.refine = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)

    def forward(self, guide: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        feat = self.predict(torch.cat([guide, high], dim=1))
        ab = self.to_ab(feat)
        a, b = torch.chunk(ab, 2, dim=1)
        a = torch.sigmoid(a) * 2.0  # keep coefficients positive / bounded
        filtered = a * guide + b
        return self.refine(filtered) + high


def build_fusion(mode: str, channels: int) -> nn.Module:
    mode = mode.lower()
    if mode == "none":
        return NoFusion(channels)
    if mode == "concat":
        return ConcatFusion(channels)
    if mode == "dgf":
        return DeepGuidedFilter(channels)
    if mode == "amfm":
        return AMFM(channels)
    raise ValueError(f"Unknown fusion mode: {mode!r}")


# --------------------------------------------------------------------------- #
# Low / high frequency branches
# --------------------------------------------------------------------------- #
class LowFrequencyBranch(nn.Module):
    """Encoder-decoder that restores the coarsest (low-frequency) band."""

    def __init__(self, width: int = 48) -> None:
        super().__init__()
        w = width
        self.stem = ConvBlock(3, w)
        self.drb1 = DRB(w, dilation=1)
        self.drb2 = DRB(w, dilation=2)

        self.down1 = nn.Sequential(nn.Conv2d(w, w * 2, 3, stride=2, padding=1), norm(w * 2), nn.GELU())
        self.drb3 = DRB(w * 2, dilation=1)
        self.down2 = nn.Sequential(nn.Conv2d(w * 2, w * 4, 3, stride=2, padding=1), norm(w * 4), nn.GELU())
        self.drb4 = DRB(w * 4, dilation=2)

        self.up1 = nn.Sequential(nn.Conv2d(w * 4, w * 2, 3, padding=1), nn.GELU())
        self.merge1 = ConvBlock(w * 4, w * 2)
        self.up2 = nn.Sequential(nn.Conv2d(w * 2, w, 3, padding=1), nn.GELU())
        self.merge2 = ConvBlock(w * 2, w)
        self.out = ConvBlock(w, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = self.drb2(self.drb1(self.stem(x)))
        f1 = self.drb3(self.down1(f0))
        f2 = self.drb4(self.down2(f1))

        u1 = F.interpolate(self.up1(f2), size=f1.shape[-2:], mode="bilinear", align_corners=False)
        f1 = self.merge1(torch.cat([u1, f1], dim=1))
        u0 = F.interpolate(self.up2(f1), size=f0.shape[-2:], mode="bilinear", align_corners=False)
        f0 = self.merge2(torch.cat([u0, f0], dim=1))
        return self.out(f0)


class HighFrequencyBranch(nn.Module):
    """Dilated residual stack that enhances one Laplacian band."""

    def __init__(self, width: int = 48, num_drb: int = 2) -> None:
        super().__init__()
        self.stem = ConvBlock(3, width)
        self.blocks = nn.Sequential(
            *[DRB(width, dilation=(1 if i % 2 == 0 else 2)) for i in range(num_drb)]
        )
        self.out = ConvBlock(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.blocks(self.stem(x)))
