"""Unit tests for the model, pyramid, losses and metrics (CPU, fast)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocks import AMFM, ChannelAttention, DeepGuidedFilter, SpatialAttention  # noqa: E402
from src.losses import LPUDCLoss  # noqa: E402
from src.metrics import psnr, ssim  # noqa: E402
from src.model import LPUDC  # noqa: E402
from src.pyramid import LaplacianPyramid  # noqa: E402


@pytest.mark.parametrize("fusion", ["none", "concat", "dgf", "amfm"])
def test_forward_shapes(fusion):
    model = LPUDC(width=16, levels=3, fusion=fusion).eval()
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_pyramid_is_invertible():
    pyr = LaplacianPyramid(3)
    x = torch.rand(2, 3, 64, 64)
    rec = pyr.reconstruct(pyr(x))
    assert torch.allclose(x, rec, atol=1e-5)


def test_attention_modules_shape():
    x = torch.rand(2, 16, 32, 32)
    assert ChannelAttention(16)(x).shape == x.shape
    assert SpatialAttention()(x).shape == x.shape
    assert AMFM(16)(x, x).shape == x.shape
    assert DeepGuidedFilter(16)(x, x).shape == x.shape


def test_loss_components():
    crit = LPUDCLoss(lambda_pyramid=0.5, levels=3)
    a, b = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    total, parts = crit(a, b)
    assert torch.isclose(total, parts["content"] + 0.5 * parts["pyramid"], atol=1e-6)


def test_metrics_perfect_prediction():
    x = torch.rand(1, 3, 64, 64)
    assert psnr(x, x).item() > 60
    assert ssim(x, x).item() > 0.999


def test_amfm_has_attention_parameters():
    m = AMFM(32)
    names = [n for n, _ in m.named_parameters()]
    assert any("channel_attn" in n for n in names)
    assert any("spatial_attn" in n for n in names)


def test_parameter_count_matches_paper_scale():
    model = LPUDC(width=48, levels=3, fusion="amfm")
    n = sum(p.numel() for p in model.parameters()) / 1e6
    assert 1.0 < n < 6.0
