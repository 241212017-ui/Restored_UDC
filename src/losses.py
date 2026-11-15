"""Loss functions from the base paper: L1 content loss + Laplacian pyramid loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pyramid import laplacian_loss


class LPUDCLoss(nn.Module):
    """``L = L_content + lambda * L_pyramid`` (lambda = 0.5 in the paper)."""

    def __init__(self, lambda_pyramid: float = 0.5, levels: int = 3) -> None:
        super().__init__()
        self.lambda_pyramid = lambda_pyramid
        self.levels = levels

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        content = F.l1_loss(pred, target)
        pyramid = laplacian_loss(pred, target, self.levels)
        total = content + self.lambda_pyramid * pyramid
        return total, {"content": content.detach(), "pyramid": pyramid.detach()}
