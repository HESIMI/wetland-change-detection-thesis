from __future__ import annotations

import torch
import torch.nn as nn


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    numerator = 2.0 * (probs * targets).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    return 1.0 - (numerator + eps) / denominator


class BinaryChangeLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.pos_weight_value = float(pos_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = torch.tensor([self.pos_weight_value], dtype=logits.dtype, device=logits.device)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        dice = dice_loss_from_logits(logits, targets).mean()
        return bce + self.dice_weight * dice
