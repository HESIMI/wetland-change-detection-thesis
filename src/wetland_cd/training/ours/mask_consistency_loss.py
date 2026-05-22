import torch
import torch.nn as nn


class MaskConsistencyLoss(nn.Module):
    """Encourage each predicted object mask to have internally consistent probabilities."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, mask_logits: torch.Tensor, class_logits: torch.Tensor) -> torch.Tensor:
        mask_probs = torch.sigmoid(mask_logits)
        class_probs = torch.sigmoid(class_logits).unsqueeze(-1).unsqueeze(-1)
        weights = (mask_probs * class_probs).detach()
        weights_sum = weights.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        mean = (mask_probs * weights).sum(dim=(2, 3), keepdim=True) / weights_sum
        variance = ((mask_probs - mean) ** 2 * weights).sum(dim=(2, 3)) / weights_sum.squeeze(-1).squeeze(-1)
        active = (class_probs.squeeze(-1).squeeze(-1) > 0.5).to(mask_logits.dtype)
        return (variance * active).sum() / active.sum().clamp_min(1.0)
