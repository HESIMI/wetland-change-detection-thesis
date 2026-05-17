from __future__ import annotations

import torch


class BinaryChangeMetrics:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = (torch.sigmoid(logits) > self.threshold).float()
        targets = (targets > 0.5).float()

        self.tp += float((preds * targets).sum().item())
        self.fp += float((preds * (1.0 - targets)).sum().item())
        self.fn += float(((1.0 - preds) * targets).sum().item())
        self.tn += float(((1.0 - preds) * (1.0 - targets)).sum().item())

    def compute(self) -> dict[str, float]:
        eps = 1e-6
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "accuracy": accuracy,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }
