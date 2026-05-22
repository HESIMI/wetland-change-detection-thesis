import torch
import torch.nn as nn
import torch.nn.functional as F


class ChangeFeatureAdapter(nn.Module):
    """Project multi-scale CDMamba features into a shared mask feature space."""

    def __init__(self, in_channels: tuple[int, ...], out_channels: int = 256) -> None:
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(ch, out_channels, kernel_size=1) for ch in in_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels), out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        projected = []
        for projection, feature in zip(self.projections, features):
            x = projection(feature)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            projected.append(x)
        return self.fuse(torch.cat(projected, dim=1))
