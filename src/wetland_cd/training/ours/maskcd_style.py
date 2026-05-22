import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import ChangeFormerEncoder
from .bitemporal_mask_decoder import BitemporalMaskInteractionDecoder
from .change_feature_adapter import ChangeFeatureAdapter


class MaskCDStyle(nn.Module):
    """Compact MaskCD-style baseline for unified local comparisons.

    This is a CUDA-friendly, self-contained approximation of the MaskCD
    paradigm: shared bi-temporal feature extraction, object-level mask queries,
    mask classification, and mask proposal aggregation. It is intentionally
    separate from the official AI4RS/MaskCD reproduction.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: tuple[int, int, int, int] = (32, 64, 128, 256),
        depths: tuple[int, int, int, int] = (1, 1, 2, 1),
        num_heads: tuple[int, int, int, int] = (1, 2, 4, 8),
        query_dim: int = 256,
        num_queries: int = 64,
        decoder_heads: int = 8,
        decoder_depth: int = 2,
        mask_logit_weight: float = 0.85,
        pixel_logit_weight: float = 0.15,
    ) -> None:
        super().__init__()
        self.mask_logit_weight = float(mask_logit_weight)
        self.pixel_logit_weight = float(pixel_logit_weight)
        self.encoder = ChangeFormerEncoder(in_channels, embed_dims, depths, num_heads)
        self.t1_adapter = ChangeFeatureAdapter(embed_dims, query_dim)
        self.t2_adapter = ChangeFeatureAdapter(embed_dims, query_dim)
        self.diff_adapter = ChangeFeatureAdapter(embed_dims, query_dim)
        self.mask_feature = nn.Sequential(
            nn.Conv2d(query_dim, query_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(query_dim),
            nn.GELU(),
        )
        self.queries = nn.Parameter(torch.randn(num_queries, query_dim) * 0.02)
        self.query_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(query_dim, query_dim, kernel_size=1),
            nn.GELU(),
        )
        self.mask_decoder = BitemporalMaskInteractionDecoder(query_dim, num_heads=decoder_heads, depth=decoder_depth)
        self.pixel_head = nn.Sequential(
            nn.Conv2d(query_dim, query_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(query_dim // 2),
            nn.GELU(),
            nn.Conv2d(query_dim // 2, 1, kernel_size=1),
        )

    def forward_masks(self, t1: torch.Tensor, t2: torch.Tensor) -> dict[str, torch.Tensor]:
        feats1 = self.encoder(t1)
        feats2 = self.encoder(t2)
        diffs = [torch.abs(f1 - f2) for f1, f2 in zip(feats1, feats2)]

        t1_feature = self.t1_adapter(feats1)
        t2_feature = self.t2_adapter(feats2)
        diff_feature = self.diff_adapter(diffs)
        mask_feature = self.mask_feature(diff_feature)

        batch_size = int(t1.shape[0])
        context = self.query_context(diff_feature).flatten(2).transpose(1, 2)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1) + context
        mask_logits, class_logits = self.mask_decoder(queries, t1_feature, t2_feature, diff_feature, mask_feature)
        proposal_logits = mask_logits + class_logits.unsqueeze(-1).unsqueeze(-1)
        mask_change_logits = torch.amax(proposal_logits, dim=1, keepdim=True)
        pixel_logits = self.pixel_head(diff_feature)
        return {
            "mask_logits": mask_logits,
            "class_logits": class_logits,
            "mask_change_logits": mask_change_logits,
            "pixel_logits": pixel_logits,
        }

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        output_size = t1.shape[-2:]
        outputs = self.forward_masks(t1, t2)
        logits = (
            self.mask_logit_weight * outputs["mask_change_logits"]
            + self.pixel_logit_weight * outputs["pixel_logits"]
        )
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
