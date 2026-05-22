import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import ChangeMambaEncoder, TemporalMambaFusion
from .bitemporal_mask_decoder import BitemporalMaskInteractionDecoder
from .change_aware_query import ChangeAwareMaskQueryGenerator
from .change_feature_adapter import ChangeFeatureAdapter


class CDMambaMaskCD(nn.Module):
    """MaskCD-dominant Mamba-enhanced object-level change detector.

    Mini validation showed MaskCD gives the stronger LEVIR-CD signal, so this
    model treats mask reasoning as the primary branch. The CDMamba-style encoder
    supplies global-local change features and an auxiliary pixel head stabilizes
    early training.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: tuple[int, int, int, int] = (32, 64, 128, 256),
        depths: tuple[int, int, int, int] = (1, 1, 2, 1),
        expand_ratio: float = 2.0,
        query_dim: int = 256,
        num_queries: int = 64,
        num_heads: int = 8,
        decoder_depth: int = 2,
        query_selection: str = "topk",
        mask_logit_weight: float = 0.7,
        pixel_logit_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.mask_logit_weight = float(mask_logit_weight)
        self.pixel_logit_weight = float(pixel_logit_weight)
        self.encoder = ChangeMambaEncoder(in_channels, embed_dims, depths, expand_ratio)
        self.temporal_fusion = nn.ModuleList(
            [TemporalMambaFusion(dim, expand_ratio=expand_ratio) for dim in embed_dims]
        )
        self.t1_adapter = ChangeFeatureAdapter(embed_dims, query_dim)
        self.t2_adapter = ChangeFeatureAdapter(embed_dims, query_dim)
        self.diff_adapter = ChangeFeatureAdapter(embed_dims, query_dim)
        self.mask_feature = nn.Sequential(
            nn.Conv2d(query_dim, query_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(query_dim),
            nn.GELU(),
        )
        self.query_generator = ChangeAwareMaskQueryGenerator(
            embed_dims[-1],
            query_dim,
            num_queries,
            selection=query_selection,
        )
        self.mask_decoder = BitemporalMaskInteractionDecoder(query_dim, num_heads=num_heads, depth=decoder_depth)
        self.pixel_head = nn.Sequential(
            nn.Conv2d(query_dim, query_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(query_dim // 2),
            nn.GELU(),
            nn.Conv2d(query_dim // 2, 1, kernel_size=1),
        )

    def forward_masks(self, t1: torch.Tensor, t2: torch.Tensor) -> dict[str, torch.Tensor]:
        feats1 = self.encoder(t1)
        feats2 = self.encoder(t2)
        diffs = [fusion(f1, f2) for fusion, f1, f2 in zip(self.temporal_fusion, feats1, feats2)]

        t1_feature = self.t1_adapter(feats1)
        t2_feature = self.t2_adapter(feats2)
        diff_feature = self.diff_adapter(diffs)
        mask_feature = self.mask_feature(diff_feature)
        queries = self.query_generator(diffs[-1])
        mask_logits, class_logits = self.mask_decoder(queries, t1_feature, t2_feature, diff_feature, mask_feature)
        proposal_logits = mask_logits + class_logits.unsqueeze(-1).unsqueeze(-1)
        mask_change_logits = torch.amax(proposal_logits, dim=1, keepdim=True)
        pixel_logits = self.pixel_head(diff_feature)
        return {
            "mask_logits": mask_logits,
            "class_logits": class_logits,
            "mask_change_logits": mask_change_logits,
            "pixel_logits": pixel_logits,
            "mask_feature": mask_feature,
        }

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        output_size = t1.shape[-2:]
        outputs = self.forward_masks(t1, t2)
        logits = (
            self.mask_logit_weight * outputs["mask_change_logits"]
            + self.pixel_logit_weight * outputs["pixel_logits"]
        )
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
