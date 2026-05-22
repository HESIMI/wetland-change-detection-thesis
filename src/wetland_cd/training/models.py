import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        return feat, self.pool(feat)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SiameseUNet(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        x = self.bottleneck(x)
        return [s1, s2, s3], x

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        skips1, bottleneck1 = self.encode(t1)
        skips2, bottleneck2 = self.encode(t2)

        x = torch.abs(bottleneck1 - bottleneck2)
        skip3 = torch.abs(skips1[2] - skips2[2])
        skip2 = torch.abs(skips1[1] - skips2[1])
        skip1 = torch.abs(skips1[0] - skips2[0])

        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.head(x)


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.proj(x)))


class TransformerStage(nn.Module):
    def __init__(self, channels: int, num_heads: int, depth: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=int(channels * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.blocks(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class ChangeFormerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dims: tuple[int, int, int, int],
        depths: tuple[int, int, int, int],
        num_heads: tuple[int, int, int, int],
    ) -> None:
        super().__init__()
        self.patch1 = PatchEmbed(in_channels, embed_dims[0], stride=4)
        self.stage1 = TransformerStage(embed_dims[0], num_heads[0], depths[0])
        self.patch2 = PatchEmbed(embed_dims[0], embed_dims[1], stride=2)
        self.stage2 = TransformerStage(embed_dims[1], num_heads[1], depths[1])
        self.patch3 = PatchEmbed(embed_dims[1], embed_dims[2], stride=2)
        self.stage3 = TransformerStage(embed_dims[2], num_heads[2], depths[2])
        self.patch4 = PatchEmbed(embed_dims[2], embed_dims[3], stride=2)
        self.stage4 = TransformerStage(embed_dims[3], num_heads[3], depths[3])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.stage1(self.patch1(x))
        f2 = self.stage2(self.patch2(f1))
        f3 = self.stage3(self.patch3(f2))
        f4 = self.stage4(self.patch4(f3))
        return [f1, f2, f3, f4]


class ChangeFormerDecoder(nn.Module):
    def __init__(self, embed_dims: tuple[int, int, int, int], decoder_dim: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(dim, decoder_dim, kernel_size=1) for dim in embed_dims])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
        )
        self.head = nn.Conv2d(decoder_dim, 1, kernel_size=1)

    def forward(self, diffs: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        target_size = diffs[0].shape[-2:]
        projected = [
            F.interpolate(proj(feat), size=target_size, mode="bilinear", align_corners=False)
            for proj, feat in zip(self.projections, diffs)
        ]
        fused = self.fuse(torch.cat(projected, dim=1))
        logits = self.head(fused)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class ChangeFormer(nn.Module):
    """Compact ChangeFormer-style baseline for unified comparisons.

    The model keeps the core ChangeFormer idea: a shared Siamese Transformer
    encoder extracts multi-scale bi-temporal features, and a lightweight MLP-like
    decoder fuses absolute feature differences into a binary change map.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: tuple[int, int, int, int] = (32, 64, 128, 256),
        depths: tuple[int, int, int, int] = (1, 1, 2, 1),
        num_heads: tuple[int, int, int, int] = (1, 2, 4, 8),
        decoder_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = ChangeFormerEncoder(in_channels, embed_dims, depths, num_heads)
        self.decoder = ChangeFormerDecoder(embed_dims, decoder_dim)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        output_size = t1.shape[-2:]
        feats1 = self.encoder(t1)
        feats2 = self.encoder(t2)
        diffs = [torch.abs(f1 - f2) for f1, f2 in zip(feats1, feats2)]
        return self.decoder(diffs, output_size)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class DirectionalScan2d(nn.Module):
    """CUDA-free 2D scan used for the compact ChangeMamba-style baseline."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        width_steps = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype).view(1, 1, 1, -1)
        height_steps = torch.arange(1, x.shape[-2] + 1, device=x.device, dtype=x.dtype).view(1, 1, -1, 1)

        left_to_right = torch.cumsum(x, dim=-1) / width_steps
        right_to_left = torch.flip(torch.cumsum(torch.flip(x, dims=[-1]), dim=-1), dims=[-1]) / torch.flip(
            width_steps, dims=[-1]
        )
        top_to_bottom = torch.cumsum(x, dim=-2) / height_steps
        bottom_to_top = torch.flip(torch.cumsum(torch.flip(x, dims=[-2]), dim=-2), dims=[-2]) / torch.flip(
            height_steps, dims=[-2]
        )
        return 0.25 * (left_to_right + right_to_left + top_to_bottom + bottom_to_top)


class ChangeMambaBlock(nn.Module):
    def __init__(self, channels: int, expand_ratio: float = 2.0) -> None:
        super().__init__()
        hidden_channels = int(channels * expand_ratio)
        self.norm = LayerNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
        )
        self.scan = DirectionalScan2d()
        self.out_proj = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        content, gate = self.in_proj(self.norm(x)).chunk(2, dim=1)
        content = self.dwconv(content)
        content = self.scan(F.silu(content))
        x = self.out_proj(content * F.silu(gate))
        return residual + x * self.scale


class ChangeMambaStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, depth: int, expand_ratio: float) -> None:
        super().__init__()
        self.patch = PatchEmbed(in_channels, out_channels, stride=stride)
        self.blocks = nn.Sequential(
            *[ChangeMambaBlock(out_channels, expand_ratio=expand_ratio) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.patch(x))


class ChangeMambaEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dims: tuple[int, int, int, int],
        depths: tuple[int, int, int, int],
        expand_ratio: float,
    ) -> None:
        super().__init__()
        self.stage1 = ChangeMambaStage(in_channels, embed_dims[0], stride=4, depth=depths[0], expand_ratio=expand_ratio)
        self.stage2 = ChangeMambaStage(embed_dims[0], embed_dims[1], stride=2, depth=depths[1], expand_ratio=expand_ratio)
        self.stage3 = ChangeMambaStage(embed_dims[1], embed_dims[2], stride=2, depth=depths[2], expand_ratio=expand_ratio)
        self.stage4 = ChangeMambaStage(embed_dims[2], embed_dims[3], stride=2, depth=depths[3], expand_ratio=expand_ratio)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class TemporalMambaFusion(nn.Module):
    def __init__(self, channels: int, expand_ratio: float) -> None:
        super().__init__()
        self.block = ChangeMambaBlock(channels * 3, expand_ratio=expand_ratio)
        self.project = nn.Conv2d(channels * 3, channels, kernel_size=1)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(f1 - f2)
        fused = self.block(torch.cat([diff, f1 * f2, f2 - f1], dim=1))
        return self.project(fused)


class ChangeMambaLite(nn.Module):
    """Compact ChangeMamba-style exploratory model without official kernels.

    It follows the same comparison role as the local ChangeFormer baseline: a
    shared Siamese encoder, multi-scale temporal interaction, and a lightweight
    decoder. This is not the official ChangeMamba implementation; official
    reproduction should use ChenHongruixuan/ChangeMamba and MambaSCD/MambaBCD.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: tuple[int, int, int, int] = (32, 64, 128, 256),
        depths: tuple[int, int, int, int] = (1, 1, 2, 1),
        expand_ratio: float = 2.0,
        decoder_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = ChangeMambaEncoder(in_channels, embed_dims, depths, expand_ratio)
        self.temporal_fusion = nn.ModuleList(
            [TemporalMambaFusion(dim, expand_ratio=expand_ratio) for dim in embed_dims]
        )
        self.decoder = ChangeFormerDecoder(embed_dims, decoder_dim)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        output_size = t1.shape[-2:]
        feats1 = self.encoder(t1)
        feats2 = self.encoder(t2)
        diffs = [fusion(f1, f2) for fusion, f1, f2 in zip(self.temporal_fusion, feats1, feats2)]
        return self.decoder(diffs, output_size)


from .ours import CDMambaMaskCD


MODEL_REGISTRY = {
    "cdmamba_maskcd": CDMambaMaskCD,
    "changeformer": ChangeFormer,
    "changemamba_lite": ChangeMambaLite,
    "change_mamba_lite": ChangeMambaLite,
    "siamese_unet": SiameseUNet,
}


def build_model(name: str, in_channels: int, **kwargs) -> nn.Module:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model '{name}'. Available models: {available}")
    return MODEL_REGISTRY[key](in_channels=in_channels, **kwargs)
