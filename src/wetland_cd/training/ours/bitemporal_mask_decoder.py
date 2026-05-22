import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(query_dim)
        self.token_norm = nn.LayerNorm(query_dim)
        self.attn = nn.MultiheadAttention(query_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, int(query_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(query_dim * mlp_ratio), query_dim),
        )

    def forward(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(self.query_norm(query), self.token_norm(tokens), self.token_norm(tokens))
        query = query + attended
        return query + self.ffn(query)


class BitemporalMaskInteractionDecoder(nn.Module):
    """Decode change-aware queries into object-level mask proposals."""

    def __init__(self, query_dim: int = 256, num_heads: int = 8, depth: int = 2) -> None:
        super().__init__()
        self.t1_blocks = nn.ModuleList([CrossAttentionBlock(query_dim, num_heads=num_heads) for _ in range(depth)])
        self.t2_blocks = nn.ModuleList([CrossAttentionBlock(query_dim, num_heads=num_heads) for _ in range(depth)])
        self.diff_blocks = nn.ModuleList([CrossAttentionBlock(query_dim, num_heads=num_heads) for _ in range(depth)])
        self.fusion = nn.Sequential(
            nn.LayerNorm(query_dim * 3),
            nn.Linear(query_dim * 3, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
        )
        self.mask_embed = nn.Linear(query_dim, query_dim)
        self.class_embed = nn.Linear(query_dim, 1)

    @staticmethod
    def _tokens(feature: torch.Tensor) -> torch.Tensor:
        return feature.flatten(2).transpose(1, 2)

    def forward(
        self,
        queries: torch.Tensor,
        t1_feature: torch.Tensor,
        t2_feature: torch.Tensor,
        diff_feature: torch.Tensor,
        mask_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t1_tokens = self._tokens(t1_feature)
        t2_tokens = self._tokens(t2_feature)
        diff_tokens = self._tokens(diff_feature)

        q1 = queries
        q2 = queries
        qd = queries
        for t1_block, t2_block, diff_block in zip(self.t1_blocks, self.t2_blocks, self.diff_blocks):
            q1 = t1_block(q1, t1_tokens)
            q2 = t2_block(q2, t2_tokens)
            qd = diff_block(qd, diff_tokens)

        queries = self.fusion(torch.cat([q1, q2, qd], dim=-1))
        mask_embed = self.mask_embed(queries)
        mask_logits = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feature)
        class_logits = self.class_embed(queries).squeeze(-1)
        return mask_logits, class_logits
