import torch
import torch.nn as nn


class ChangeAwareMaskQueryGenerator(nn.Module):
    """Generate mask queries from high-level bi-temporal difference features.

    The default top-k mode follows the MaskCD mini-result observation: mask
    reasoning is useful, but the queries should be anchored on the strongest
    candidate change regions instead of uniformly pooled background tokens.
    """

    def __init__(
        self,
        in_channels: int,
        query_dim: int = 256,
        num_queries: int = 100,
        selection: str = "topk",
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, query_dim, kernel_size=1)
        self.score = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.pool_size = int(num_queries**0.5)
        if self.pool_size * self.pool_size < num_queries:
            self.pool_size += 1
        self.pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.num_queries = num_queries
        self.query_dim = query_dim
        self.selection = selection
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.fallback_queries = nn.Parameter(torch.randn(num_queries, query_dim) * 0.02)

    def forward(self, diff_feat: torch.Tensor) -> torch.Tensor:
        if self.selection == "topk":
            projected = self.proj(diff_feat).flatten(2).transpose(1, 2)
            scores = self.score(diff_feat).flatten(2).squeeze(1)
            query_count = min(self.num_queries, projected.shape[1])
            indices = torch.topk(scores, k=query_count, dim=1).indices
            gather_indices = indices.unsqueeze(-1).expand(-1, -1, self.query_dim)
            x = projected.gather(1, gather_indices)
        else:
            x = self.proj(diff_feat)
            x = self.pool(x)
            x = x.flatten(2).transpose(1, 2)

        if x.shape[1] >= self.num_queries:
            x = x[:, : self.num_queries, :]
        else:
            pad = self.fallback_queries[: self.num_queries - x.shape[1]].unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([x, pad.to(dtype=x.dtype, device=x.device)], dim=1)
        return self.query_proj(x)
