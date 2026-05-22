import torch
import torch.nn as nn


class ChangeAwareMaskQueryGenerator(nn.Module):
    """Generate mask queries from high-level bi-temporal difference features."""

    def __init__(self, in_channels: int, query_dim: int = 256, num_queries: int = 100) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, query_dim, kernel_size=1)
        self.pool_size = int(num_queries**0.5)
        if self.pool_size * self.pool_size < num_queries:
            self.pool_size += 1
        self.pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.num_queries = num_queries
        self.query_dim = query_dim
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.fallback_queries = nn.Parameter(torch.randn(num_queries, query_dim) * 0.02)

    def forward(self, diff_feat: torch.Tensor) -> torch.Tensor:
        x = self.proj(diff_feat)
        x = self.pool(x)
        x = x.flatten(2).transpose(1, 2)
        if x.shape[1] >= self.num_queries:
            x = x[:, : self.num_queries, :]
        else:
            pad = self.fallback_queries[: self.num_queries - x.shape[1]].unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([x, pad.to(dtype=x.dtype, device=x.device)], dim=1)
        return self.query_proj(x)
