from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["Linear"]


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)