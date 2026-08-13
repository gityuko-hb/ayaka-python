from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["LMHead", "VocabEmbedding"]


class VocabEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)


class LMHead(nn.Module):
    def __init__(self, embedding: VocabEmbedding) -> None:
        super().__init__()
        self._embedding = embedding

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self._embedding.weight, None)
