from __future__ import annotations

import torch
from torch import nn

__all__ = ["RotaryEmbedding", "apply_rotary_pos_emb"]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return (q * cos + _rotate_half(q) * sin), (k * cos + _rotate_half(k) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        *,
        base: float = 1_000_000.0,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.inv_freq: torch.Tensor
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._max_seq_len = max_seq_len
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    def _ensure_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cos is None or self._sin is None:
            positions = torch.arange(self._max_seq_len, dtype=torch.float32)
            freqs = positions[:, None] @ self.inv_freq[None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos = emb.cos()
            self._sin = emb.sin()
        return self._cos, self._sin

    def get_cos_sin(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._ensure_tables()
        pos = positions.to(torch.long)
        if cos.device != positions.device:
            cos = cos.to(positions.device)
            sin = sin.to(positions.device)
            self._cos, self._sin = cos, sin
        return cos[pos].to(dtype), sin[pos].to(dtype)
