from __future__ import annotations

import torch
from torch.nn import functional as F

__all__ = ["swiglu"]

def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up
