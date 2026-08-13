from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ExtraKeys", "RequestIR", "SamplingParams"]

#: Secondary cache key: `(lora_id, mm_hashes, tenant)`.
type ExtraKeys = tuple[str | None, tuple[str, ...], str]

#: `top_k` disabled — consider the whole vocabulary.
TOP_K_DISABLED = -1


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Per-request generation parameters, normalized off the wire format."""

    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = TOP_K_DISABLED
    stop_token_ids: tuple[int, ...] = ()
    seed: int | None = None

    def validate(self) -> None:
        """Raise `ValueError` on nonsensical parameters, return `None` if valid."""
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0, got {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0.0, 1.0], got {self.top_p}")
        if self.top_k != TOP_K_DISABLED and self.top_k < 1:
            raise ValueError(f"top_k must be >= 1 or {TOP_K_DISABLED} (disabled), got {self.top_k}")
        if any(tid < 0 for tid in self.stop_token_ids):
            raise ValueError(f"stop_token_ids must be non-negative, got {self.stop_token_ids}")


@dataclass(frozen=True, slots=True)
class RequestIR:
    """Immutable view of one request after tokenization and normalization."""

    request_id: str
    token_ids: tuple[int, ...]
    sampling: SamplingParams = field(default_factory=SamplingParams)
    arrival_ts: float = 0.0
    session_id: str | None = None
    tenant: str = "default"
    sla_class: str = "default"
    lora_id: str | None = None
    mm_hashes: tuple[str, ...] = ()

    def extra_keys(self) -> ExtraKeys:
        """Secondary cache key folded into every block hash."""
        return (self.lora_id, self.mm_hashes, self.tenant)

    def num_prompt_tokens(self) -> int:
        """Prompt length, used for admission control and the token budget."""
        return len(self.token_ids)
