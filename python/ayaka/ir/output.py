from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ModelOutput", "RequestOutput", "SampledToken", "UsageDelta"]

@dataclass(frozen=True, slots=True)
class SampledToken:
    """One freshly sampled token, tagged with the request it belongs to."""
    request_id: str
    token_id: int
    logprob: float | None = None

@dataclass(frozen=True, slots=True)
class ModelOutput:
    """The result of one forward step"""

    step_id: int
    sampled: tuple[SampledToken, ...] = ()
    forward_ms: float = 0.0

@dataclass(frozen=True, slots=True)
class UsageDelta:
    """The usage a single chunk *adds*, not a running total."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

@dataclass(frozen=True, slots=True)
class RequestOutput:
    """One chunk of a request's result."""

    request_id: str
    new_token_ids: tuple[int, ...] = ()
    #: `None` while the request is still running; `"stop"` / `"length"` /
    #: `"abort"` once it is done.
    finish_reason: str | None = None
    usage: UsageDelta = field(default_factory=UsageDelta)
