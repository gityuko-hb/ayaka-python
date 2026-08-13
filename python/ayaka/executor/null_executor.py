"""Deterministic fake executor for scheduler and engine-core tests."""

from __future__ import annotations

import time
from collections.abc import Callable

from ayaka.ir import ExecutionPlan, ModelOutput, SampledToken

from .base import BaseExecutor

__all__ = ["NullExecutor", "TokenSource"]

type TokenSource = Callable[[str, int], int]


def _default_token_source(
    _request_id: str,
    logical_position: int
) -> int:
    return logical_position


class NullExecutor(BaseExecutor):
    """Execute plans without a model while preserving deterministic outputs."""

    def __init__(
        self,
        *,
        prefill_seconds_per_token: float = 1e-6,
        decode_seconds_per_step: float = 1e-5,
    ) -> None:
        if prefill_seconds_per_token < 0.0 or decode_seconds_per_step < 0.0:
            raise ValueError("latency values must be non-negative")
        self._prefill_seconds_per_token = prefill_seconds_per_token
        self._decode_seconds_per_step = decode_seconds_per_step
        self._token_source: TokenSource = _default_token_source

    def set_token_source(self, fn: TokenSource) -> None:
        """Replace the deterministic token source used for later plans."""
        self._token_source = fn

    def latency_model(self, plan: ExecutionPlan) -> float:
        """Return modeled execution latency in seconds for one plan."""
        prefill_tokens = sum(seq.num_scheduled_tokens for seq in plan.seqs if seq.is_prefill)
        has_decode = any(not seq.is_prefill for seq in plan.seqs)
        return (
            prefill_tokens * self._prefill_seconds_per_token
            + float(has_decode) * self._decode_seconds_per_step
        )

    def execute(self, plan: ExecutionPlan) -> ModelOutput:
        """Return one deterministic sampled token for every planned sequence."""
        latency_seconds = self.latency_model(plan)
        if latency_seconds > 0.0:
            time.sleep(latency_seconds)
        sampled = tuple(
            SampledToken(
                request_id=seq.request_id,
                token_id=self._token_source(
                    seq.request_id,
                    seq.num_computed_tokens + seq.num_scheduled_tokens,
                ),
            )
            for seq in plan.seqs
        )
        return ModelOutput(
            step_id=plan.step_id,
            sampled=sampled,
            forward_ms=latency_seconds * 1000.0,
        )

    def warmup(self) -> None:
        """Perform no work; the fake runtime has no resources to prepare."""

    def shutdown(self) -> None:
        """Perform no work; the fake runtime owns no external resources."""
