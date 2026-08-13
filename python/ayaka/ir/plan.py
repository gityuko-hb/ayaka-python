from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum

from .request import SamplingParams

__all__ = ["ExecutionPlan", "KVOp", "KVOpArg", "KVOpKind", "SamplingMeta", "SeqSlice"]

#: Operand of a `KVOp`: an id (block, extent, VA offset) or a string handle.
type KVOpArg = int | str


class KVOpKind(IntEnum):
    """Vocabulary of KV memory operations.

    The integer values are part of the wire format: only ever append.
    """

    ALLOC = 1
    FREE = 2
    COPY_BLOCK = 3
    MAP = 4
    UNMAP = 5
    SET_ACCESS = 6


@dataclass(frozen=True, slots=True)
class KVOp:
    """One memory operation"""
    kind: KVOpKind
    request_id: str
    args: tuple[KVOpArg, ...] = ()


@dataclass(frozen=True, slots=True)
class SeqSlice:
    """The work owned by exactly one sequence in exactly one step."""

    request_id: str
    num_computed_tokens: int
    num_scheduled_tokens: int
    block_ids: tuple[int, ...] = ()
    is_prefill: bool = False


@dataclass(frozen=True, slots=True)
class SamplingMeta:
    """Batch-wide sampling parameters, gathered column by column.

    Holds plain Python tuples, never tensors. Column order matches
    `ExecutionPlan.seqs` position for position — that is the load-bearing
    invariant of this class.
    """
    temperatures: tuple[float, ...] = ()
    top_ps: tuple[float, ...] = ()
    top_ks: tuple[int, ...] = ()
    seeds: tuple[int | None, ...] = ()

    def __post_init__(self) -> None:
        lengths = {
            "temperatures": len(self.temperatures),
            "top_ps": len(self.top_ps),
            "top_ks": len(self.top_ks),
            "seeds": len(self.seeds),
        }
        if len(set(lengths.values())) > 1:
            raise ValueError(f"SamplingMeta columns must share one length, got {lengths}")

    def __len__(self) -> int:
        """Number of sequences in the batch."""
        return len(self.temperatures)

    @classmethod
    def from_slices(
        cls,
        seqs: Sequence[SeqSlice],
        params_by_request: Mapping[str, SamplingParams],
    ) -> SamplingMeta:
        """Build the columns in `seqs` order. The only factory.

        Raises `KeyError` naming any scheduled request that has no parameters.
        """
        rows: list[SamplingParams] = []
        for seq in seqs:
            try:
                rows.append(params_by_request[seq.request_id])
            except KeyError:
                raise KeyError(
                    f"no SamplingParams for scheduled request {seq.request_id!r}"
                ) from None
        return cls(
            temperatures=tuple(p.temperature for p in rows),
            top_ps=tuple(p.top_p for p in rows),
            top_ks=tuple(p.top_k for p in rows),
            seeds=tuple(p.seed for p in rows),
        )


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """One step in full: what to do with memory, what to forward, how to sample."""

    step_id: int
    kv_ops: tuple[KVOp, ...] = ()
    seqs: tuple[SeqSlice, ...] = ()
    total_scheduled_tokens: int = 0
    sampling_meta: SamplingMeta = field(default_factory=SamplingMeta)
    #: Bumped whenever the VA to PA mapping table changes. Always 0.
    mapping_generation: int = 0

    def is_empty(self) -> bool:
        """Nothing to do — `EngineCore` should block on input instead of spinning."""
        return not self.seqs and not self.kv_ops

    def shape_bucket(self) -> tuple[int, int, int]:
        """Lookup key for the CUDA Graph pool."""
        return (
            _next_pow2(len(self.seqs)),
            _next_pow2(self.total_scheduled_tokens),
            self.mapping_generation,
        )


def _next_pow2(n: int) -> int:
    """Round up to the nearest power of two; `0` stays `0`."""
    if n <= 0:
        return 0
    return 1 << (n - 1).bit_length()
