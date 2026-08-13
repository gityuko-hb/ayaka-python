"""Disposable one-request-at-a-time FCFS scheduler."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ayaka.ir import (
    ExecutionPlan,
    KVOp,
    KVOpKind,
    ModelOutput,
    RequestIR,
    RequestOutput,
    SamplingMeta,
    SeqSlice,
    UsageDelta,
)

__all__ = ["Scheduler"]

@dataclass(slots=True)
class _RequestState:
    ir: RequestIR
    block_id: int
    num_computed_tokens: int = 0
    sampled_token_ids: list[int] = field(default_factory=list)
    prefill_complete: bool = False


class Scheduler:
    """Run queued requests to completion in strict arrival order."""

    def __init__(self) -> None:
        self._waiting: deque[RequestIR] = deque()
        self._known_request_ids: set[str] = set()
        self._active: _RequestState | None = None
        self._pending_plan: ExecutionPlan | None = None
        self._next_step_id = 0
        self._next_block_id = 0

    def add_request(self, ir: RequestIR) -> None:
        """Append a unique request to the FCFS queue."""
        if ir.request_id in self._known_request_ids:
            raise ValueError(f"duplicate request_id {ir.request_id!r}")
        self._known_request_ids.add(ir.request_id)
        self._waiting.append(ir)

    def schedule(self) -> ExecutionPlan:
        """Create the next prefill or single-token decode plan."""
        if self._pending_plan is not None:
            raise RuntimeError("cannot schedule while a plan is outstanding")
        if self._active is None and self._waiting:
            ir = self._waiting.popleft()
            self._active = _RequestState(ir=ir, block_id=self._next_block_id)
            self._next_block_id += 1
        if self._active is None:
            return ExecutionPlan(step_id=self._next_step_id)

        state = self._active
        is_prefill = not state.prefill_complete
        scheduled_tokens = len(state.ir.token_ids) if is_prefill else 1
        seq = SeqSlice(
            request_id=state.ir.request_id,
            num_computed_tokens=state.num_computed_tokens,
            num_scheduled_tokens=scheduled_tokens,
            block_ids=(state.block_id,),
            is_prefill=is_prefill,
        )
        kv_ops = (
            (KVOp(KVOpKind.ALLOC, state.ir.request_id, (state.block_id,)),) if is_prefill else ()
        )
        plan = ExecutionPlan(
            step_id=self._next_step_id,
            kv_ops=kv_ops,
            seqs=(seq,),
            total_scheduled_tokens=scheduled_tokens,
            sampling_meta=SamplingMeta.from_slices(
                (seq,), {state.ir.request_id: state.ir.sampling}
            ),
        )
        self._next_step_id += 1
        self._pending_plan = plan
        return plan

    def update_from_output(self, output: ModelOutput) -> list[RequestOutput]:
        """Apply one model result and emit one streaming request chunk."""
        plan = self._pending_plan
        if plan is None or self._active is None:
            raise RuntimeError("cannot update without an outstanding plan")
        if output.step_id != plan.step_id:
            raise ValueError(
                f"output step {output.step_id} does not match plan step {plan.step_id}"
            )
        if len(output.sampled) != 1:
            raise ValueError("output must contain exactly one sampled token")

        state = self._active
        sampled = output.sampled[0]
        if sampled.request_id != state.ir.request_id:
            raise ValueError(
                f"expected sampled token for {state.ir.request_id!r}, "
                f"got {sampled.request_id!r}"
            )

        seq = plan.seqs[0]
        state.num_computed_tokens += seq.num_scheduled_tokens
        state.prefill_complete = True
        state.sampled_token_ids.append(sampled.token_id)
        self._pending_plan = None

        stopped = sampled.token_id in state.ir.sampling.stop_token_ids
        reached_length = len(state.sampled_token_ids) >= state.ir.sampling.max_tokens
        finish_reason = "stop" if stopped else "length" if reached_length else None
        result = RequestOutput(
            request_id=state.ir.request_id,
            new_token_ids=(sampled.token_id,),
            finish_reason=finish_reason,
            usage=UsageDelta(
                prompt_tokens=seq.num_scheduled_tokens if seq.is_prefill else 0,
                completion_tokens=1,
            ),
        )
        if finish_reason is not None:
            self._active = None
        return [result]
