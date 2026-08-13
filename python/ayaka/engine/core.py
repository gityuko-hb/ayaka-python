from __future__ import annotations

from collections.abc import Callable

from ayaka.executor.base import BaseExecutor
from ayaka.ir import RequestIR, RequestOutput
from ayaka.sched.scheduler import Scheduler

__all__ = ["EngineCore", "OutputSink"]

type OutputSink = Callable[[tuple[RequestOutput, ...]], None]


def _discard_output(_outputs: tuple[RequestOutput, ...]) -> None:
    return None

class EngineCore:
    """Drain scheduler work through an executor without idle spinning."""

    def __init__(
        self,
        scheduler: Scheduler,
        executor: BaseExecutor,
        output_sink: OutputSink | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._executor = executor
        self._output_sink = output_sink if output_sink is not None else _discard_output
        self._warmed_up = False
        self._closed = False

    def add_request(self, ir: RequestIR) -> None:
        """Submit one request to the scheduler."""
        self._ensure_open()
        self._scheduler.add_request(ir)

    def run_busy_loop(self) -> None:
        """Drain all currently schedulable work in strict pipeline order."""
        self._ensure_open()
        if not self._warmed_up:
            self._executor.warmup()
            self._warmed_up = True

        while not self._closed:
            plan = self._scheduler.schedule()
            if plan.is_empty():
                return
            model_output = self._executor.execute(plan)
            request_outputs = tuple(self._scheduler.update_from_output(model_output))
            self._output_sink(request_outputs)

    def shutdown(self) -> None:
        """Close the core and shut down the executor exactly once."""
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("engine core is shut down")
