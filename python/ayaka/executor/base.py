from __future__ import annotations

from abc import ABC, abstractmethod

from ayaka.ir import ExecutionPlan, ModelOutput

__all__ = ["BaseExecutor"]


class BaseExecutor(ABC):
    """Execute scheduler plans and own runtime lifecycle hooks."""

    @abstractmethod
    def execute(self, plan: ExecutionPlan) -> ModelOutput:
        """Execute one complete scheduler step."""

    @abstractmethod
    def warmup(self) -> None:
        """Prepare runtime resources before the first execution step."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release runtime resources exactly once at engine shutdown."""