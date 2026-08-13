from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["build", "register", "supported"]

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register(name: str, factory: Callable[..., Any]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"architecture {name!r} already registered")
    _REGISTRY[name] = factory


def build(arch: str, config: Any, backend: Any) -> Any:
    if arch not in _REGISTRY:
        raise ValueError(
            f"unknown architecture {arch!r}; supported: {', '.join(supported())}"
        )
    return _REGISTRY[arch](config, backend=backend)


def supported() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
