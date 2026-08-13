from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

from .distributed import describe, is_master, rank, world_size

_CYAN = "\033[0;36m"
_RESET = "\033[0;0m"

class _OnceFilter:
    """A logging filter that only allows a message to be emitted once per process.

    Examples:
        >>> logger = logging.getLogger("mylogger")
        >>> logger.addFilter(_OnceFilter())
        >>> logger.info("This will be logged")
        >>> logger.info("This will not be logged")
    """

    __slots__ = ("_capacity", "_overflow", "_seen")

    def __init__(self, capacity: int = 4096) -> None:
        self._seen: set[Any] = set()
        self._capacity = capacity
        self._overflow = 0

    def should_emit(self, key: Any) -> bool:
        if key in self._seen:
            return False
        if len(self._seen) >= self._capacity:
            self._overflow += 1
            return True
        self._seen.add(key)
        return True

    @property
    def suppressed_overflow(self) -> int:
        return self._overflow

    def reset(self) -> None:
        self._seen.clear()
        self._overflow = 0


_once = _OnceFilter()

def log_once(
    logger: logging.Logger,
    level: int,
    message: str,
    master_only: bool = True,
    *args: Any,
    **kwargs: Any
) -> None:
    """Emit `message` at most once per process."""
    key = (logger.name, level, message, args)
    if master_only and not is_master():
        return
    if _once.should_emit(key):
        logger.log(level, message, *args, **kwargs)
        

def log_master(
    logger: logging.Logger,
    level: int,
    message: str,
    *args: Any,
    **kwargs: Any
) -> None:
    """Emit only on rank 0. For anything that would otherwise appear N times."""
    if is_master():
        logger.log(level, message, *args, **kwargs)


def reset_once_state() -> None:
    """Clear the dedup set. For tests."""
    _once.reset()

class JSONFormatter(logging.Formatter):
    """
    A logging formatter that outputs JSON. The output is a single line of JSON
    per log record, with the following fields:
    - ts: timestamp in ISO 8601 format with milliseconds
    - level: log level name
    - logger: logger name
    - rank: rank of the process (from `ayaka.distributed.rank()`)
    - message: the log message
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
            + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "rank": rank(),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Anything passed via extra={...} that is not a standard field.
        for key, value in record.__dict__.items():
            if key.startswith("ayaka_"):
                payload[key[6:]] = value
        return json.dumps(payload, ensure_ascii=False, default=str)

class TextFormatter(logging.Formatter):
    """Human-readable, with a rank prefix only when there is more than one."""

    def __init__(self) -> None:
        prefix = f"[rank{rank()}] " if world_size() > 1 else ""
        super().__init__(
            fmt=f"%(asctime)s {prefix}%(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        
