from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from threading import RLock

from .errors import (
    InvalidHandleError,
    InvalidStateTransitionError,
    InvariantViolationError,
)
from .handler import KVPageHandle, PhysicalPageId
from .states import PageAllocationState, PageMeta


@dataclass(frozen=True, slots=True)
class AllocatorSnapshot:
    """Capacity accounting for one allocator instance.

    The state counts partition ``total_pages`` exactly; ownership counts are
    independent derived signals (a page can be LIVE and shared and in-flight
    simultaneously).
    """

    total_pages: int
    usable_pages: int
    """Total minus permanent pages; what the runtime may actually allocate."""
    free_pages: int
    reserved_pages: int
    live_pages: int
    reclaim_pending_pages: int
    permanent_pages: int

    request_owned_pages: int
    cache_owned_pages: int
    shared_pages: int
    inflight_pages: int
    reclaimed_pages_total: int
    """Cumulative pages returned to FREE through the deferred reclaim queue."""


class PageAllocator:
    """Metadata allocator for indices into preallocated KV storage.

    It never performs a per-request CUDA allocation. Physical IDs are stable
    indices into storage buffers; generations belong to ownership lifetimes.
    """

    def __init__(self, *, total_pages: int, page_size: int) -> None:
        if total_pages <= 0:
            raise ValueError("total_pages must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")

        self._total_pages = total_pages
        self._page_size = page_size
        # Generation 0 marks pages as never allocated; allocate() bumps to 1.
        self._pages = [
            PageMeta(generation=0, physical_id=PhysicalPageId(index))
            for index in range(total_pages)
        ]
        # Index-ordered free pool: deterministic, O(1) acquire at either end.
        self._ready_free: deque[int] = deque(range(total_pages))
        # Min-heap of (pending_free_epoch, physical_index, generation): the
        # epoch key lets us reclaim only when safe; index+generation make the
        # tuple a deterministic total order and guard against stale entries.
        self._reclaim_heap: list[tuple[int, int, int]] = []
        self._current_epoch = 0
        self._reclaimed_pages_total = 0
        self._lock = RLock()
