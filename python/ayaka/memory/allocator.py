"""Fixed-size page allocator with generation checks and deferred reclamation.

Allocates generation-safe metadata identities into preallocated KV storage.
Deferred free defers recycling until a safe completion epoch, so a page can
never be handed out while a launched GPU step may still write to it.
"""

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

    @property
    def total_pages(self) -> int:
        return self._total_pages

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def current_epoch(self) -> int:
        """Epoch of the newest completed GPU step; monotonic."""
        return self._current_epoch

    def reserve_permanent_page(self) -> KVPageHandle:
        """Carve one page out of the pool as the permanent padding page.

        The padding page is terminal: it stays PERMANENT for the allocator's
        lifetime and is excluded from usable capacity.

        Returns:
            The handle of the padding page.

        Raises:
            InvalidStateTransitionError: if the free pool is empty.
        """
        with self._lock:
            if not self._ready_free:
                raise InvalidStateTransitionError(
                    "no page is available for the permanent padding page"
                )
            index = self._ready_free.popleft()
            meta = self._pages[index]
            self._reset_for_new_generation(meta)
            meta.allocation_state = PageAllocationState.PERMANENT
            meta.valid_tokens = self._page_size
            return KVPageHandle(index=index, generation=meta.generation)

    def allocate(self, num_pages: int) -> tuple[KVPageHandle, ...] | None:
        """Reserve ``num_pages`` pages as RESERVED, or return None if short.

        Pages are only marked RESERVED here; they become LIVE through
        :meth:`commit_reserved` once execution succeeds. The request is
        all-or-nothing: nothing is handed out when capacity is short.

        Args:
            num_pages: Number of pages to reserve; must be non-negative.

        Returns:
            New generation-safe handles, or None when the free pool is short.
        """
        if num_pages < 0:
            raise ValueError("num_pages must be non-negative")
        if num_pages == 0:
            return ()
        with self._lock:
            self._reclaim_completed_locked(self._current_epoch)
            if len(self._ready_free) < num_pages:
                return None

            indices = [self._ready_free.popleft() for _ in range(num_pages)]
            handles: list[KVPageHandle] = []
            for index in indices:
                meta = self._pages[index]
                self._reset_for_new_generation(meta)
                meta.allocation_state = PageAllocationState.RESERVED
                # A reserved page has exactly one owner: the reserving transaction.
                meta.reservation_refs = 1
                handles.append(KVPageHandle(index=index, generation=meta.generation))
            return tuple(handles)

    def commit_reserved(self, pages: Sequence[KVPageHandle]) -> None:
        """Convert RESERVED pages into request-owned LIVE pages (RESERVED->LIVE).

        Called by ``complete_step`` after a successful step; converts the
        reservation owner into one request reference.

        Raises:
            InvalidStateTransitionError: if a page is not RESERVED.
            InvariantViolationError: if a page does not have exactly one
                reservation owner.
        """
        with self._lock:
            metas = [self._require_state(page, PageAllocationState.RESERVED) for page in pages]
            for meta in metas:
                if meta.reservation_refs != 1:
                    raise InvariantViolationError(
                        "a reserved page must have exactly one reservation owner"
                    )
            for meta in metas:
                meta.reservation_refs = 0
                meta.request_refs += 1
                meta.allocation_state = PageAllocationState.LIVE

    def rollback_reserved(self, pages: Sequence[KVPageHandle]) -> None:
        """Undo a reservation before launch (RESERVED->FREE immediately).

        Safe only when no GPU work touched the pages: in-flight reserved pages
        are rejected here and instead go through :meth:`abandon_reserved`.
        """
        with self._lock:
            metas = [self._require_state(page, PageAllocationState.RESERVED) for page in pages]
            for meta in metas:
                if meta.inflight_refs:
                    raise InvalidStateTransitionError(
                        "an in-flight reserved page cannot be rolled back immediately"
                    )
                if meta.reservation_refs != 1:
                    raise InvariantViolationError("invalid reservation refcount")
            for meta in metas:
                meta.reservation_refs = 0
                self._make_free(meta)

    def abandon_reserved(
        self,
        pages: Sequence[KVPageHandle],
        *,
        safe_epoch: int,
    ) -> None:
        """Drop reservation ownership after an in-flight execution failed.

        Unlike :meth:`rollback_reserved`, the bytes may have been partially
        written, so the page is not reused before ``safe_epoch``.

        Args:
            pages: RESERVED pages to abandon.
            safe_epoch: Epoch whose completion makes the pages reusable.

        Raises:
            ValueError: if ``safe_epoch`` precedes the completed epoch.
        """

        self._validate_epoch(safe_epoch)
        with self._lock:
            metas = [self._require_state(page, PageAllocationState.RESERVED) for page in pages]
            for meta in metas:
                if meta.reservation_refs != 1:
                    raise InvariantViolationError("invalid reservation refcount")
            for meta in metas:
                meta.reservation_refs = 0
                # Deferred rather than immediate: the failed step's writes may
                # still be racing with future readers.
                self._mark_for_reclaim_if_unowned(meta, safe_epoch=safe_epoch)

    def mark_inflight(self, pages: Iterable[KVPageHandle]) -> None:
        """Acquire transient step ownership over pages the step may touch.

        Increments ``inflight_refs`` on each unique page; duplicate handles in
        the input are deduplicated so one step counts once per page.

        Raises:
            InvalidStateTransitionError: if a page is not LIVE or RESERVED.
        """
        unique = self._unique_pages(pages)
        with self._lock:
            metas = [self._require_live_or_reserved(page) for page in unique]
            for meta in metas:
                meta.inflight_refs += 1

    def unmark_inflight(self, pages: Iterable[KVPageHandle]) -> None:
        """Release step ownership after the step finished (success or failure).

        When this drops the last ref on a page that was already marked for
        deferred free, the page moves to RECLAIM_PENDING now that no step can
        touch it.

        Raises:
            InvariantViolationError: on inflight_refs underflow.
        """
        unique = self._unique_pages(pages)
        with self._lock:
            metas = [self._require_live_or_reserved(page) for page in unique]
            for meta in metas:
                if meta.inflight_refs <= 0:
                    raise InvariantViolationError("in-flight refcount underflow")
            for meta in metas:
                meta.inflight_refs -= 1
                # Deferred free was recorded while the step was still running;
                # only now is it safe to enter the reclaim queue.
                if meta.ownership_refs == 0 and meta.pending_free_epoch is not None:
                    self._mark_for_reclaim_if_unowned(
                        meta,
                        safe_epoch=meta.pending_free_epoch,
                    )

    def set_valid_tokens(self, page: KVPageHandle, valid_tokens: int) -> None:
        """Publish the number of valid KV tokens on a live/reserved page.

        Args:
            page: A LIVE or RESERVED page handle.
            valid_tokens: Count in ``[0, page_size]``.

        Raises:
            ValueError: if ``valid_tokens`` is outside the page boundary.
        """
        if not 0 <= valid_tokens <= self._page_size:
            raise ValueError("valid_tokens is outside the page boundary")
        with self._lock:
            meta = self._require_live_or_reserved(page)
            meta.valid_tokens = valid_tokens

    def release_request_ref(self, page: KVPageHandle, *, safe_epoch: int) -> None:
        """Drop one request reference (sequence finish, preemption, release).

        A LIVE page with no remaining ownership refs and no in-flight refs
        enters the deferred-free queue for ``safe_epoch``.

        Args:
            page: A LIVE page handle.
            safe_epoch: Epoch whose completion makes the page reusable.

        Raises:
            InvariantViolationError: on request refcount underflow.
        """
        self._validate_epoch(safe_epoch)
        with self._lock:
            meta = self._require_state(page, PageAllocationState.LIVE)
            if meta.request_refs <= 0:
                raise InvariantViolationError("request refcount underflow")
            meta.request_refs -= 1
            self._mark_for_reclaim_if_unowned(meta, safe_epoch=safe_epoch)

    def acquire_request_ref(self, page: KVPageHandle) -> None:
        """Add one request reference (e.g. prefix attachment to a sequence)."""
        with self._lock:
            meta = self._require_state(page, PageAllocationState.LIVE)
            meta.request_refs += 1
            self._clear_stale_pending_free(meta)

    def acquire_cache_ref(self, page: KVPageHandle) -> None:
        """Add one prefix-cache reference when a block enters the cache."""
        with self._lock:
            meta = self._require_state(page, PageAllocationState.LIVE)
            meta.cache_refs += 1
            self._clear_stale_pending_free(meta)

    def release_cache_ref(self, page: KVPageHandle, *, safe_epoch: int) -> None:
        """Drop one prefix-cache reference (eviction or pruning).

        Args:
            page: A LIVE page handle.
            safe_epoch: Epoch whose completion makes the page reusable.

        Raises:
            InvariantViolationError: on cache refcount underflow.
        """
        self._validate_epoch(safe_epoch)
        with self._lock:
            meta = self._require_state(page, PageAllocationState.LIVE)
            if meta.cache_refs <= 0:
                raise InvariantViolationError("cache refcount underflow")
            meta.cache_refs -= 1
            self._mark_for_reclaim_if_unowned(meta, safe_epoch=safe_epoch)

    def advance_epoch(self, completed_epoch: int) -> int:
        """Advance the completion watermark and reclaim now-safe pages.

        Args:
            completed_epoch: Newest completed GPU step epoch; must not regress.

        Returns:
            Number of pages reclaimed in this call.
        """
        self._validate_epoch(completed_epoch)
        with self._lock:
            if completed_epoch < self._current_epoch:
                raise ValueError("completed epoch must be monotonic")
            self._current_epoch = completed_epoch
            return self._reclaim_completed_locked(completed_epoch)

    def reclaim_completed(self) -> int:
        """Reclaim every deferred page whose epoch is already complete.

        Returns:
            Number of pages returned to the free pool.
        """
        with self._lock:
            return self._reclaim_completed_locked(self._current_epoch)

    def available_pages(self) -> int:
        """Number of pages immediately allocatable (free pool length)."""
        with self._lock:
            return len(self._ready_free)

    def physical_id(self, handle: KVPageHandle) -> PhysicalPageId:
        """Resolve a handle to its stable physical storage index.

        Raises:
            InvalidHandleError: if the handle is stale or the page is FREE.
        """
        with self._lock:
            return self._get_meta(handle).physical_id

    def get_meta(self, handle: KVPageHandle) -> PageMeta:
        """Return an immutable copy of a page's metadata.

        Raises:
            InvalidHandleError: if the handle is stale or the page is FREE.
        """
        with self._lock:
            return self._get_meta(handle).snapshot()

    def snapshot(self) -> AllocatorSnapshot:
        """Return full capacity and ownership accounting.

        ``states.count`` partitions the page list by allocation state; each
        page contributes to exactly one state bucket, so the buckets sum to
        ``total_pages`` (invariant 1).
        """
        with self._lock:
            states = [meta.allocation_state for meta in self._pages]
            permanent = states.count(PageAllocationState.PERMANENT)
            return AllocatorSnapshot(
                total_pages=self._total_pages,
                usable_pages=self._total_pages - permanent,
                free_pages=states.count(PageAllocationState.FREE),
                reserved_pages=states.count(PageAllocationState.RESERVED),
                live_pages=states.count(PageAllocationState.LIVE),
                reclaim_pending_pages=states.count(PageAllocationState.RECLAIM_PENDING),
                permanent_pages=permanent,
                request_owned_pages=sum(meta.request_refs > 0 for meta in self._pages),
                cache_owned_pages=sum(meta.cache_refs > 0 for meta in self._pages),
                shared_pages=sum(meta.is_shared for meta in self._pages),
                inflight_pages=sum(meta.inflight_refs > 0 for meta in self._pages),
                reclaimed_pages_total=self._reclaimed_pages_total,
            )

    def assert_invariants(self) -> None:
        """Debug-only exhaustive check of allocator invariants.

        Verifies the free queue holds each free page exactly once, free pages
        carry no residual metadata, refcounts stay non-negative, reserved/live
        pages are owned or in deferred flight, and the page-state buckets
        partition total capacity.
        """
        with self._lock:
            free_indices = list(self._ready_free)
            if len(free_indices) != len(set(free_indices)):
                raise InvariantViolationError("duplicate page in ready-free queue")
            free_set = set(free_indices)

            for index, meta in enumerate(self._pages):
                counts = (
                    meta.request_refs,
                    meta.cache_refs,
                    meta.reservation_refs,
                    meta.inflight_refs,
                )
                if any(count < 0 for count in counts):
                    raise InvariantViolationError("negative page refcount")
                if not 0 <= meta.valid_tokens <= self._page_size:
                    raise InvariantViolationError("invalid page token count")
                if meta.ownership_refs != 0 and meta.pending_free_epoch is not None:
                    raise InvariantViolationError("owned page retains a stale deferred-free epoch")

                if meta.allocation_state is PageAllocationState.FREE:
                    if index not in free_set:
                        raise InvariantViolationError("free page missing from free queue")
                    if meta.total_refs or meta.valid_tokens or meta.pending_free_epoch is not None:
                        raise InvariantViolationError("free page retains live metadata")
                else:
                    if index in free_set:
                        raise InvariantViolationError("allocated page appears in free queue")

                if meta.allocation_state is PageAllocationState.RESERVED:
                    if meta.reservation_refs not in (0, 1):
                        raise InvariantViolationError("invalid reserved-page owner count")
                    # Zero-owner reserved pages exist only between an in-flight
                    # failure and the safe epoch (abandon_reserved path).
                    if meta.reservation_refs == 0 and not (
                        meta.inflight_refs > 0 and meta.pending_free_epoch is not None
                    ):
                        raise InvariantViolationError("unowned reserved page is not in flight")
                elif meta.allocation_state is PageAllocationState.LIVE:
                    if meta.reservation_refs:
                        raise InvariantViolationError("live page retains reservation owner")
                    # A LIVE page with zero ownership refs must be deferred
                    # until its in-flight ownership disappears (release while a
                    # step still runs).
                    if meta.ownership_refs == 0 and not (
                        meta.inflight_refs > 0 and meta.pending_free_epoch is not None
                    ):
                        raise InvariantViolationError("unowned live page is not deferred")
                elif meta.allocation_state is PageAllocationState.RECLAIM_PENDING:
                    if meta.total_refs:
                        raise InvariantViolationError("deferred page still has references")
                    if meta.pending_free_epoch is None:
                        raise InvariantViolationError("deferred page has no safe epoch")
                elif meta.allocation_state is PageAllocationState.PERMANENT:
                    if meta.pending_free_epoch is not None:
                        raise InvariantViolationError("permanent page cannot be reclaimed")

            snapshot = self.snapshot()
            partition = (
                snapshot.free_pages
                + snapshot.reserved_pages
                + snapshot.live_pages
                + snapshot.reclaim_pending_pages
                + snapshot.permanent_pages
            )
            if partition != self._total_pages:
                raise InvariantViolationError("page-state partition does not equal capacity")

    def _get_meta(self, handle: KVPageHandle) -> PageMeta:
        """Resolve a handle under lock, rejecting stale generations and FREE pages."""
        if handle.index >= self._total_pages:
            raise InvalidHandleError(f"page index {handle.index} is out of range")
        meta = self._pages[handle.index]
        if (
            meta.generation != handle.generation
            or meta.allocation_state is PageAllocationState.FREE
        ):
            raise InvalidHandleError(f"stale page handle: {handle}")
        return meta

    def _require_state(
        self,
        handle: KVPageHandle,
        expected: PageAllocationState,
    ) -> PageMeta:
        """Resolve a handle and assert its allocation state."""
        meta = self._get_meta(handle)
        if meta.allocation_state is not expected:
            raise InvalidStateTransitionError(
                f"page {handle} is {meta.allocation_state.name}, expected {expected.name}"
            )
        return meta

    def _require_live_or_reserved(self, handle: KVPageHandle) -> PageMeta:
        """Resolve a handle that an execution step may touch."""
        meta = self._get_meta(handle)
        if meta.allocation_state not in (
            PageAllocationState.RESERVED,
            PageAllocationState.LIVE,
        ):
            raise InvalidStateTransitionError(
                f"page {handle} cannot participate in an execution step"
            )
        return meta

    def _reset_for_new_generation(self, meta: PageMeta) -> None:
        """Reinitialize metadata for a fresh ownership lifetime on a FREE page."""
        if meta.allocation_state is not PageAllocationState.FREE:
            raise InvariantViolationError("only a free page can start a new generation")
        meta.generation += 1
        meta.request_refs = 0
        meta.cache_refs = 0
        meta.reservation_refs = 0
        meta.inflight_refs = 0
        meta.valid_tokens = 0
        meta.pending_free_epoch = None
        meta.last_access_epoch = self._current_epoch

    def _make_free(self, meta: PageMeta) -> None:
        """Return a fully unrefed page to the ready-free queue."""
        index = meta.physical_id.value
        if meta.total_refs:
            raise InvariantViolationError("cannot free a referenced page")
        meta.allocation_state = PageAllocationState.FREE
        meta.valid_tokens = 0
        meta.pending_free_epoch = None
        meta.last_access_epoch = self._current_epoch
        self._ready_free.append(index)

    def _clear_stale_pending_free(self, meta: PageMeta) -> None:
        """Invalidate a leftover deferred-free epoch once a page has an owner.

        ``pending_free_epoch`` is only meaningful while ``ownership_refs == 0``
        (see :meth:`_mark_for_reclaim_if_unowned`): a page can reach that state
        while still ``inflight_refs > 0`` and stay LIVE/RESERVED rather than
        transition to RECLAIM_PENDING. Every acquire call must clear the field
        here rather than rely on callers never acquiring a ref on such a page,
        because :meth:`_mark_for_reclaim_if_unowned` folds any surviving value
        into the *next* deferred-free decision via ``max(previous, safe_epoch)``
        -- a stale, too-large epoch would then delay reuse for an unrelated,
        later owner's release with no safety benefit (never unsafe, only a
        silent capacity leak). Idempotent: a page whose ownership was already
        non-zero already has ``pending_free_epoch is None``, so this is a no-op
        in the common case.
        """
        meta.pending_free_epoch = None

    def _mark_for_reclaim_if_unowned(self, meta: PageMeta, *, safe_epoch: int) -> None:
        """Queue a page for deferred free once it has no ownership refs.

        Keeps the latest requested epoch (``max``) so an earlier safe epoch can
        never free a page a later release marked for a later step. If the page
        still has in-flight refs, it stays LIVE/RESERVED and only records
        ``pending_free_epoch``; ``unmark_inflight`` re-enters here to complete
        the transition to RECLAIM_PENDING.
        """
        if meta.ownership_refs != 0:
            return
        previous = meta.pending_free_epoch
        meta.pending_free_epoch = max(previous or safe_epoch, safe_epoch)
        if meta.inflight_refs:
            return
        meta.allocation_state = PageAllocationState.RECLAIM_PENDING
        # Heap tuple (pending_free_epoch, index, generation): epoch orders
        # safety; index and generation disambiguate ties and invalidate stale
        # heap entries after a page is recycled.
        heapq.heappush(
            self._reclaim_heap,
            (meta.pending_free_epoch, meta.physical_id.value, meta.generation),
        )

    def _reclaim_completed_locked(self, completed_epoch: int) -> int:
        """Pop reclaim entries whose epoch is done, freeing safe pages.

        Each entry is revalidated against the live page state before freeing:
        a recycled generation, a changed state, or a changed epoch means the
        heap entry is stale and is skipped.
        """
        reclaimed = 0
        while self._reclaim_heap and self._reclaim_heap[0][0] <= completed_epoch:
            epoch, index, generation = heapq.heappop(self._reclaim_heap)
            meta = self._pages[index]
            if meta.generation != generation:
                continue
            if meta.allocation_state is not PageAllocationState.RECLAIM_PENDING:
                continue
            if meta.pending_free_epoch != epoch:
                continue
            if meta.total_refs:
                raise InvariantViolationError("reclaim queue contains a referenced page")
            self._make_free(meta)
            reclaimed += 1
            self._reclaimed_pages_total += 1
        return reclaimed

    @staticmethod
    def _unique_pages(pages: Iterable[KVPageHandle]) -> tuple[KVPageHandle, ...]:
        """Deduplicate page handles while preserving first-seen order.

        ``dict.fromkeys`` keeps insertion order (Python 3.7+), so one step's
        shared pages count once without reordering.
        """
        return tuple(dict.fromkeys(pages))

    @staticmethod
    def _validate_epoch(epoch: int) -> None:
        """Reject negative epochs early so epoch math stays non-negative."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
