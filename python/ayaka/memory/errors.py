"""Typed failures raised by the runtime-memory control plane.

Each exception maps to a distinct ownership or lifecycle violation so callers
can distinguish stale handles, invalid state transitions, accounting bugs, and
resource exhaustion without inspecting page internals.
"""

from __future__ import annotations


class RuntimeMemoryError(RuntimeError):
    """Base class for runtime-memory failures.

    Every control-plane failure in this package derives from this type, giving
    callers a single catch-all for memory-subsystem errors.
    """


class InvalidHandleError(RuntimeMemoryError):
    """A handle is stale, out of range, or belongs to no live object.

    Raised when a ``SequenceHandle``/``KVPageHandle`` generation no longer
    matches the arena/allocator generation, or when the index is out of range.
    Terminated transactions, reservations, and leases also surface here.
    """


class InvalidStateTransitionError(RuntimeMemoryError):
    """An operation is not valid in the object's current lifecycle state.

    For example, releasing a page that is not ``LIVE``, preparing an empty
    transaction, or completing a step that is not ``IN_FLIGHT``.
    """


class InvariantViolationError(RuntimeMemoryError):
    """An internal accounting or ownership invariant was violated.

    Raised by the debug ``assert_invariants`` paths and by any operation that
    detects refcount underflow, duplicated free entries, or an inconsistent
    page-state partition. Indicates a bug in the memory subsystem itself.
    """


class SequenceCapacityError(RuntimeMemoryError):
    """The sequence arena has no free slots.

    Raised when ``max_sequences`` concurrent sequences are already live and a
    new ``create_sequence`` call cannot be satisfied.
    """


class SequenceBusyError(RuntimeMemoryError):
    """A sequence already participates in a transaction or execution lease.

    A sequence may participate in at most one transaction or execution lease;
    cache/attach/release operations on a busy sequence fail with this error.
    """


class TransactionClosedError(RuntimeMemoryError):
    """A transaction is no longer open for reservation.

    Raised when reservations are attempted after the transaction was prepared,
    rolled back, or already closed.
    """


class StorageUnavailableError(RuntimeMemoryError):
    """The requested optional tensor-storage backend is unavailable.

    Raised when PyTorch (or a specific dtype) is missing while constructing
    ``TorchMHAKVStorage``/``TorchMLAKVStorage`` or running storage adapters.
    """
