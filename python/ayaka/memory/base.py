from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, TypeVar

from .errors import InvalidHandleError, InvalidStateTransitionError
from .handler import KVReservationHandle, MemoryTransactionHandle, StepMemoryLeaseHandle


class _HandleRecord(Protocol):
    """Structural contract every transaction/reservation/lease record meets."""

    handle: object


class _ReservationRecordLike(Protocol):
    """The subset of a reservation record the written-token check needs."""

    handle: KVReservationHandle
    num_new_tokens: int


TxT = TypeVar("TxT", bound=_HandleRecord)  # Transaction record type
ResT = TypeVar("ResT", bound=_ReservationRecordLike)  # Reservation record type
LeaseT = TypeVar("LeaseT", bound=_HandleRecord)  # Lease record type


class TransactionRegistryMixin[
    TxT: _HandleRecord,
    ResT: _ReservationRecordLike,
    LeaseT: _HandleRecord,
]:
    _transactions: dict[int, TxT]  # Maps index -> record
    _reservations: dict[int, ResT]  # Maps index -> record
    _leases: dict[int, LeaseT]  # Maps index -> record

    def _get_transaction(self, handle: MemoryTransactionHandle) -> TxT:
        """Resolve a transaction handle, rejecting stale generations."""
        record = self._transactions.get(handle.index)
        if record is None or record.handle != handle:
            raise InvalidHandleError(f"stale transaction handle: {handle}")
        return record

    def _get_reservation(self, handle: KVReservationHandle) -> ResT:
        """Resolve a reservation handle, rejecting stale generations."""
        record = self._reservations.get(handle.index)
        if record is None or record.handle != handle:
            raise InvalidHandleError(f"stale reservation handle: {handle}")
        return record

    def _get_lease(self, handle: StepMemoryLeaseHandle) -> LeaseT:
        """Resolve a lease handle, rejecting stale generations."""
        record = self._leases.get(handle.index)
        if record is None or record.handle != handle:
            raise InvalidHandleError(f"stale step-memory lease: {handle}")
        return record

    @staticmethod
    def _validate_written_tokens(
        records: Sequence[ResT],
        written_tokens: Mapping[KVReservationHandle, int] | None,
        *,
        mismatch_message: str,
        incomplete_message: str,
    ) -> None:
        if written_tokens is None:
            return
        expected_handles = {record.handle for record in records}
        if set(written_tokens) != expected_handles:
            raise InvalidStateTransitionError(mismatch_message)
        for record in records:
            if written_tokens[record.handle] != record.num_new_tokens:
                raise InvalidStateTransitionError(incomplete_message)
