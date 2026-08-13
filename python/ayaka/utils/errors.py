"""Error taxonomy for the utils layer."""

from __future__ import annotations

__all__ = [
    "AyakaError",
    "CapabilityError",
    "ConfigError",
    "IntegrityError",
    "capability_check",
]


class AyakaError(Exception):
    """Base class for every error raised by Ayaka."""


class CapabilityError(AyakaError):
    """A required hardware/software capability is missing.

    Carries structured fields so callers can branch on the *reason*
    rather than on a substring of the message.

    Args:
        capability: Stable identifier, e.g. ``"sm90"``, ``"triton>=3.0"``,
            ``"flash_attn_varlen"``. It shows up in logs and metrics, so
            keep it stable across releases.
        detail: What was found instead.
        remedy: What the operator can do about it, if anything.
    """

    def __init__(
        self,
        capability: str,
        detail: str = "",
        remedy: str | None = None,
    ) -> None:
        self.capability = capability
        self.detail = detail
        self.remedy = remedy

        msg = f"missing capability {capability!r}"
        if detail:
            msg += f": {detail}"
        if remedy:
            msg += f" ({remedy})"
        super().__init__(msg)


class ConfigError(AyakaError):
    """Configuration is malformed, contradictory, or out of range."""


class IntegrityError(AyakaError):
    """On-disk or in-flight data failed a verification check."""


def capability_check(
    ok: bool,
    capability: str,
    detail: str = "",
    remedy: str | None = None,
) -> None:
    """Raise `CapabilityError` unless `ok`.

    Examples:
        >>> capability_check(True, "sm90")

        >>> capability_check(False, "fp8_gemm", "device is sm_80",
        ...                  "use an H100 or newer")
        Traceback (most recent call last):
            ...
        ayaka.utils.errors.CapabilityError: missing capability 'fp8_gemm': device is sm_80
        (use an H100 or newer)

        Callers branch on the structured field, never on the message:

        >>> try:
        ...     capability_check(False, "fp8_gemm", "device is sm_80")
        ... except CapabilityError as exc:
        ...     exc.capability
        'fp8_gemm'
    """
    if not ok:
        raise CapabilityError(capability, detail, remedy)
