from __future__ import annotations

from typing import Any

import msgspec

from .output import ModelOutput, RequestOutput, SampledToken, UsageDelta
from .plan import ExecutionPlan, KVOp, SamplingMeta, SeqSlice
from .request import RequestIR, SamplingParams

__all__ = [
    "IRDecodeError",
    "IRSerializationError",
    "IRVersionMismatch",
    "IR_VERSION",
    "decode",
    "encode",
    "registered_types",
    "wrap",
]

IR_VERSION = 1

class IRSerializationError(Exception):
    """Base class for every encode/decode failure in this module."""


class IRVersionMismatch(IRSerializationError):
    """The buffer was encoded under a different `IR_VERSION`."""


class IRDecodeError(IRSerializationError):
    """Corrupt buffer, unknown tag, or a payload that does not match its tag."""


_TYPES: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        SamplingParams,
        RequestIR,
        KVOp,
        SeqSlice,
        SamplingMeta,
        ExecutionPlan,
        SampledToken,
        ModelOutput,
        UsageDelta,
        RequestOutput,
    )
}
_TAGS: dict[type, str] = {cls: tag for tag, cls in _TYPES.items()}

#: `[version, tag, payload]` — the payload stays raw until the type is known.
type _Envelope = tuple[int, str, msgspec.Raw]

_encode_msgpack = msgspec.msgpack.encode
_decode_msgpack = msgspec.msgpack.decode


def registered_types() -> tuple[type, ...]:
    """Every `ir/` type this module knows how to serialize."""
    return tuple(_TYPES.values())


def wrap(version: int, tag: str, payload: bytes) -> bytes:
    """Put an already-encoded msgpack payload into an envelope."""
    return _encode_msgpack((version, tag, msgspec.Raw(payload)))


def encode(obj: Any) -> bytes:
    """Serialize one `ir/` dataclass to msgpack inside an envelope."""
    tag = _TAGS.get(type(obj))
    if tag is None:
        raise IRSerializationError(
            f"{type(obj).__name__} is not an ayaka.ir type; encodable types are {sorted(_TYPES)}"
        )
    try:
        payload = _encode_msgpack(obj)
    except (TypeError, msgspec.EncodeError) as exc:
        raise IRSerializationError(f"cannot encode {tag}: {exc}") from exc
    return wrap(IR_VERSION, tag, payload)


def decode[T](buf: bytes, *, expect: type[T] | None = None) -> T:
    """Deserialize a buffer produced by `encode()`.

    `expect` is an optional assertion for a receiver that already knows which
    type it is waiting for — cheap, and turns a routing bug into an error at
    the boundary.
    """
    try:
        version, tag, payload = _decode_msgpack(buf, type=_Envelope)
    except msgspec.ValidationError as exc:
        raise IRDecodeError(f"not an ayaka.ir envelope [version, tag, payload]: {exc}") from exc
    except msgspec.DecodeError as exc:
        raise IRDecodeError(f"not a well-formed ayaka.ir envelope: {exc}") from exc

    if version != IR_VERSION:
        raise IRVersionMismatch(
            f"IR version mismatch: buffer was encoded with IR_VERSION={version}, "
            f"this process decodes IR_VERSION={IR_VERSION}. "
            f"Two processes are running different builds — restart them from one build."
        )

    cls = _TYPES.get(tag)
    if cls is None:
        raise IRDecodeError(f"unknown ayaka.ir type tag {tag!r}; known tags are {sorted(_TYPES)}")
    if expect is not None and cls is not expect:
        raise IRDecodeError(f"expected {expect.__name__}, buffer carries {tag}")

    try:
        return _decode_msgpack(payload, type=cls)
    except msgspec.DecodeError as exc:
        raise IRDecodeError(f"payload does not match its {tag} tag: {exc}") from exc
