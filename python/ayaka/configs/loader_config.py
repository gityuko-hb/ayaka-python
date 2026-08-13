from __future__ import annotations

import enum
from dataclasses import dataclass


class LoaderFormat(enum.StrEnum):
    AUTO = "auto"
    PT = "pt"
    SAFETENSORS = "safetensors"

    @classmethod
    def parse(cls, value: str | LoaderFormat) -> LoaderFormat:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            supported = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Invalid load format: {value}. Supported formats are: {supported}"
            ) from exc

@dataclass(slots=True)
class LoaderConfig:
    """Runtime options shared by local, Hub, remote and quantized loaders."""

    load_format: LoaderFormat | str = LoaderFormat.AUTO
