"""Minimal Pydantic compatibility layer for local validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass
class FieldInfo:
    default: Any
    ge: float | None = None
    le: float | None = None
    description: str | None = None
    default_factory: Any | None = None


def Field(
    default: Any = None,
    *,
    ge: float | None = None,
    le: float | None = None,
    description: str | None = None,
    default_factory: Any | None = None,
) -> FieldInfo:
    return FieldInfo(default=default, ge=ge, le=le, description=description, default_factory=default_factory)


class ConfigDict(dict):
    pass


class BaseModel:
    model_config: Mapping[str, Any] = {}

    def __init__(self, **data: Any) -> None:
        annotations = getattr(self, "__annotations__", {})
        for name in annotations:
            value = data.get(name, getattr(self.__class__, name, None))
            field_info = value if isinstance(value, FieldInfo) else None
            if field_info:
                if field_info.default_factory is not None:
                    value = field_info.default_factory()
                else:
                    value = field_info.default
            setattr(self, name, value)

    @classmethod
    def model_validate(cls, data: Mapping[str, Any]) -> "BaseModel":
        instance = cls(**data)
        for name, annotation in getattr(cls, "__annotations__", {}).items():
            value = getattr(instance, name, None)
            field_info = getattr(cls, name, None)
            if isinstance(field_info, FieldInfo) and value is not None:
                if field_info.ge is not None and value < field_info.ge:
                    raise ValueError(f"{name} below minimum")
                if field_info.le is not None and value > field_info.le:
                    raise ValueError(f"{name} above maximum")
        return instance

    def model_dump(self) -> Dict[str, Any]:
        return {key: getattr(self, key) for key in getattr(self, "__annotations__", {})}
