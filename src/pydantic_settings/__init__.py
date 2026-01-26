"""Minimal BaseSettings implementation for local use."""

from __future__ import annotations

import os
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict


class SettingsConfigDict(ConfigDict):
    pass


class BaseSettings(BaseModel):
    model_config: Mapping[str, Any] = {}

    def __init__(self, **data: Any) -> None:
        config = dict(self.model_config or {})
        env_prefix = config.get("env_prefix", "")
        env_nested_delimiter = config.get("env_nested_delimiter", "__")
        env_data: dict[str, Any] = {}
        for key, value in os.environ.items():
            if not key.startswith(env_prefix):
                continue
            stripped = key[len(env_prefix) :]
            parts = stripped.split(env_nested_delimiter) if env_nested_delimiter else [stripped]
            cursor = env_data
            for part in parts[:-1]:
                cursor = cursor.setdefault(part.lower(), {})
            cursor[parts[-1].lower()] = value
        merged = _merge_dicts(env_data, data)
        super().__init__(**merged)


def _merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
