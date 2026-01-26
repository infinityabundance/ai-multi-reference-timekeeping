"""Minimal TOML parser for settings."""

from __future__ import annotations

from typing import Any


def loads(content: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current: dict[str, Any] = data
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]")
            current = data.setdefault(section, {})
            continue
        if "=" in line:
            key, value = [part.strip() for part in line.split("=", 1)]
            current[key] = _parse_value(value)
    return data


def _parse_value(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith('"') and value.endswith('"'):
        return value.strip('"')
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
