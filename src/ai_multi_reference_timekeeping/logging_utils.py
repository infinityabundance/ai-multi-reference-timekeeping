"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Any

from .config import LoggingConfig


class JsonFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(config: LoggingConfig) -> None:
    """Configure root logger."""

    level = getattr(logging, config.level.upper(), logging.INFO)
    handlers: list[logging.Handler] = []
    if config.log_file:
        handlers.append(
            RotatingFileHandler(
                config.log_file,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
            )
        )
    else:
        handlers.append(logging.StreamHandler())

    formatter: logging.Formatter
    if config.json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("%(levelname)s %(name)s %(message)s")

    for handler in handlers:
        handler.setFormatter(formatter)

    logging.basicConfig(level=level, handlers=handlers)
