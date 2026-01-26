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
        # Minimal structured payload for machine ingestion.
        payload: dict[str, Any] = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            # Include stack traces when provided.
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(config: LoggingConfig) -> None:
    """Configure root logger."""

    # Map the string level to the logging module value.
    level = getattr(logging, config.level.upper(), logging.INFO)
    handlers: list[logging.Handler] = []
    if config.log_file:
        # File-based logging with rotation.
        handlers.append(
            RotatingFileHandler(
                config.log_file,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
            )
        )
    else:
        # Default to stdout.
        handlers.append(logging.StreamHandler())

    formatter: logging.Formatter
    if config.json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("%(levelname)s %(name)s %(message)s")

    for handler in handlers:
        # Consistent formatting across handlers.
        handler.setFormatter(formatter)

    # Initialize root logging configuration.
    logging.basicConfig(level=level, handlers=handlers)
