"""Central logging setup: one JSON formatter on the root logger.

Level comes from ``LOG_LEVEL`` (default INFO). ``configure_logging`` is
idempotent so it can be called from the app lifespan and from tests.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from config import settings


class JsonFormatter(logging.Formatter):
    """Render each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Install the JSON handler on the root logger."""
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
