from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import os
import sys
from typing import Any, Dict

request_id_ctx_var: ContextVar[str] = ContextVar("request_id", default="")


class StructuredJsonFormatter(logging.Formatter):
    """Outputs production logs in structured JSON format."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        req_id = request_id_ctx_var.get()
        if req_id:
            log_obj["request_id"] = req_id

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


class HumanReadableFormatter(logging.Formatter):
    """Clean readable log formatter for development and terminal outputs."""

    def format(self, record: logging.LogRecord) -> str:
        req_id = request_id_ctx_var.get()
        req_tag = f" [{req_id[:8]}]" if req_id else ""
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{date_str}] [{record.levelname}]{req_tag} [{record.name}] {record.getMessage()}"
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return msg


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("cyclone_x")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        env = os.getenv("ENVIRONMENT", "development").lower()
        if env == "production":
            handler.setFormatter(StructuredJsonFormatter())
        else:
            handler.setFormatter(HumanReadableFormatter())

        logger.addHandler(handler)
        logger.propagate = False

    return logger


logger = setup_logging()
