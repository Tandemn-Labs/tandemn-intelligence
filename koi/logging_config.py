"""
koi/logging_config.py — Structured logging via structlog.

Usage:
    from koi.logging_config import setup_logging, get_logger
    setup_logging()  # call once at startup
    logger = get_logger("koi.server")
    logger.info("job_started", job_id="abc", gpu_type="H100")

Context propagation:
    from koi.logging_config import bind_context, clear_context
    bind_context(request_id="abc123", job_id="job-1")
    logger.info("processing")  # automatically includes request_id + job_id
    clear_context()
"""

import logging
import os

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars, merge_contextvars


def setup_logging(json_output: bool = None, level: str = None):
    """Configure structlog + stdlib logging. Call once at startup."""
    if json_output is None:
        json_output = os.environ.get("KOI_LOG_FORMAT", "console") == "json"
    if level is None:
        level = os.environ.get("KOI_LOG_LEVEL", "INFO")

    shared_processors = [
        merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    structlog.configure(
        processors=shared_processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger by name."""
    return structlog.get_logger(name)


def bind_context(**kwargs):
    """Bind key-value pairs to the current async context (propagates to all logs)."""
    bind_contextvars(**kwargs)


def clear_context():
    """Clear all bound context variables."""
    clear_contextvars()
