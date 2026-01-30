"""
Structured logging configuration using structlog.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import structlog


def setup_logging(
    log_level: str = "INFO",
    log_file: Path | None = None,
    json_format: bool = True,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        json_format: Whether to use JSON format for logs
    """
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        stream=sys.stdout,
    )

    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logging.getLogger().addHandler(file_handler)

    # Configure structlog processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        shared_processors.append(
            structlog.dev.ConsoleRenderer(colors=True, exception_formatter=structlog.dev.plain_traceback)
        )

    structlog.configure(
        processors=shared_processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger for the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capability to any class."""

    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


def log_progress(
    logger: structlog.stdlib.BoundLogger,
    current: int,
    total: int,
    source: str,
    topic: str,
    **extra: Any,
) -> None:
    """
    Log collection progress.

    Args:
        logger: Logger instance
        current: Current item number
        total: Total items
        source: Data source
        topic: Topic being collected
        **extra: Additional context
    """
    percentage = (current / total * 100) if total > 0 else 0
    logger.info(
        "Collection progress",
        source=source,
        topic=topic,
        current=current,
        total=total,
        percentage=f"{percentage:.1f}%",
        **extra,
    )


def log_error(
    logger: structlog.stdlib.BoundLogger,
    error: Exception,
    context: str,
    **extra: Any,
) -> None:
    """
    Log an error with context.

    Args:
        logger: Logger instance
        error: The exception
        context: Description of what was happening
        **extra: Additional context
    """
    logger.error(
        "Error occurred",
        error_type=type(error).__name__,
        error_message=str(error),
        context=context,
        **extra,
        exc_info=True,
    )


def log_stats(
    logger: structlog.stdlib.BoundLogger,
    stats: dict[str, Any],
    context: str = "Statistics",
) -> None:
    """
    Log statistics.

    Args:
        logger: Logger instance
        stats: Statistics dictionary
        context: Description context
    """
    logger.info(context, **stats)
