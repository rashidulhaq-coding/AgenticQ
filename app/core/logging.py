"""Logging configuration and setup for the application.

This module provides structured logging configuration using structlog,
with environment-specific formatters and handlers. It supports both
console-friendly development logging and JSON-formatted production logging.
"""

from contextvars import ContextVar
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import structlog

from app.core.config import Environment, settings

# Ensure log directory exists
settings.LOG_DIR.mkdir(parents=True, exist_ok=True)

# Context variables for storing request-specific data
_request_context: ContextVar[Dict[str, Any]] = ContextVar("request_context", default={})


def bind_context(**kwargs: Any) -> None:
    """Bind context variables to the current request.

    Args:
        **kwargs: Key-value pairs to bind to the logging context
    """
    current = _request_context.get()
    _request_context.set({**current, **kwargs})


def clear_context() -> None:
    """Clear all context variables for the current request."""
    _request_context.set({})


def get_context() -> Dict[str, Any]:
    """Get the current logging context.

    Returns:
        Dict[str, Any]: Current context dictionary
    """
    return _request_context.get()


def add_context_to_event_dict(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add context variables to the event dictionary.

    This processor adds any bound context variables to each log event.

    Args:
        logger: The logger instance
        method_name: The name of the logging method
        event_dict: The event dictionary to modify

    Returns:
        Dict[str, Any]: Modified event dictionary with context variables
    """
    context = get_context()
    if context:
        event_dict.update(context)
    return event_dict


def get_log_file_path() -> Path:
    """Get the current log file path based on date and environment.

    Returns:
        Path: The path to the log file
    """
    env_prefix = settings.ENVIRONMENT.value
    return settings.LOG_DIR / f"{env_prefix}-{datetime.now().strftime('%Y-%m-%d')}.jsonl"


class JsonlFileHandler(logging.Handler):
    """Custom handler for writing JSONL logs to daily files."""

    def __init__(self, file_path: Path):
        """Initialize the JSONL file handler.

        Args:
            file_path: Path to the log file where entries will be written.
        """
        super().__init__()
        self.file_path = file_path

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record to the JSONL file."""
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "filename": record.pathname,
                "line": record.lineno,
                "environment": settings.ENVIRONMENT.value,
            }
            if hasattr(record, "extra"):
                log_entry.update(record.extra)

            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """Close the handler."""
        super().close()


def get_structlog_processors(include_file_info: bool = True) -> List[Any]:
    """Get the structlog processors based on configuration.

    Args:
        include_file_info: Whether to include file information in the logs

    Returns:
        List[Any]: List of structlog processors
    """
    # Set up processors that are common to both outputs
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        # Add context variables (user_id, session_id, etc.) to all log events
        add_context_to_event_dict,
    ]

    # Add callsite parameters if file info is requested
    if include_file_info:
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.MODULE,
                    structlog.processors.CallsiteParameter.PATHNAME,
                }
            )
        )

    # Add environment info
    processors.append(lambda _, __, event_dict: {**event_dict, "environment": settings.ENVIRONMENT.value})

    return processors


def setup_logging() -> None:
    """Configure structured logging based on the environment.

    - ``console``  → pretty colourful output.
    - ``json``     → JSON output to a daily JSONL file.
    """
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO

    # Shared processors (callsite info only in local/dev environments)
    shared_processors = get_structlog_processors(
        include_file_info=settings.ENVIRONMENT in [Environment.LOCAL, Environment.DEVELOPMENT]
    )

    # Processors for foreign logs (standard library logs)
    # We exclude filter_by_level here because standard library logs are already filtered
    # at the logger/handler level. Including it here can cause AttributeError: 'NoneType'
    # if the logger object is not fully initialized (e.g., in Windows multiprocessing subprocesses).
    foreign_pre_chain = [p for p in shared_processors if p != structlog.stdlib.filter_by_level]

    # JSONL file handler — always active in every environment
    file_handler = JsonlFileHandler(get_log_file_path())
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=foreign_pre_chain,
        )
    )

    # Development-friendly console logging (colourful stdout + file)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
            foreign_pre_chain=foreign_pre_chain,
        )
    )

    # Configure structlog based on environment
    if settings.LOG_FORMAT == "console":
        # Development-friendly console logging (colourful stdout + file)
        handlers = [console_handler, file_handler]
    else:
        # Production JSON logging
        handlers = [file_handler]

    # Apply handlers via basicConfig
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=handlers,
    )

    # Silence noisy third-party loggers
    for logger_name in [
        "asyncio",
        "python_multipart",
        "httpx",
        "httpcore",
        "langsmith",
        "urllib3",
        "openai",
        "boto3",
        "botocore",
        "s3transfer",
        "pdfminer",
        "sse_starlette",
        "sse",
        "rustls",
        "hickory_net",
        "hickory_resolver",
        "h2",
        "hyper_util",
        "primp",
        "ddgs",
        "cookie_store",
        "reqwest",
        "hpack",
        "langchain",
        "langgraph",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Add a filter to suppress non-app DEBUG logs at the handler level
    class AppOnlyFilter(logging.Filter):
        """Only allow DEBUG+ logs from app.* modules; suppress DEBUG from others."""

        def filter(self, record: logging.LogRecord) -> bool:
            if record.levelno >= logging.INFO:
                return True
            return record.name.startswith("app.")

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(AppOnlyFilter())

    logging.getLogger("unstructured").setLevel(logging.ERROR)

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Initialize logging
setup_logging()

# Create logger instance
logger = structlog.get_logger()
logger.info(
    "logging_initialized",
    environment=settings.ENVIRONMENT.value,
    log_level=settings.LOG_LEVEL.upper(),
    log_format=settings.LOG_FORMAT,
    debug=settings.DEBUG,
)
