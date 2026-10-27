"""Logging for DeepRM: "[time] LEVEL message" to standard error, in colour on a terminal; optionally a log file."""

import logging
import os
import sys

_DATEFMT = "%Y-%m-%d %H:%M:%S"
_RESET = "\033[0m"
_LEVEL_COLOR = {
    logging.DEBUG: "\033[34m",  # blue
    logging.INFO: "\033[36m",  # cyan
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[35m",  # magenta
}


def is_terminal(stream) -> bool:
    """True if `stream` is a terminal; False for a file, a pipe, or a stream that cannot tell."""
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):  # no isatty, or a closed stream
        return False


def use_color(stream) -> bool:
    """True if `stream` is a terminal and NO_COLOR is unset or empty (https://no-color.org)."""
    return not os.environ.get("NO_COLOR") and is_terminal(stream)


class ColorFormatter(logging.Formatter):
    """Formats a record as "[time] LEVEL message", with the message coloured by level when use_color is True.

    Later lines of the message, and a traceback, are indented to line up under the first.
    """

    def __init__(self, datefmt: str = _DATEFMT, use_color: bool = True):
        super().__init__(datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Returns the record as "[time] LEVEL message"."""
        prefix = f"[{self.formatTime(record, self.datefmt)}] {record.levelname}"
        message = record.getMessage()  # keeps logging's lazy %-formatting
        if record.exc_info:  # log.exception() or exc_info=True
            message += "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            message += "\n" + self.formatStack(record.stack_info)
        message = message.replace("\n", "\n" + " " * (len(prefix) + 1))  # line up later lines under the first
        color = _LEVEL_COLOR.get(record.levelno, "") if self.use_color else ""
        return f"{prefix} {color}{message}{_RESET if color else ''}"


def get_logger(name: str = "deeprm", level: int = logging.INFO) -> logging.Logger:
    """Returns a logger that writes to standard error, in colour only when standard error is a terminal.

    The logger is set up on the first call; later calls return it unchanged and ignore `level`.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(ColorFormatter(use_color=use_color(sys.stderr)))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def add_log_path(logger: logging.Logger, path: str, level: int = logging.INFO) -> None:
    """Adds a plain-text (no colour) log file at `path` to `logger`. A second call with the same file adds nothing."""
    full_path = os.path.abspath(path)
    if any(isinstance(h, logging.FileHandler) and h.baseFilename == full_path for h in logger.handlers):
        return None  # already logging to this file
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(fmt="[{asctime}] {levelname} {name}: {message}", datefmt=_DATEFMT, style="{")
    )
    logger.addHandler(file_handler)
    logger.info(f"Logging to file: {path}")
    return None
