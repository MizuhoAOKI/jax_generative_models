import logging
from typing import Literal

from rich.logging import RichHandler


def setup_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
) -> None:
    """Sets up the global logging configuration using Rich.

    This configures the root logger to output formatted, colored logs to the console,
    including rich tracebacks for exceptions.

    Args:
        level: The logging level to set (e.g., "DEBUG", "INFO"). Defaults to "INFO".
    """
    # Create a mapping from string to logging level constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = level_map.get(level.upper(), logging.INFO)

    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,  # Enable colorful tracebacks
                markup=True,  # Allow Rich markup in logs
                show_path=False,  # Hide file path for cleaner output
            )
        ],
    )
