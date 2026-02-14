import logging


logger = logging.getLogger(__name__)


def setup_logging(level_name = "INFO") -> None:
    """Configure the root logger for the application.

    Args:
        level_name: The textual logging level, such as INFO or DEBUG.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(getattr(logging, level_name.upper(), logging.INFO))
        return

    formatter = logging.Formatter(
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger.setLevel(getattr(logging, level_name.upper(), logging.INFO))
    root_logger.addHandler(handler)

