import logging


logger = logging.getLogger(__name__)


def setup_logging(level_name = "INFO") -> None:
    """Configure logging with debug-only output policy.

    Args:
        level_name: The textual logging level, only DEBUG enables output.
    """
    normalized_level = str(level_name or "").strip().upper()
    if normalized_level != "DEBUG":
        logging.disable(logging.CRITICAL)
        return

    logging.disable(logging.NOTSET)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(logging.DEBUG)
        return

    formatter = logging.Formatter(
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)
