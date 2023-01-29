"""Module for providing a configured logger."""
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Configuring Logging.

    The log level depends on the environment variable `VM_LOG_LEVEL`.
    - 0: NOTSET, will be set to DEBUG
    - 1: DEBUG
    - 2: INFO (default)
    - 3: WARNING
    - 4: ERROR
    - 5: CRITICAL

    Args:
        name: module name.

    Returns:
        logger: configured logger.
    """
    # create logger
    logger = logging.getLogger(name=name)
    logger.propagate = False
    log_level = 1
    log_level_int = max(int(log_level) * 10, 10)
    logger.setLevel(logging.DEBUG)
    # create console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(log_level_int)
    # create formatter and add it to handler
    formatter = logging.Formatter(
        fmt="%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    # add handler to the logger
    logger.addHandler(console_handler)
    return logger
