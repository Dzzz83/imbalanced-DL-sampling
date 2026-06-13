import logging
import os
from datetime import datetime

_debug_logger = None

def get_debug_logger(debug=False, log_dir=None):
    """
    Return a logger for debug messages.
    If debug is False, returns a dummy logger that does nothing.
    The log file is placed in './debug' directory (or a custom log_dir if provided).
    """
    global _debug_logger
    if _debug_logger is not None:
        return _debug_logger
    if not debug:
        _debug_logger = logging.getLogger('debug_dummy')
        _debug_logger.addHandler(logging.NullHandler())
        return _debug_logger

    # Use fixed directory relative to project root (default './debug')
    if log_dir is None:
        log_dir = './debug'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'debug_{timestamp}.log')

    logger = logging.getLogger('debug')
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    _debug_logger = logger
    return _debug_logger