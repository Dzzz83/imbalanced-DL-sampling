import logging
import logging.config
import os
import numpy as np

logging_level_dict = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG
}

def setup_logging():
    """Basic console logging for other modules (no file)."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')

def setup_logger(log_file_path, name='main', verbose=1):
    """
    Create a logger that writes to both console and a file.
    Both handlers use a simple format: only the log message.
    The file handler is line‑buffered (flushes after each line).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging_level_dict[verbose])
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler: simple format (just the message)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging_level_dict[verbose])
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler: also simple format (just the message)
    if log_file_path:
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        # Open with buffering=1 (line buffering) to flush after each line
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8', delay=False)
        file_handler.setLevel(logging_level_dict[verbose])
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger, log_file_path

def create_distribution_table(logger, original_dict, selected_dict):
    """Create a formatted table showing class distribution before and after selection."""
    if not isinstance(original_dict, dict) or not isinstance(selected_dict, dict):
        raise TypeError("Both arguments must be dictionaries with class IDs as keys and counts as values.")
    
    all_classes = sorted(set(original_dict.keys()).union(set(selected_dict.keys())))
    max_class_id = max(all_classes) if all_classes else 0
    
    header = f"{'Class ID':<10} | {'Original':<10} | {'Selected':<10} | {'Keep %':<8}"
    separator = "-" * (10 + 3 + 10 + 3 + 10 + 3 + 8)
    logger.info(separator)
    logger.info(header)
    logger.info(separator)
    
    for cls_id in range(max_class_id + 1):
        orig = original_dict.get(cls_id, 0)
        sel = selected_dict.get(cls_id, 0)
        keep_percent = (sel / orig * 100) if orig > 0 else 0.0
        logger.info(f"{cls_id:<10} | {orig:<10} | {sel:<10} | {keep_percent:>6.1f}%")
    
    total_orig = sum(original_dict.values())
    total_sel = sum(selected_dict.values())
    total_keep_percent = (total_sel / total_orig * 100) if total_orig > 0 else 0.0
    logger.info(separator)
    logger.info(f"{'TOTAL':<10} | {total_orig:<10} | {total_sel:<10} | {total_keep_percent:>6.1f}%")
    logger.info(separator)

# Default configuration for other modules (console only)
setup_logging()