import logging
import os

def setup_logger(log_path):
    """
    log_path: full path to the log file (e.g., 'results/train/xxx.log')
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger = logging.getLogger(log_path)  # unique logger per file
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_path, mode='w', delay=True)  # 'w' overwrites per run
    fh.setFormatter(logging.Formatter("%(message)s"))
    
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt='%H:%M:%S'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_path

def create_distribution_table(logger, counts_original, counts_selection=None):
    sep = "-" * 52
    header = f"{'Class ID':<10} | {'Original':<10} | {'Selected':<10} | {'Keep %':<8}"

    logger.info(sep)
    logger.info(header)
    logger.info(sep)

    total_orig = 0
    total_sel = 0

    for i in sorted(counts_original.keys()):
        before = counts_original[i]
        after = counts_selection.get(i, 0) if counts_selection is not None else before
        
        total_orig += before
        total_sel += after
        
        keep_perc = (after / before * 100) if before > 0 else 0
        logger.info(f"{i:<10} | {before:<10} | {after:<10} | {keep_perc:>7.1f}%")

    logger.info(sep)
    total_perc = (total_sel / total_orig * 100) if total_orig > 0 else 0
    logger.info(f"{'TOTAL':<10} | {total_orig:<10} | {total_sel:<10} | {total_perc:>7.1f}%")
    logger.info(sep)