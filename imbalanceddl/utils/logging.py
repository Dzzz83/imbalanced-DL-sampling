import logging
import os

def setup_logger(exp_name):
    os.makedirs("results", exist_ok=True)
    
    log_filename = f"results/{exp_name}.log"
    
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_filename, mode='a', delay=True)
    fh.setFormatter(logging.Formatter("%(message)s")) 
    
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt='%H:%M:%S'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_filename 

def create_distribution_table(logger, counts_original, counts_selection=None):
    sep = "-" * 38
    header = f"{'Class ID':<10} | {'Original':<10} | {'Selected':<10}"

    logger.info(sep)
    logger.info(header)
    logger.info(sep)

    total_orig = 0
    total_sel = 0

    # log each class
    for i in sorted(counts_original.keys()):
        before = counts_original[i]
        after = counts_selection.get(i, 0) if counts_selection is not None else before
        
        total_orig += before
        total_sel += after
        
        logger.info(f"{i:<10} | {before:<10} | {after:<10}")

    # final row
    logger.info(sep)
    logger.info(f"{'TOTAL':<10} | {total_orig:<10} | {total_sel:<10}")
    
    # log the percentage kept
    perc = (total_sel / total_orig) * 100 if total_orig > 0 else 0
    logger.info(f"Summary: Kept {total_sel}/{total_orig} samples ({perc:.1f}%)")
    logger.info(sep)