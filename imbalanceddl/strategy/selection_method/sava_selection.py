import numpy as np
import os
from imbalanceddl.utils.sava_helpers import get_sava_sorted_indices
<<<<<<< HEAD
from imbalanceddl.utils.debug_logger import get_debug_logger
=======
>>>>>>> hieu

def get_sava_selection_indices(train_dataset, val_dataset, keep_ratio,
                               device='cuda', file_key=None, batch_size=1024,
                               num_classes=10, resize=32,
<<<<<<< HEAD
                               cache_label_distances=True, corrupt_por=0.0,
                               debug=False):
    """
    Returns indices to keep (lowest scores = most valuable).

    Args:
        debug: If True, write detailed debug information to a log file.
    """
    logger = get_debug_logger(debug=debug)
    
    if file_key is not None:
        cache_dir = 'sava_selection_results'
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{file_key}.npy")
        if os.path.exists(cache_path):
            print(f"Loading SAVA sorted indices from cache: {cache_path}")
            sorted_indices = np.load(cache_path)
            if debug:
                logger.debug(f"Loaded indices shape: {sorted_indices.shape}")
                logger.debug(f"First 10 loaded indices: {sorted_indices[:10]}")
                logger.debug(f"Last 10 loaded indices: {sorted_indices[-10:]}")
        else:
            if debug:
                logger.debug("Cache not found, computing SAVA scores...")
            sorted_indices = get_sava_sorted_indices(
                train_dataset, val_dataset, device=device, batch_size=batch_size,
                num_classes=num_classes, resize=resize,
                cache_label_distances=cache_label_distances, corrupt_por=corrupt_por,
                debug=debug   # propagate debug flag
            )
            np.save(cache_path, sorted_indices)
            print(f"Saved SAVA sorted indices to {cache_path}")
            if debug:
                logger.debug(f"Saved indices shape: {sorted_indices.shape}")
                logger.debug(f"First 10 computed indices: {sorted_indices[:10]}")
                logger.debug(f"Last 10 computed indices: {sorted_indices[-10:]}")
    else:
        if debug:
            logger.debug("No cache key provided, computing SAVA scores directly.")
        sorted_indices = get_sava_sorted_indices(
            train_dataset, val_dataset, device=device, batch_size=batch_size,
            num_classes=num_classes, resize=resize,
            cache_label_distances=cache_label_distances, corrupt_por=corrupt_por,
            debug=debug
=======
                               cache_label_distances=True, corrupt_por=0.0):
    """Returns indices to keep (lowest scores = most valuable)."""
    if file_key is not None:
        cache_dir = 'sava_selection_results'
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{file_key}.npy")   # Removed "_sorted_indices"
        if os.path.exists(cache_path):
            print(f"Loading SAVA sorted indices from cache: {cache_path}")
            sorted_indices = np.load(cache_path)
        else:
            sorted_indices = get_sava_sorted_indices(
                train_dataset, val_dataset, device=device, batch_size=batch_size,
                num_classes=num_classes, resize=resize,
                cache_label_distances=cache_label_distances, corrupt_por=corrupt_por
            )
            np.save(cache_path, sorted_indices)
            print(f"Saved SAVA sorted indices to {cache_path}")
    else:
        sorted_indices = get_sava_sorted_indices(
            train_dataset, val_dataset, device=device, batch_size=batch_size,
            num_classes=num_classes, resize=resize,
            cache_label_distances=cache_label_distances, corrupt_por=corrupt_por
>>>>>>> hieu
        )

    num_keep = int(len(train_dataset) * keep_ratio)
    keep_indices = sorted_indices[:num_keep]
    print(f"SAVA selection: keeping {num_keep} out of {len(train_dataset)} samples.")
<<<<<<< HEAD
    if debug:
        logger.debug(f"keep_ratio={keep_ratio}, num_keep={num_keep}")
        logger.debug(f"First 10 kept indices: {keep_indices[:10]}")
        logger.debug(f"Last 10 kept indices: {keep_indices[-10:]}")
=======
>>>>>>> hieu
    return keep_indices