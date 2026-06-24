# imbalanceddl/strategy/selection_method/sava_selection.py

import numpy as np
import os
from imbalanceddl.utils.sava_helpers import get_sava_sorted_indices
from imbalanceddl.utils.debug_logger import get_debug_logger

def get_sava_selection_indices(train_dataset, val_dataset, keep_ratio,
                               device='cuda', file_key=None, batch_size=1024,
                               num_classes=10, resize=32,
                               cache_label_distances=True, corrupt_por=0.0,
                               debug=False, return_scores=False):
    """
    Returns indices to keep (lowest scores = most valuable) and optionally the
    raw SAVA scores in original dataset order.

    Args:
        train_dataset: training dataset (plain, no augmentation)
        val_dataset: validation dataset
        keep_ratio: fraction of data to keep (0.0 to 1.0)
        device: 'cuda' or 'cpu'
        file_key: cache key (if None, no caching)
        batch_size: batch size for OT computation
        num_classes: number of classes
        resize: image resize dimension
        cache_label_distances: cache label‑to‑label distances
        corrupt_por: corruption portion (for SAVA paper experiments)
        debug: enable debug logging
        return_scores: if True, return (keep_indices, scores) else only indices

    Returns:
        If return_scores is False: numpy array of kept indices (sorted by value)
        If return_scores is True: (keep_indices, scores) where scores is a 1D
        numpy array of length len(train_dataset) in original order.
    """
    logger = get_debug_logger(debug=debug)

    # Determine cache paths
    if file_key is not None:
        cache_dir = 'sava_selection_results'
        os.makedirs(cache_dir, exist_ok=True)
        idx_path = os.path.join(cache_dir, f"{file_key}_sorted_indices.npy")
        scores_path = os.path.join(cache_dir, f"{file_key}_scores.npy")

        # Try to load from cache
        if os.path.exists(idx_path) and os.path.exists(scores_path):
            print(f"Loading SAVA sorted indices and scores from cache: {cache_dir}")
            sorted_indices = np.load(idx_path)
            scores = np.load(scores_path)
            if debug:
                logger.debug(f"Loaded sorted_indices shape {sorted_indices.shape}, scores shape {scores.shape}")
                logger.debug(f"First 10 sorted indices: {sorted_indices[:10]}")
                logger.debug(f"First 10 scores (original order): {scores[:10]}")

            # Validate lengths
            if len(sorted_indices) != len(train_dataset):
                raise RuntimeError(f"Cached indices length {len(sorted_indices)} != dataset size {len(train_dataset)}")
            if len(scores) != len(train_dataset):
                raise RuntimeError(f"Cached scores length {len(scores)} != dataset size {len(train_dataset)}")

            num_keep = int(len(train_dataset) * keep_ratio)
            keep_indices = sorted_indices[:num_keep]

            if return_scores:
                return keep_indices, scores
            else:
                return keep_indices

    # --- Compute from scratch ---
    if debug:
        logger.debug("Cache not found or file_key is None. Computing SAVA scores...")

    # Call helper that returns both sorted indices and scores
    sorted_indices, scores = get_sava_sorted_indices(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        batch_size=batch_size,
        num_classes=num_classes,
        resize=resize,
        cache_label_distances=cache_label_distances,
        corrupt_por=corrupt_por,
        debug=debug,
        return_scores=True   # <-- request scores
    )

    # Cache if key provided
    if file_key is not None:
        np.save(idx_path, sorted_indices)
        np.save(scores_path, scores)
        print(f"Cached SAVA sorted indices and scores to {cache_dir}")
        if debug:
            logger.debug(f"Saved sorted_indices shape {sorted_indices.shape}, scores shape {scores.shape}")

    num_keep = int(len(train_dataset) * keep_ratio)
    keep_indices = sorted_indices[:num_keep]

    if return_scores:
        return keep_indices, scores
    else:
        return keep_indices