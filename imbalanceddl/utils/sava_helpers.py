# imbalanceddl/utils/sava_helpers.py

import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from imbalanceddl.utils.debug_logger import get_debug_logger

# ----------------------------------------------------------------------
# Path setup for SAVA modules (must be done before importing from api)
# ----------------------------------------------------------------------
def _setup_sava_paths():
    current_file = os.path.abspath(__file__)
    # Go up from imbalanceddl/utils/sava_helpers.py to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    sava_root = os.path.join(project_root, 'sava')
    
    if sava_root not in sys.path:
        sys.path.insert(0, sava_root)
    
    otdd_path = os.path.join(sava_root, 'otdd')
    if otdd_path not in sys.path:
        sys.path.insert(0, otdd_path)
    
    # Remove any cached otdd module to force reload from correct path
    modules_to_remove = [m for m in sys.modules if m.startswith('otdd')]
    for m in modules_to_remove:
        del sys.modules[m]
    
    models_path = os.path.join(sava_root, 'models')
    if models_path not in sys.path:
        sys.path.insert(0, models_path)
    
    return sava_root, project_root

SAVA_ROOT, PROJECT_ROOT = _setup_sava_paths()

# Now safe to import SAVA modules
from api import hierarchical_ot_experiment

# ----------------------------------------------------------------------
# Identity feature extractor (raw pixels)
# ----------------------------------------------------------------------
class IdentityExtractor(torch.nn.Module):
    def forward(self, x):
        return x

def get_sava_sorted_indices(train_dataset, val_dataset, device='cuda',
                            batch_size=1024, num_classes=10, resize=32,
                            cache_label_distances=True, corrupt_por=0.0,
                            debug=False, return_scores=False):
    """
    Compute SAVA scores (hierarchical OT) using raw pixels.

    Args:
        train_dataset: torch Dataset for training
        val_dataset: torch Dataset for validation
        device: 'cuda' or 'cpu'
        batch_size: batch size for OT
        num_classes: number of classes
        resize: image resize dimension
        cache_label_distances: cache label-to-label distances
        corrupt_por: corruption portion (for SAVA paper experiments)
        debug: enable debug logging
        return_scores: if True, return (sorted_indices, scores) else only sorted_indices

    Returns:
        sorted_indices: numpy array of training indices sorted by increasing SAVA score
        scores (optional): numpy array of raw SAVA scores in original training order
    """
    logger = get_debug_logger(debug=debug)
    
    # Sanity checks
    if not isinstance(train_dataset, torch.utils.data.Dataset):
        raise TypeError("train_dataset must be a torch Dataset")
    if not isinstance(val_dataset, torch.utils.data.Dataset):
        raise TypeError("val_dataset must be a torch Dataset")
    
    # Create data loaders (no shuffling)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    training_size = len(train_dataset)
    
    if debug:
        logger.debug(f"train_dataset size: {training_size}, val_dataset size: {len(val_dataset)}")
        logger.debug(f"batch_size: {batch_size}, device: {device}")
    
    # Use identity feature extractor (raw pixels)
    model = IdentityExtractor().to(device)
    model.eval()
    print("Using raw pixels (feat_repr=False).")
    
    # Simulate corruption (only if corrupt_por > 0)
    n_corrupt = int(training_size * corrupt_por)
    if n_corrupt > 0:
        shuffle_ind = list(range(n_corrupt))
        print(f"Using shuffle_ind with {len(shuffle_ind)} samples (corrupt_por={corrupt_por})")
        if debug:
            logger.debug(f"First 10 shuffle_ind: {shuffle_ind[:10]}")
    else:
        shuffle_ind = []
        print("shuffle_ind is empty (corrupt_por=0).")
    
    # Run SAVA hierarchical OT experiment
    print(f"Running SAVA with: batch_size={batch_size}, device={device}")
    if debug:
        logger.debug("Calling hierarchical_ot_experiment...")
    result = hierarchical_ot_experiment(
        feature_extractor=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_size=training_size,
        batch_size=batch_size,
        shuffle_ind=shuffle_ind,
        resize=resize,
        portion=corrupt_por,
        device=device,
        cache_label_distances=cache_label_distances,
        visualise_hot=False,
        tag="",
        feat_repr=False,
        num_classes=num_classes,
        parallel=False,
        cuda_num=0,
        n_gpu=1,
    )
    if debug:
        logger.debug("hierarchical_ot_experiment returned")

    # ---------- Extract sorted indices and scores ----------
    # The result can be:
    #   - (sorted_indices, scores) when portion=0 and shuffle_ind empty
    #   - (sorted_indices, trained_with_flag, scores) when portion>0
    if isinstance(result, tuple):
        if len(result) == 3:
            sorted_indices, trained_with_flag, scores = result
        elif len(result) == 2:
            sorted_indices, scores = result
        else:
            raise RuntimeError(f"Unexpected result length: {len(result)}")
    else:
        sorted_indices = result
        scores = None
        if debug:
            logger.debug("Result is not a tuple; assuming only indices.")

    # Convert to flat int64 numpy array
    if isinstance(sorted_indices, list):
        if len(sorted_indices) > 0 and hasattr(sorted_indices[0], '__len__') and len(sorted_indices[0]) == 1:
            sorted_indices = np.array([int(x[0]) for x in sorted_indices], dtype=np.int64)
        else:
            sorted_indices = np.array(sorted_indices, dtype=np.int64)
    else:
        sorted_indices = np.asarray(sorted_indices).ravel().astype(np.int64)

    if len(sorted_indices) != training_size:
        raise RuntimeError(f"Expected {training_size} indices but got {len(sorted_indices)}")

    if scores is not None:
        # Ensure scores is a 1D numpy array in original order (not sorted)
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)
        if scores.ndim > 1:
            scores = scores.ravel()
        if len(scores) != training_size:
            raise RuntimeError(f"Score length {len(scores)} != training size {training_size}")
        # scores are already in original order (they are values for each training sample)
        if debug:
            logger.debug(f"Scores stats: min={np.min(scores):.6f}, max={np.max(scores):.6f}, mean={np.mean(scores):.6f}")
            logger.debug(f"First 10 scores (original order): {scores[:10]}")
            # Show scores in sorted order (most valuable first)
            sorted_scores = scores[sorted_indices]
            if len(sorted_scores) > 0:
                logger.debug(f"Sorted scores (most valuable first) - first 10: {sorted_scores[:10]}")
                logger.debug(f"Sorted scores - last 10: {sorted_scores[-10:]}")
    else:
        # Fallback: create dummy scores (should not happen with current SAVA API)
        scores = np.zeros(training_size, dtype=np.float64)
        if debug:
            logger.debug("No scores returned; using zeros (this may indicate an issue).")

    if return_scores:
        return sorted_indices, scores
    else:
        return sorted_indices