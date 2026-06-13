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
                            debug=False):
    """
    Compute SAVA scores (sorted training indices by increasing value) using raw pixels.
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
        portion=0.0,
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
    if isinstance(result, tuple):
        sorted_indices = result[0]
        # Determine scores position based on tuple length
        if len(result) == 2:
            scores = result[1]
        elif len(result) == 3:
            scores = result[2]  # scores are third element when trained_with_flag is present
        else:
            scores = None
        if scores is not None and debug:
            try:
                if not isinstance(scores, np.ndarray):
                    scores = np.array(scores)
                if scores.ndim > 1:
                    scores = scores.ravel()
                # scores is aligned with original training order (index 0..training_size-1)
                # Now reorder scores according to sorted_indices to see the ranking
                sorted_scores = scores[sorted_indices]
                if sorted_scores.size > 0:
                    logger.debug(f"Sorted scores (most valuable first) - min: {np.min(sorted_scores):.6f}, max: {np.max(sorted_scores):.6f}, mean: {np.mean(sorted_scores):.6f}, std: {np.std(sorted_scores):.6f}")
                    logger.debug(f"First 10 scores (most valuable): {sorted_scores[:10]}")
                    logger.debug(f"Last 10 scores (least valuable): {sorted_scores[-10:]}")
                else:
                    logger.debug("Scores array is empty.")
            except Exception as e:
                logger.debug(f"Could not process scores: {e}. Raw type: {type(scores)}")
        elif debug:
            logger.debug("No scores in result tuple.")
    else:
        sorted_indices = result
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
    
    if debug:
        logger.debug(f"Sorted indices shape: {sorted_indices.shape}, dtype: {sorted_indices.dtype}")
        logger.debug(f"First 10 training indices (most valuable): {sorted_indices[:10]}")
        logger.debug(f"Last 10 training indices (least valuable): {sorted_indices[-10:]}")
    
    return sorted_indices