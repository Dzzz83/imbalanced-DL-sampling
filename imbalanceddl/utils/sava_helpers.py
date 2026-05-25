import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

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

# ----------------------------------------------------------------------
# Main function: compute SAVA scores and return sorted training indices
# ----------------------------------------------------------------------
def get_sava_sorted_indices(train_dataset, val_dataset, device='cuda',
                            batch_size=1024, num_classes=10, resize=32,
                            cache_label_distances=True, corrupt_por=0.0):
    """
    Compute SAVA scores (sorted training indices by increasing value) using raw pixels.
    
    Args:
        train_dataset: Training dataset (will be wrapped in DataLoader)
        val_dataset: Validation dataset
        device: 'cuda' or 'cpu'
        batch_size: Batch size for OT computation
        num_classes: Number of classes (for SAVA's internal use)
        resize: Resize images to this size (default 32)
        cache_label_distances: Cache label-to-label OT distances
        corrupt_por: Fraction of training data to mark as corrupted (shuffled); default 0.0
    Returns:
        sorted_indices: 1D numpy array of training indices sorted from most valuable to least
    """
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
    
    # Use identity feature extractor (raw pixels)
    model = IdentityExtractor().to(device)
    model.eval()
    print("Using raw pixels (feat_repr=False).")
    
    # Simulate corruption (only if corrupt_por > 0)
    n_corrupt = int(training_size * corrupt_por)
    if n_corrupt > 0:
        shuffle_ind = list(range(n_corrupt))
        print(f"Using shuffle_ind with {len(shuffle_ind)} samples (corrupt_por={corrupt_por})")
    else:
        shuffle_ind = []
        print("shuffle_ind is empty (corrupt_por=0).")
    
    # Run SAVA hierarchical OT experiment
    print(f"Running SAVA with: batch_size={batch_size}, device={device}")
    result = hierarchical_ot_experiment(
        feature_extractor=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_size=training_size,
        batch_size=batch_size,
        shuffle_ind=shuffle_ind,
        resize=resize,
        portion=0.0,                      # Important: avoids division by zero
        device=device,
        cache_label_distances=cache_label_distances,
        visualise_hot=False,
        tag="",
        num_classes=num_classes,
        parallel=False,                   # Not used with raw pixels
        cuda_num=0,
        n_gpu=1,
    )
    
    # Extract sorted indices from the result
    if isinstance(result, tuple):
        sorted_indices = result[0]   # assume first element is the sorted indices
    else:
        sorted_indices = result
    
    # Convert to flat int64 numpy array
    if isinstance(sorted_indices, list):
        # Handle case where each element is a list/tuple of one element
        if len(sorted_indices) > 0 and hasattr(sorted_indices[0], '__len__') and len(sorted_indices[0]) == 1:
            sorted_indices = np.array([int(x[0]) for x in sorted_indices], dtype=np.int64)
        else:
            sorted_indices = np.array(sorted_indices, dtype=np.int64)
    else:
        sorted_indices = np.asarray(sorted_indices).ravel().astype(np.int64)
    
    # Final sanity: the number of indices should equal training_size
    if len(sorted_indices) != training_size:
        raise RuntimeError(f"Expected {training_size} indices but got {len(sorted_indices)}")
    
    print(f"Sorted indices shape: {sorted_indices.shape}, dtype: {sorted_indices.dtype}")
    return sorted_indices