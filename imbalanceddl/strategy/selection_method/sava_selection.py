import numpy as np
import os
from imbalanceddl.utils.sava_helpers import get_sava_sorted_indices

def get_sava_selection_indices(train_dataset, val_dataset, keep_ratio,
                               device='cuda', file_key=None, batch_size=1024,
                               num_classes=10, feat_repr=True, parallel=False,
                               cuda_num=0, n_gpu=1, resize=32,
                               cache_label_distances=True, model_path=None,
                               corrupt_por=0.01):   # added parameter
    """Returns indices to keep (lowest scores = most valuable)."""
    if file_key is not None:
        cache_dir = 'sava_selection_results'
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{file_key}_sorted_indices.npy")
        if os.path.exists(cache_path):
            print(f"Loading SAVA sorted indices from cache: {cache_path}")
            sorted_indices = np.load(cache_path)
        else:
            sorted_indices = get_sava_sorted_indices(
                train_dataset, val_dataset, device=device, batch_size=batch_size,
                num_classes=num_classes, feat_repr=feat_repr,
                parallel=parallel, cuda_num=cuda_num, n_gpu=n_gpu, resize=resize,
                cache_label_distances=cache_label_distances, model_path=model_path,
                corrupt_por=corrupt_por      # pass along
            )
            np.save(cache_path, sorted_indices)
            print(f"Saved SAVA sorted indices to {cache_path}")
    else:
        sorted_indices = get_sava_sorted_indices(
            train_dataset, val_dataset, device=device, batch_size=batch_size,
            num_classes=num_classes, feat_repr=feat_repr,
            parallel=parallel, cuda_num=cuda_num, n_gpu=n_gpu, resize=resize,
            cache_label_distances=cache_label_distances, model_path=model_path,
            corrupt_por=corrupt_por
        )

    num_keep = int(len(train_dataset) * keep_ratio)
    keep_indices = sorted_indices[:num_keep]
    print(f"SAVA selection: keeping {num_keep} out of {len(train_dataset)} samples.")
    return keep_indices