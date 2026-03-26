import torch
from torch.utils.data import Dataset, Subset
import numpy as np

# Updated imports to match your new package structure
from imbalanceddl.strategy.selection_method.lava_selection import get_lava_selection_indices
from imbalanceddl.strategy.selection_method.random_selection import random_selection

class LavaDataset(Dataset):
    def __init__(self, config, base_dataset, ratio, method, device='cuda'):
        """
        Args:
            config: Configuration object.
            base_dataset: The ImbalancedDataset instance.
            ratio: Fraction of data to keep (0.0 to 1.0).
            method: 'lava' or 'random'.
            device: Device to run LAVA computation on.
        """
        self.config = config
        self.base_dataset = base_dataset
        self.ratio = ratio
        self.method = method
        self.device = device

        train_ds, val_ds = self.base_dataset.train_val_sets

        print(f"==> Starting Data Selection via {method}...")
        
        # 1. Get indices to keep
        if method.lower() == 'lava':
            # We pass the underlying training set and labels to LAVA
            indices = get_lava_selection_indices(
                train_ds, 
                keep_ratio=self.ratio, 
                device=self.device
            )
        elif method.lower() == 'random':
            indices = random_selection(
                train_ds, 
                ratio=self.ratio
            )
        else:
            raise ValueError(f"Unknown selection method: {method}")

        # 2. Create the subset
        self.subset = Subset(train_ds, indices)
        
        # 3. Update internal info for the trainer/model
        self.train_dataset = self.subset
        self.val_dataset = val_ds
        self.cls_num_list = self._compute_new_cls_num_list(indices, train_ds)
        
        print(f"==> Selection Complete. New training size: {len(self.subset)}")

    def _compute_new_cls_num_list(self, indices, train_ds):
        """Calculates the new class distribution after selection."""
        all_labels = np.array(train_ds.targets)
        selected_labels = all_labels[indices]
        unique, counts = np.unique(selected_labels, return_counts=True)
        
        # Create a full list including classes that might now have 0 samples
        new_list = [0] * len(self.base_dataset.cfg.cls_num_list)
        for cls, count in zip(unique, counts):
            new_list[int(cls)] = int(count)
        return new_list

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        return self.subset[index]