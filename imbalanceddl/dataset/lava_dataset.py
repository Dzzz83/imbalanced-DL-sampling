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
            method: 'lava', 'random', or 'none' / None.
            device: Device to run LAVA computation on.
        """
        self.config = config
        self.base_dataset = base_dataset
        self.ratio = ratio
        self.method = method
        self.device = device

        train_ds, val_ds = self.base_dataset.train_val_sets
        
        # Guard against NoneType for printing
        print(f"==> Starting Data Selection via {method}...")

        # 1. Get indices to keep
        method_str = str(method).lower()

        if method_str == 'lava':
            indices = get_lava_selection_indices(
                train_ds, 
                val_ds,
                keep_ratio=self.ratio, 
                device=self.device
            )
        elif method_str == 'random':
            indices = random_selection(
                train_ds, 
                ratio=self.ratio
            )
        elif method_str == 'none':
            indices = list(range(len(train_ds)))
            print("==> No selection method specified. Using full dataset.")
        else:
            raise ValueError(f"Unknown selection method: {method}")

        self.indices = indices
        self.subset = Subset(train_ds, indices)

        if hasattr(train_ds, 'targets'):
            self.targets = np.array(train_ds.targets)[indices].tolist()
        elif hasattr(train_ds, 'labels'):
            self.targets = np.array(train_ds.labels)[indices].tolist()
        else:
            self.targets = [train_ds[i][1] for i in indices]

        self.train_dataset = self 
        self.val_dataset = val_ds
        self.cls_num_list = self._compute_new_cls_num_list(indices, train_ds)
        
        print(f"==> Selection Complete. New training size: {len(self.subset)}")

    @property
    def train_val_sets(self):
        """Exposes the train and validation sets to the Trainer"""
        return self.train_dataset, self.val_dataset

    def _compute_new_cls_num_list(self, indices, train_ds):
        """Calculates the new class distribution after selection."""
        unique, counts = np.unique(self.targets, return_counts=True)        
        # Create a full list including classes that might now have 0 samples
        new_list = [0] * len(self.base_dataset.cfg.cls_num_list)
        for cls, count in zip(unique, counts):
            new_list[int(cls)] = int(count)
        return new_list

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        return self.subset[index]
        
    def get_cls_num_list(self):
        """Getter for the trainer to access the new class distribution."""
        return self.cls_num_list