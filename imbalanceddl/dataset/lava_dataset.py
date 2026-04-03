import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import os
from imbalanceddl.dataset import ImbalancedDataset 
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
        
        print(f"==> Starting Data Selection via {method}...")

        # 1. Get indices to keep
        method_str = str(method).lower()

        if method_str == 'lava':
            file_key = f"{self.config.dataset}_{self.config.imb_type}_{self.config.imb_factor}_{self.config.rand_number}"
            print("Creating dataset (no augmentation) for LAVA scoring...")
            no_aug_dataset = ImbalancedDataset(self.config, self.config.dataset, augmentation='none')
            no_aug_train_dataset, _ = no_aug_dataset.train_val_sets

            if (len(no_aug_train_dataset) > 30000):
                the_device='cpu'
                print("Large dataset, using CPU for OT computation")
            else:
                the_device=self.device
            
            indices = get_lava_selection_indices(
                no_aug_train_dataset, 
                val_ds,
                keep_ratio=self.ratio, 
                device=the_device,
                file_key=file_key
            )
        elif method_str == 'random':
            indices = random_selection(
                train_ds, 
                keep_ratio=self.ratio, 
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
        self.config.cls_num_list = self.cls_num_list      

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