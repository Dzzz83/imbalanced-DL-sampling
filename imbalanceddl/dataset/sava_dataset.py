import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import os
from imbalanceddl.dataset import ImbalancedDataset
from imbalanceddl.strategy.selection_method.sava_selection import get_sava_selection_indices

class SavaDataset(Dataset):
    def __init__(self, config, base_dataset, ratio, method, device='cuda'):
        """
        Args:
            config: Configuration object.
            base_dataset: The ImbalancedDataset instance.
            ratio: Fraction of data to keep (0.0 to 1.0).
            method: Should be 'sava' (or 'random' fallback).
            device: Device to run SAVA computation on.
        """
        self.config = config
        self.base_dataset = base_dataset
        self.ratio = ratio
        self.method = method
        self.device = device

        train_ds, val_ds = self.base_dataset.train_val_sets
        
        print(f"==> Starting Data Selection via SAVA...")

        method_str = str(method).lower()

        if method_str == 'sava':
            # Generate a unique cache key (same as LAVA, but we'll store in sava_selection_results)
            is_noisy = hasattr(self.config, 'noise_ratio') and self.config.noise_ratio > 0
            key_gen = LavaCacheKey(config=self.config, is_deepsmote=False, is_noisy=is_noisy)
            file_key = key_gen.generate()
            # For SAVA we might want to append '_sava' to avoid mixing with LAVA caches
            file_key = f"{file_key}_sava"

            print("Creating dataset (no augmentation) for SAVA scoring...")
            no_aug_dataset = ImbalancedDataset(self.config, self.config.dataset, augmentation='none')
            no_aug_train_dataset, _ = no_aug_dataset.train_val_sets

            indices = get_sava_selection_indices(
                train_dataset=no_aug_train_dataset,
                val_dataset=val_ds,
                keep_ratio=self.ratio,
                device=self.device,
                file_key=file_key,
                batch_size=getattr(self.config, 'sava_batch_size', 1024),
                num_classes=self.config.num_classes,
                feat_repr=getattr(self.config, 'sava_feat_repr', False),
                parallel=getattr(self.config, 'sava_parallel', False),
                cuda_num=getattr(self.config, 'sava_cuda_num', 0),
                n_gpu=getattr(self.config, 'sava_n_gpu', 1)
            )

            if hasattr(train_ds, 'targets'):
                selected_targets = np.array(train_ds.targets)[indices]
                unique, counts = np.unique(selected_targets, return_counts=True)
                print(f"[SavaDataset] Selected class distribution: {dict(zip(unique, counts))}")

        elif method_str == 'random':
            from imbalanceddl.strategy.selection_method.random_selection import random_selection
            indices = random_selection(train_ds, keep_ratio=self.ratio)
            if hasattr(train_ds, 'targets'):
                selected_targets = np.array(train_ds.targets)[indices]
                unique, counts = np.unique(selected_targets, return_counts=True)
                print(f"[SavaDataset] Random selection class distribution: {dict(zip(unique, counts))}")
        elif method_str == 'none':
            indices = list(range(len(train_ds)))
            print("==> No selection method specified. Using full dataset.")
        else:
            raise ValueError(f"Unknown selection method for SavaDataset: {method}")

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

        print(f"[SavaDataset] Final cls_num_list: {self.cls_num_list}")
        print(f"==> Selection Complete. New training size: {len(self.subset)}")

    @property
    def train_val_sets(self):
        return self.train_dataset, self.val_dataset

    def _compute_new_cls_num_list(self, indices, train_ds):
        unique, counts = np.unique(self.targets, return_counts=True)
        new_list = [0] * len(self.base_dataset.cfg.cls_num_list)
        for cls, count in zip(unique, counts):
            new_list[int(cls)] = int(count)
        return new_list

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        return self.subset[index]

    def get_cls_num_list(self):
        return self.cls_num_list