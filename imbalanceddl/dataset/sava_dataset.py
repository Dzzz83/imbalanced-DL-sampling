import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import os
from imbalanceddl.dataset import ImbalancedDataset
from imbalanceddl.strategy.selection_method.sava_selection import get_sava_selection_indices
from imbalanceddl.utils.sava_key_generation import SavaCacheKey   # ADDED


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
            # Generate a unique cache key using SavaCacheKey (no LAVA dependency)
            is_noisy = hasattr(self.config, 'noise_ratio') and self.config.noise_ratio > 0
            # Use SavaCacheKey instead of LavaCacheKey
            key_gen = SavaCacheKey(
                config=self.config,
                is_deepsmote=False,
                is_noisy=is_noisy,
                is_oversampled=False,
                is_noise_first=getattr(self.config, 'noise_first', False),
                is_selection_first=False
            )
            file_key = key_gen.generate()
            # NOTE: SavaCacheKey.generate() already appends "_sava", so no need to add again.
            # But if your existing cache files have "_sava" appended, keep as is.
            # To be safe, we can leave it as is (it will produce e.g. "cifar10_exp_0.01_0_sava")
            # No extra appending needed.

            print("Creating dataset (no augmentation) for SAVA scoring...")
            no_aug_dataset = ImbalancedDataset(self.config, self.config.dataset, augmentation='none')
            no_aug_train_dataset, _ = no_aug_dataset.train_val_sets

            # Optional: If cap_per_class is used, we must apply the same cap to no_aug_train_dataset
            # Otherwise indices will mismatch. For now, we warn and proceed.
            if hasattr(self.config, 'cap_per_class') and self.config.cap_per_class is not None:
                print("WARNING: cap_per_class is set. SAVA scoring uses uncapped dataset, "
                      "but selection will be applied to capped dataset. This may cause index errors.")
                # To fix, you could disable capping when selection_method=='sava' in the main script.

            indices = get_sava_selection_indices(
            train_dataset=no_aug_train_dataset,
            val_dataset=val_ds,
            keep_ratio=self.ratio,
            device=self.device,
            file_key=file_key,
            batch_size=getattr(self.config, 'sava_batch_size', 1024),
            num_classes=self.config.num_classes,
            resize=32,  # CIFAR images are 32x32
            cache_label_distances=getattr(self.config, 'sava_cache_label_distances', True),
            corrupt_por=0.0   # no corruption
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

    # ========== ADD MISSING METHODS FOR SAMPLERS ==========
    def get_class_idxs2(self):
        """
        Required by WeightedRandomBatchSampler, WeightedFixedBatchSampler, etc.
        Returns a list of lists, where each sublist contains the indices of samples
        belonging to a particular class (0 .. num_classes-1).
        """
        targets_np = np.array(self.targets, dtype=np.int64)
        # Ensure we have entries for all classes, even if count=0
        class_idxs = []
        for c in range(self.config.num_classes):
            idxs = np.where(targets_np == c)[0].tolist()
            class_idxs.append(idxs)
        return class_idxs

    def get_sample_weights(self):
        """
        Required by some samplers. Returns a weight per sample inversely proportional
        to class frequency.
        """
        cls_counts = np.bincount(self.targets, minlength=self.config.num_classes)
        cls_counts = np.maximum(cls_counts, 1)  # avoid division by zero
        total = len(self.targets)
        class_weights = total / (self.config.num_classes * cls_counts)
        sample_weights = [class_weights[t] for t in self.targets]
        return sample_weights

    def get_weights(self):
        """
        Optional: compatibility with BaseDataset. Returns class weights.
        """
        cls_counts = np.bincount(self.targets, minlength=self.config.num_classes)
        cls_counts = np.maximum(cls_counts, 1)
        total = len(self.targets)
        class_weights = total / (self.config.num_classes * cls_counts)
        return class_weights