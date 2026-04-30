#!/usr/bin/env python3
"""
Standalone script to compute LAVA scores for a given config and exit.
Scores are cached for later training runs.
Usage: python compute_lava_scores.py --config /path/to/config.yaml
"""

import sys
from unittest.mock import MagicMock
import logging
import numpy as np
import torch
import os

# Silence torchtext to avoid import issues
def silence_torchtext():
    modules_to_mock = [
        "torchtext", "torchtext.data", "torchtext.data.utils",
        "torchtext.datasets", "torchtext.vocab"
    ]
    for mod in modules_to_mock:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()
silence_torchtext()

from imbalanceddl.utils.utils import fix_all_seed, prepare_store_name, prepare_folders
from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.utils.config import get_args
from imbalanceddl.strategy.selection_method.lava_selection import get_lava_selection_indices
from torchvision import datasets
from imbalanceddl.utils._augmentation import get_weak_augmentation
from imbalanceddl.utils.key_generation import LavaCacheKey
from imbalanceddl.dataset.imbalance_cifar import IMBALANCECIFAR10
from imbalanceddl.utils.deep_smote_data_loader import inject_label_noise, CustomImageDataset
from imbalanceddl.dataset.capped_dataset import CappedDataset
from imbalanceddl.utils.deep_smote_data_loader import load_and_cap_deepsmote

def main():
    # 1. Load Configuration
    config = get_args()

    # 2. Setup basic folders (for logging, but we won't train)
    prepare_store_name(config)
    print(f"=> Store Name = {config.store_name}")
    prepare_folders(config)

    # 3. Seed for reproducibility
    if config.seed is None:
        config.seed = np.random.randint(10000)
    fix_all_seed(config.seed)

    if hasattr(config, 'gpu') and config.gpu is not None and torch.cuda.is_available():
        device = f'cuda:{config.gpu}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    # 4. Build the training dataset (plain, no augmentation) based on strategy
    _, val_transform = get_weak_augmentation()

    if config.strategy == 'DeepSMOTE_Selection':
        print("Loading DeepSMOTE data (capped) for LAVA scoring...")
        X_capped, Y_capped = load_and_cap_deepsmote(
            dataset=config.dataset,
            imb_type=config.imb_type,
            imb_factor=config.imb_factor,
            class_caps=None  # default [5000,4000,...]
        )
        train_ds = CustomImageDataset(X_capped, Y_capped, transform=val_transform)

    elif config.strategy == 'RandomOversampling_Selection':
        print("Loading original imbalanced dataset for random oversampling...")

        base_dataset = IMBALANCECIFAR10(
            root='./data',
            imb_type=config.imb_type,
            imb_factor=config.imb_factor,
            rand_number=config.rand_number,
            train=True,
            download=True,
            transform=None
        )
        X = base_dataset.data
        Y = np.array(base_dataset.targets).astype(int)
        print(f"[DEBUG] Loaded clean dataset: X.shape={X.shape}, Y.shape={Y.shape}")
        print(f"[DEBUG] Original class distribution: {dict(zip(*np.unique(Y, return_counts=True)))}")

        # Check noise_first flag
        noise_first = hasattr(config, 'noise_first') and config.noise_first
        print(f"[DEBUG] noise_first = {noise_first}")

        if noise_first:
            # ------------------------------------------------------------
            # Pipeline A: Noise -> Oversample -> Cap
            # ------------------------------------------------------------
            clean_counts = np.bincount(Y, minlength=config.num_classes)
            orig_majority = max(clean_counts)

            if hasattr(config, 'noise_ratio') and config.noise_ratio > 0:
                print(f"Applying {config.noise_ratio*100}% label noise to original dataset (before oversampling)")
                Y = inject_label_noise(Y, config.noise_ratio, config.num_classes, seed=config.rand_number)
                print(f"[DEBUG] After noise: class distribution: {dict(zip(*np.unique(Y, return_counts=True)))}")

            majority_count = orig_majority
            print(f"Majority class size (original): {majority_count}")

            # Oversample to majority_count
            oversampled_indices = []
            for c in range(config.num_classes):
                idx = np.where(Y == c)[0]
                if len(idx) == 0:
                    continue
                chosen = np.random.choice(idx, size=majority_count, replace=True)
                oversampled_indices.extend(chosen)
            oversampled_indices = np.array(oversampled_indices)
            X_bal = X[oversampled_indices]
            Y_bal = Y[oversampled_indices]
            print(f"Oversampled dataset size: {len(X_bal)} (balanced to {majority_count} per class)")

            # Cap if requested
            if hasattr(config, 'cap_per_class') and config.cap_per_class is not None:
                print(f"Capping dataset to {config.cap_per_class} samples per class")
                temp_dataset = CustomImageDataset(X_bal, Y_bal, transform=None)
                capped_dataset = CappedDataset(temp_dataset, config.cap_per_class, num_classes=config.num_classes)
                subset_indices = capped_dataset.keep_indices
                X_bal = X_bal[subset_indices]
                Y_bal = Y_bal[subset_indices]
                print(f"Capped dataset size: {len(X_bal)} (all classes capped to {config.cap_per_class})")

        else:
            # ------------------------------------------------------------
            # Pipeline B: Oversample -> Cap -> Noise (original order)
            # ------------------------------------------------------------
            # Compute majority count from clean labels
            original_counts = np.bincount(Y, minlength=config.num_classes)
            majority_count = max(original_counts)
            print(f"Majority class size: {majority_count}")

            # Oversample to majority_count
            oversampled_indices = []
            for c in range(config.num_classes):
                idx = np.where(Y == c)[0]
                if len(idx) == 0:
                    continue
                chosen = np.random.choice(idx, size=majority_count, replace=True)
                oversampled_indices.extend(chosen)
            oversampled_indices = np.array(oversampled_indices)
            X_bal = X[oversampled_indices]
            Y_bal = Y[oversampled_indices]
            print(f"Oversampled dataset size: {len(X_bal)} (balanced to {majority_count} per class)")

            # Cap if requested
            if hasattr(config, 'cap_per_class') and config.cap_per_class is not None:
                print(f"Capping dataset to {config.cap_per_class} samples per class")
                temp_dataset = CustomImageDataset(X_bal, Y_bal, transform=None)
                capped_dataset = CappedDataset(temp_dataset, config.cap_per_class, num_classes=config.num_classes)
                subset_indices = capped_dataset.keep_indices
                X_bal = X_bal[subset_indices]
                Y_bal = Y_bal[subset_indices]
                print(f"Capped dataset size: {len(X_bal)} (all classes capped to {config.cap_per_class})")

            # Inject noise after capping
            if hasattr(config, 'noise_ratio') and config.noise_ratio > 0:
                print(f"Applying {config.noise_ratio*100}% label noise to capped balanced dataset")
                Y_bal = inject_label_noise(Y_bal, config.noise_ratio, config.num_classes, seed=config.rand_number)
                print(f"[DEBUG] After noise: class distribution: {dict(zip(*np.unique(Y_bal, return_counts=True)))}")

        # Create plain dataset (ToTensor + Normalize)
        train_ds = CustomImageDataset(X_bal, Y_bal, transform=val_transform)

    else:
        print("Creating plain dataset (no augmentation) for LAVA scoring...")
        plain_dataset = ImbalancedDataset(config, dataset_name=config.dataset, augmentation='none')
        train_ds, _ = plain_dataset.train_val_sets

    # 5. Build validation set (balanced test set)
    if config.dataset == 'cifar10' or config.dataset == 'cifar10_noisy':
        val_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    elif config.dataset == 'cifar100':
        val_ds = datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)
    else:
        raise NotImplementedError(f"Validation set for {config.dataset} not implemented")

    # 6. Generate cache key using LavaCacheKey
    is_deepsmote = (config.strategy == 'DeepSMOTE_Selection')
    is_oversampled = (config.strategy == 'RandomOversampling_Selection')
    is_noisy = (hasattr(config, 'noise_ratio') and config.noise_ratio > 0)
    is_noise_first = (hasattr(config, 'noise_first') and config.noise_first)

    key_gen = LavaCacheKey(config=config, is_deepsmote=is_deepsmote, is_noisy=is_noisy,
                           is_oversampled=is_oversampled, is_noise_first=is_noise_first)
    file_key = key_gen.generate()
    print(f"Computing LAVA scores with file_key = {file_key}")

    # 7. Compute LAVA scores (cached automatically)
    indices = get_lava_selection_indices(
        train_dataset=train_ds,
        val_dataset=val_ds,
        keep_ratio=config.selection_ratio,
        device=device,
        file_key=file_key
    )
    print(f"LAVA scores computed and cached. Selected {len(indices)} samples (keep_ratio={config.selection_ratio})")
    print("Exiting without training.")

if __name__ == "__main__":
    main()