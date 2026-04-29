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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    # 4. Build the training dataset (plain, no augmentation) based on strategy
    _, val_transform = get_weak_augmentation()

    # --- Branch for DeepSMOTE_Selection ---
    if config.strategy == 'DeepSMOTE_Selection':
        print("Loading DeepSMOTE data (capped) for LAVA scoring...")
        from imbalanceddl.utils.deep_smote_data_loader import load_and_cap_deepsmote, CustomImageDataset
        X_capped, Y_capped = load_and_cap_deepsmote(
            dataset=config.dataset,
            imb_type=config.imb_type,
            imb_factor=config.imb_factor,
            class_caps=None  # default [5000,4000,...]
        )
        # Use validation transform (ToTensor + Normalize) for plain scoring
        train_ds = CustomImageDataset(X_capped, Y_capped, transform=val_transform)

    # --- Branch for RandomOversampling_Selection ---
    elif config.strategy == 'RandomOversampling_Selection':
        print("Loading original imbalanced dataset and applying random oversampling...")
        from imbalanceddl.dataset.imbalance_cifar import IMBALANCECIFAR10
        from imbalanceddl.dataset.imbalance_cifar_noisy import IMBALANCECIFAR10_NOISY
        from imbalanceddl.utils.deep_smote_data_loader import inject_label_noise, CustomImageDataset

        # Load original dataset (no transform, we need raw numpy arrays)
        if config.dataset == 'cifar10':
            base_dataset = IMBALANCECIFAR10(
                root='./data',
                imb_type=config.imb_type,
                imb_factor=config.imb_factor,
                rand_number=config.rand_number,
                train=True,
                download=True,
                transform=None
            )
        elif config.dataset == 'cifar10_noisy':
            base_dataset = IMBALANCECIFAR10_NOISY(
                root='./data',
                imb_type=config.imb_type,
                imb_factor=config.imb_factor,
                rand_number=config.rand_number,
                train=True,
                download=True,
                transform=None,
                noise_ratio=getattr(config, 'noise_ratio', 0.25),
                num_classes=config.num_classes,
                seed=config.rand_number
            )
        else:
            raise NotImplementedError(f"Dataset {config.dataset} not supported for random oversampling")

        X = base_dataset.data
        Y = np.array(base_dataset.targets).astype(int)

        # If the dataset is clean cifar10 but noise_ratio > 0, inject noise manually
        if config.dataset == 'cifar10' and hasattr(config, 'noise_ratio') and config.noise_ratio > 0:
            print(f"Applying {config.noise_ratio*100}% label noise to original dataset")
            Y = inject_label_noise(Y, config.noise_ratio, config.num_classes, seed=config.rand_number)

        # Compute majority class count
        original_counts = np.bincount(Y, minlength=config.num_classes)
        majority_count = max(original_counts)
        print(f"Original class distribution: {dict(enumerate(original_counts))}")
        print(f"Majority class size: {majority_count}")

        # Random oversample each class to majority_count (with replacement)
        oversampled_indices = []
        for c in range(config.num_classes):
            idx = np.where(Y == c)[0]
            if len(idx) == 0:
                continue
            chosen = np.random.choice(idx, size=majority_count, replace=True)
            oversampled_indices.extend(chosen)
        oversampled_indices = np.array(oversampled_indices)
        X_oversampled = X[oversampled_indices]
        Y_oversampled = Y[oversampled_indices]

        print(f"Oversampled dataset size: {len(X_oversampled)} samples (balanced to {majority_count} per class)")

        # Create plain dataset (ToTensor + Normalize)
        train_ds = CustomImageDataset(X_oversampled, Y_oversampled, transform=val_transform)

    # --- Default: plain ImbalancedDataset (no selection, no oversampling) ---
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

    key_gen = LavaCacheKey(config=config, is_deepsmote=is_deepsmote, is_noisy=is_noisy, is_oversampled=is_oversampled)
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