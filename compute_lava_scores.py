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
from abc import ABC, abstractmethod

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
from imbalanceddl.utils.deep_smote_data_loader import (
    inject_label_noise,
    CustomImageDataset,
    load_and_cap_deepsmote,
    load_deepsmote_raw
)
from imbalanceddl.dataset.capped_dataset import CappedDataset

# ==============================
# Abstract Builder
# ==============================
class DatasetBuilder(ABC):
    @abstractmethod
    def build_plain_dataset(self, config, val_transform):
        """Returns a torch Dataset (plain, no augmentation) and a dict of key flags."""
        pass

# ==============================
# Concrete Builders
# ==============================
class DeepSMOTEBuilder(DatasetBuilder):
    def build_plain_dataset(self, config, val_transform):
        noise_first = hasattr(config, 'noise_first') and config.noise_first
        print(f"[DEBUG] DeepSMOTEBuilder: noise_first = {noise_first}")
        if noise_first:
            X, Y = load_deepsmote_raw(config.dataset, config.imb_type, config.imb_factor)
            if hasattr(config, 'noise_ratio') and config.noise_ratio > 0:
                Y = inject_label_noise(Y, config.noise_ratio, config.num_classes, seed=config.rand_number)
            if hasattr(config, 'cap_per_class') and config.cap_per_class is not None:
                temp = CustomImageDataset(X, Y, transform=None)
                capped = CappedDataset(temp, config.cap_per_class, num_classes=config.num_classes)
                X = X[capped.keep_indices]
                Y = Y[capped.keep_indices]
        else:
            X, Y = load_and_cap_deepsmote(config.dataset, config.imb_type, config.imb_factor, class_caps=None)
            if hasattr(config, 'noise_ratio') and config.noise_ratio > 0:
                Y = inject_label_noise(Y, config.noise_ratio, config.num_classes, seed=config.rand_number)
        train_ds = CustomImageDataset(X, Y, transform=val_transform)
        flags = {
            'is_deepsmote': True,
            'is_oversampled': False,
            'is_noisy': hasattr(config, 'noise_ratio') and config.noise_ratio > 0,
            'is_noise_first': noise_first,
            'is_selection_first': False
        }
        return train_ds, flags

class RandomOversamplingBuilder(DatasetBuilder):
    def build_plain_dataset(self, config, val_transform):
        base = IMBALANCECIFAR10(root='./data', imb_type=config.imb_type, imb_factor=config.imb_factor,
                                rand_number=config.rand_number, train=True, download=True, transform=None)
        X = base.data
        Y = np.array(base.targets).astype(int)
        noise_first = hasattr(config, 'noise_first') and config.noise_first
        print(f"[DEBUG] RandomOversamplingBuilder: noise_first = {noise_first}")

        # Store original majority count (before any noise)
        orig_counts = np.bincount(Y, minlength=config.num_classes)
        orig_majority = max(orig_counts)

        if noise_first:
            # Noise FIRST, then oversample to original majority
            if hasattr(config, 'noise_ratio') and config.noise_ratio > 0:
                Y = inject_label_noise(Y, config.noise_ratio, config.num_classes, seed=config.rand_number)
            majority_count = orig_majority
        else:
            # Oversample to original majority FIRST, then noise
            majority_count = orig_majority
            if hasattr(config, 'noise_ratio') and config.noise_ratio > 0:
                Y = inject_label_noise(Y, config.noise_ratio, config.num_classes, seed=config.rand_number)

        # Oversample to majority_count
        oversampled = []
        for c in range(config.num_classes):
            idx = np.where(Y == c)[0]
            if len(idx) == 0:
                continue
            chosen = np.random.choice(idx, size=majority_count, replace=True)
            oversampled.extend(chosen)
        X = X[oversampled]
        Y = Y[oversampled]

        # Cap if requested
        if hasattr(config, 'cap_per_class') and config.cap_per_class is not None:
            temp = CustomImageDataset(X, Y, transform=None)
            capped = CappedDataset(temp, config.cap_per_class, num_classes=config.num_classes)
            X = X[capped.keep_indices]
            Y = Y[capped.keep_indices]

        train_ds = CustomImageDataset(X, Y, transform=val_transform)
        flags = {
            'is_deepsmote': False,
            'is_oversampled': True,
            'is_noisy': hasattr(config, 'noise_ratio') and config.noise_ratio > 0,
            'is_noise_first': noise_first,
            'is_selection_first': False
        }
        return train_ds, flags

class SelectionRandomOversamplingBuilder(DatasetBuilder):
    def build_plain_dataset(self, config, val_transform):
        base = IMBALANCECIFAR10(root='./data', imb_type=config.imb_type, imb_factor=config.imb_factor,
                                rand_number=config.rand_number, train=True, download=True, transform=None)
        X = base.data
        Y = np.array(base.targets).astype(int)
        if hasattr(config, 'noise_ratio') and config.noise_ratio > 0:
            Y = inject_label_noise(Y, config.noise_ratio, config.num_classes, seed=config.rand_number)
        train_ds = CustomImageDataset(X, Y, transform=val_transform)
        flags = {
            'is_deepsmote': False,
            'is_oversampled': False,
            'is_noisy': hasattr(config, 'noise_ratio') and config.noise_ratio > 0,
            'is_noise_first': False,
            'is_selection_first': True
        }
        return train_ds, flags

class DefaultBuilder(DatasetBuilder):
    def build_plain_dataset(self, config, val_transform):
        plain = ImbalancedDataset(config, dataset_name=config.dataset, augmentation='none')
        train_ds, _ = plain.train_val_sets
        flags = {
            'is_deepsmote': False,
            'is_oversampled': False,
            'is_noisy': False,
            'is_noise_first': False,
            'is_selection_first': False
        }
        return train_ds, flags

# ==============================
# Main
# ==============================
def main():
    config = get_args()
    print(f"[DEBUG] Config loaded. Strategy: {config.strategy}")

    prepare_store_name(config)
    print(f"=> Store Name = {config.store_name}")
    prepare_folders(config)

    if config.seed is None:
        config.seed = np.random.randint(10000)
    fix_all_seed(config.seed)

    if hasattr(config, 'gpu') and config.gpu is not None and torch.cuda.is_available():
        device = f'cuda:{config.gpu}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    print(f"[DEBUG] Device set to: {device}")

    _, val_transform = get_weak_augmentation()

    # Factory mapping
    builders = {
        'DeepSMOTE_Selection': DeepSMOTEBuilder,
        'RandomOversampling_Selection': RandomOversamplingBuilder,
        'Selection_RandomOversampling': SelectionRandomOversamplingBuilder,
    }
    builder_class = builders.get(config.strategy, DefaultBuilder)
    builder = builder_class()
    train_ds, flags = builder.build_plain_dataset(config, val_transform)

    # Build validation set
    if config.dataset in ('cifar10', 'cifar10_noisy'):
        val_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    elif config.dataset == 'cifar100':
        val_ds = datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)
    else:
        raise NotImplementedError(f"Validation set for {config.dataset} not implemented")
    print(f"[DEBUG] Validation set size: {len(val_ds)}")

    # Generate cache key
    key_gen = LavaCacheKey(config=config, **flags)
    file_key = key_gen.generate()
    print(f"[DEBUG] Cache key: {file_key}")

    # Compute LAVA scores
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