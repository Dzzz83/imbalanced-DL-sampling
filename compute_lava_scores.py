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

    # 4. Build the dataset WITHOUT AUGMENTATION (for scoring)
    #    We need the plain training set and the validation set.
    #    We use the same ImbalancedDataset with augmentation='none'.
    print(f"Creating plain dataset (no augmentation) for LAVA scoring...")
    plain_dataset = ImbalancedDataset(config, dataset_name=config.dataset, augmentation='none')
    train_ds, _ = plain_dataset.train_val_sets   # training set (plain)

    # 5. Build validation set (balanced test set) – same as in ImbalancedDataset
    _, val_transform = get_weak_augmentation()
    if config.dataset == 'cifar10' or config.dataset == 'cifar10_noisy':
        val_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    elif config.dataset == 'cifar100':
        val_ds = datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)
    else:
        raise NotImplementedError(f"Validation set for {config.dataset} not implemented")

    # 6. Compute LAVA scores (cached automatically)
    file_key = f"{config.dataset}_{config.imb_type}_{config.imb_factor}_{config.rand_number}"
    print(f"Computing LAVA scores with file_key = {file_key}")
    indices = get_lava_selection_indices(
        train_dataset=train_ds,
        val_dataset=val_ds,
        keep_ratio=config.selection_ratio,   # still needed for the function, but scores are saved regardless
        device=device,
        file_key=file_key
    )
    print(f"LAVA scores computed and cached. Selected {len(indices)} samples (keep_ratio={config.selection_ratio})")
    print("Exiting without training.")

if __name__ == "__main__":
    main()