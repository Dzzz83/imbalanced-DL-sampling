#!/usr/bin/env python3
import sys
import os
import datetime
import numpy as np
import torch
from torch.utils.data import Subset, Dataset
from unittest.mock import MagicMock
from torchvision import datasets

# Disable wandb logging
import wandb
wandb.init = lambda *args, **kwargs: None
wandb.log = lambda *args, **kwargs: None

class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

SAVA_ROOT = '/mnt/hdd2/phatht/phat/imbalanced-DL-sampling/sava'
if SAVA_ROOT not in sys.path:
    sys.path.insert(0, SAVA_ROOT)
otdd_path = os.path.join(SAVA_ROOT, 'otdd')
if otdd_path not in sys.path:
    sys.path.insert(0, otdd_path)

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
from imbalanceddl.utils.config import get_args
from imbalanceddl.utils._augmentation import get_weak_augmentation
from imbalanceddl.utils.sava_key_generation import SavaCacheKey
from imbalanceddl.strategy.selection_method.sava_selection import get_sava_selection_indices
from imbalanceddl.utils.deep_smote_data_loader import load_deepsmote_raw
from imbalanceddl.utils.deep_smote_data_loader import CustomImageDataset
import torchvision.transforms as transforms

def main():
    log_filename = f"sava_deepsmote_compute_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = Tee(log_filename)
    sys.stderr = sys.stdout
    print(f"Logging to {log_filename}")

    config = get_args()
    prepare_store_name(config)
    prepare_folders(config)
    if config.seed is None:
        config.seed = np.random.randint(10000)
    fix_all_seed(config.seed)

    device = 'cpu'
    torch.cuda.empty_cache()
    print(f"Using device: {device}")

    # Set number of classes based on dataset
    if config.dataset == 'cifar10':
        config.num_classes = 10
    elif config.dataset == 'cifar100':
        config.num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    # Use the same validation transform as in training (ToTensor + Normalize)
    # FIX: pass dataset argument to get_weak_augmentation
    _, val_transform = get_weak_augmentation(config.dataset)

    # Load DeepSMOTE balanced data
    print(f"Loading DeepSMOTE balanced data for {config.dataset}, imb_type={config.imb_type}, imb_factor={config.imb_factor}")
    X, Y = load_deepsmote_raw(config.dataset, config.imb_type, config.imb_factor)
    # X is (N,32,32,3) uint8, Y is (N,) int
    print(f"Data shape: {X.shape}, labels shape: {Y.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(Y, return_counts=True)))}")

    # Create dataset with validation transform (no augmentation, just ToTensor+Normalize)
    train_ds = CustomImageDataset(X, Y, transform=val_transform)
    print(f"Training dataset size: {len(train_ds)}")

    # Validation dataset – use a subset of the original CIFAR-10 test set (2000 samples)
    data_root = '/mnt/hdd2/phatht/phat/imbalanced-DL-sampling/data'
    full_val = datasets.CIFAR10(root=data_root, train=False, download=False, transform=val_transform)
    val_subset_size = 2000
    val_ds = Subset(full_val, range(val_subset_size))
    print(f"Validation subset size: {val_subset_size}")

    # Label check
    train_labels = [train_ds[i][1] for i in range(min(1000, len(train_ds)))]
    val_labels   = [val_ds[i][1]   for i in range(val_subset_size)]
    print(f"Train unique classes: {np.unique(train_labels)}")
    print(f"Val unique classes: {np.unique(val_labels)}")

    # Cache key: deepsmote=True, no noise, etc.
    flags = {
        'is_deepsmote': True,
        'is_oversampled': False,
        'is_noisy': False,
        'is_noise_first': False,
        'is_selection_first': False
    }
    key_gen = SavaCacheKey(config=config, **flags)
    file_key = key_gen.generate()
    print(f"Cache key: {file_key}")

    # SAVA parameters (raw pixels, batch size 500, no corruption)
    config.sava_batch_size = 500   # safe batch size
    config.sava_cache_label_distances = True

    print("Calling SAVA selection (raw pixels, DeepSMOTE balanced data, batch_size=500)...")
    indices = get_sava_selection_indices(
        train_dataset=train_ds,
        val_dataset=val_ds,
        keep_ratio=config.selection_ratio,   # not used for caching, but passed
        device=device,
        file_key=file_key,
        batch_size=config.sava_batch_size,
        num_classes=config.num_classes,
        resize=32,
        cache_label_distances=config.sava_cache_label_distances,
        corrupt_por=0.0
    )
    print(f"SAVA scores computed. Selected {len(indices)} out of {len(train_ds)} samples.")
    print("Sorted indices cached in sava_selection_results/")
    print("Exiting.")

if __name__ == "__main__":
    main()