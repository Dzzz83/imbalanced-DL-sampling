#!/usr/bin/env python3
import sys
import os
import datetime
import numpy as np
import torch
from torch.utils.data import Subset
from unittest.mock import MagicMock

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

SAVA_ROOT = '/home/phatht/phat/imbalanced-DL-sampling/sava'

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
from torchvision import datasets

def main():
    log_filename = f"sava_compute_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = Tee(log_filename)
    sys.stderr = sys.stdout
    print(f"Logging to {log_filename}")

    config = get_args()
    prepare_store_name(config)
    prepare_folders(config)
    if config.seed is None:
        config.seed = np.random.randint(10000)
    fix_all_seed(config.seed)

    device = f'cuda:{config.gpu}' if hasattr(config, 'gpu') and config.gpu is not None else 'cuda'
    torch.cuda.empty_cache()
    print(f"Using device: {device}")

    _, val_transform = get_weak_augmentation()
    print("Loading CIFAR‑10...")
    full_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=val_transform)
    full_val   = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)

    # ---- Option 1: full training set (50k), small validation (500) ----
    train_subset_size = 2000
    val_subset_size = 500
    train_ds = Subset(full_train, range(train_subset_size))
    val_ds   = Subset(full_val,   range(val_subset_size))
    print(f"Training subset: {train_subset_size} samples (single batch)")
    print(f"Validation subset: {val_subset_size} samples")

    # Verify labels (only first 1000 for speed)
    train_labels = [train_ds[i][1] for i in range(min(1000, len(train_ds)))]
    val_labels   = [val_ds[i][1]   for i in range(val_subset_size)]
    print(f"Train unique classes: {np.unique(train_labels)}")
    print(f"Val unique classes: {np.unique(val_labels)}")

    flags = {
        'is_deepsmote': False, 'is_oversampled': False, 'is_noisy': False,
        'is_noise_first': False, 'is_selection_first': False
    }
    key_gen = SavaCacheKey(config=config, **flags)
    file_key = key_gen.generate()

    # Raw pixels, batch size 1024 (original default)
    config.sava_feat_repr = False
    config.sava_batch_size = train_subset_size   
    config.sava_parallel = False
    config.sava_n_gpu = 1
    config.sava_cuda_num = getattr(config, 'sava_cuda_num', 0)
    config.sava_cache_label_distances = True
    config.sava_model_path = None

    if hasattr(config, 'workers'):
        config.workers = 0

    print("Calling SAVA selection (raw pixels, full training set, batch_size=1024)...")
    indices = get_sava_selection_indices(
        train_dataset=train_ds,
        val_dataset=val_ds,
        keep_ratio=config.selection_ratio,
        device=device,
        file_key=file_key,
        batch_size=config.sava_batch_size,
        num_classes=config.num_classes,
        feat_repr=config.sava_feat_repr,
        parallel=config.sava_parallel,
        cuda_num=config.sava_cuda_num,
        n_gpu=config.sava_n_gpu,
        resize=getattr(config, 'resize', 32),
        cache_label_distances=config.sava_cache_label_distances,
        model_path=None,
        corrupt_por=0.01
    )
    print(f"SAVA scores computed. Selected {len(indices)} out of {len(train_ds)} samples.")
    print("Exiting.")

if __name__ == "__main__":
    main()