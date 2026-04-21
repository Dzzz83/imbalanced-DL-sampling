#!/usr/bin/env python3
"""
verify_lava_noise_detection.py

Verifies whether LAVA scores are significantly higher (worse) for samples whose labels have been corrupted.
Usage: python verify_lava_noise_detection.py [--noise_ratio 0.1] [--seed 42] [--imb_factor 0.01]
"""

import sys
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from unittest.mock import MagicMock
from collections import Counter

# ----------------------------------------------------------------------
# 1. Locate project root and change working directory
# ----------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 4 levels: test -> selection_method -> strategy -> imbalanceddl -> imbalanced-DL-sampling
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
os.chdir(project_root)
sys.path.insert(0, project_root)

print(f"Working directory set to: {os.getcwd()}")

# ----------------------------------------------------------------------
# 2. Mock unnecessary libraries
# ----------------------------------------------------------------------
lib = ["torchtext", "torchtext.data", "torchtext.data.utils",
       "torchtext.datasets", "torchtext.vocab", "vgg", "resnet"]
for mod in lib:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# ----------------------------------------------------------------------
# 3. Imports
# ----------------------------------------------------------------------
from imbalanceddl.dataset.imbalance_cifar import IMBALANCECIFAR10
from imbalanceddl.strategy.selection_method.lava_selection import (
    get_lava_selection_indices, get_saved_scores
)

# ----------------------------------------------------------------------
# 4. Helper functions
# ----------------------------------------------------------------------
def get_imbalanced_dataset(imb_type='exp', imb_factor=0.01, rand_number=0, transform=None):
    """Return an IMBALANCECIFAR10 dataset (clean, no noise)."""
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    train_dataset = IMBALANCECIFAR10(
        root='./data',
        imb_type=imb_type,
        imb_factor=imb_factor,
        rand_number=rand_number,
        train=True,
        download=True,
        transform=transform
    )
    return train_dataset

def get_validation_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    return val_dataset

def inject_label_noise(dataset, noise_ratio=0.1, seed=42):
    """
    Injects symmetric label noise into the dataset (modifies dataset.targets in place).
    Returns:
        corrupted_indices: list of indices whose labels were flipped.
    """
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    num_samples = len(targets)
    num_noisy = int(num_samples * noise_ratio)
    corrupted_idx = rng.choice(num_samples, num_noisy, replace=False)
    num_classes = len(np.unique(targets))
    for idx in corrupted_idx:
        current = targets[idx]
        possible = [c for c in range(num_classes) if c != current]
        new_label = rng.choice(possible)
        targets[idx] = new_label
    dataset.targets = targets.tolist()
    # Update internal class count attributes if present
    if hasattr(dataset, 'num_per_cls_dict'):
        dataset.num_per_cls_dict = dict(Counter(dataset.targets))
    if hasattr(dataset, 'cls_num_list'):
        dataset.cls_num_list = [Counter(dataset.targets)[i] for i in range(num_classes)]
    return corrupted_idx

def ensure_scores(dataset, val_dataset, file_key, device='cuda', recompute=False):
    """Ensure LAVA scores are computed and return the score array."""
    training_size = len(dataset)
    scores, _ = get_saved_scores(file_key, training_size)
    if scores is None or recompute:
        print(f"Computing LAVA scores for key {file_key} ...")
        _ = get_lava_selection_indices(
            dataset, val_dataset,
            keep_ratio=1.0,
            device=device,
            file_key=file_key
        )
        scores, _ = get_saved_scores(file_key, training_size)
        if scores is None:
            raise RuntimeError(f"Failed to compute LAVA scores for key {file_key}")
    else:
        print(f"Loaded cached LAVA scores for key {file_key}")
    return scores

# ----------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify LAVA noise detection.")
    parser.add_argument('--noise_ratio', type=float, default=0.1, help='Fraction of labels to corrupt')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for noise injection')
    parser.add_argument('--imb_type', type=str, default='exp', help='Imbalance type')
    parser.add_argument('--imb_factor', type=float, default=0.01, help='Imbalance factor')
    parser.add_argument('--rand_number', type=int, default=0, help='Random number for dataset generation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run LAVA computation')
    parser.add_argument('--recompute', action='store_true', help='Force recompute LAVA scores')
    args = parser.parse_args()

    # Parameters
    imb_type = args.imb_type
    imb_factor = args.imb_factor
    rand_number = args.rand_number
    dataset_name = 'cifar10'
    device = args.device

    print(f"Parameters: noise_ratio={args.noise_ratio}, seed={args.seed}, "
          f"imb_type={imb_type}, imb_factor={imb_factor}")

    # Load clean dataset and validation set
    print("Loading clean imbalanced training set...")
    clean_dataset = get_imbalanced_dataset(imb_type, imb_factor, rand_number)
    val_dataset = get_validation_dataset()
    print(f"Clean training set size: {len(clean_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Clean cache key (no noise)
    clean_file_key = f"{dataset_name}_{imb_type}_{imb_factor}_{rand_number}"
    print(f"Clean file key: {clean_file_key}")

    # Ensure clean scores are available
    clean_scores = ensure_scores(clean_dataset, val_dataset, clean_file_key, device, recompute=args.recompute)

    # Create a noisy copy of the dataset
    print("\nCreating a copy of the dataset for noise injection...")
    # Because the dataset is a torchvision dataset, we can reload it with the same parameters.
    # The transform must be the same as used for scoring (no augmentation).
    noisy_dataset = get_imbalanced_dataset(imb_type, imb_factor, rand_number)
    # Inject label noise
    corrupted_idx = inject_label_noise(noisy_dataset, noise_ratio=args.noise_ratio, seed=args.seed)
    print(f"Injected noise into {len(corrupted_idx)} samples (ratio={args.noise_ratio})")
    # Show new class distribution
    if hasattr(noisy_dataset, 'num_per_cls_dict'):
        print(f"New class distribution: {noisy_dataset.num_per_cls_dict}")

    # Noisy cache key (include noise ratio to avoid collision)
    noisy_file_key = f"{dataset_name}_{imb_type}_{imb_factor}_{rand_number}_noise{args.noise_ratio}"
    print(f"Noisy file key: {noisy_file_key}")

    # Compute noisy scores
    noisy_scores = ensure_scores(noisy_dataset, val_dataset, noisy_file_key, device, recompute=args.recompute)

    # Compare scores for corrupted indices
    corrupted_scores = noisy_scores[corrupted_idx]
    clean_corrupted_scores = clean_scores[corrupted_idx]   # scores of same images in clean dataset
    all_other_scores = np.delete(noisy_scores, corrupted_idx)

    print("\n=== LAVA Score Comparison ===")
    print(f"Corrupted samples (noisy labels): mean={np.mean(corrupted_scores):.4f}, std={np.std(corrupted_scores):.4f}")
    print(f"Same images in clean dataset: mean={np.mean(clean_corrupted_scores):.4f}")
    print(f"Uncorrupted samples in noisy dataset: mean={np.mean(all_other_scores):.4f}, std={np.std(all_other_scores):.4f}")

    # Statistical test (one‑sided: corrupted scores > uncorrupted)
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(corrupted_scores, all_other_scores, alternative='greater')
    print(f"Mann-Whitney U test (corrupted > uncorrupted): p={p:.6f}")
    if p < 0.05:
        print("=> Corrupted samples have significantly HIGHER (worse) LAVA scores.")
    else:
        print("=> No significant difference detected (LAVA may not detect this noise level).")

    # Plot histogram
    out_dir = 'lava_noise_test_results'
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10,6))
    plt.hist(all_other_scores, bins=50, alpha=0.5, label='Uncorrupted', density=True)
    plt.hist(corrupted_scores, bins=20, alpha=0.5, label='Corrupted', density=True)
    plt.xlabel('LAVA Score (lower is better)')
    plt.ylabel('Density')
    plt.title(f'LAVA Score Distribution: Clean vs Noisy Labels (noise_ratio={args.noise_ratio})')
    plt.legend()
    plt.grid(True)
    out_plot = os.path.join(out_dir, f'lava_noise_detection_noise{args.noise_ratio}_seed{args.seed}.png')
    plt.savefig(out_plot, dpi=150)
    print(f"Saved histogram to {out_plot}")

    # Optionally, also show boxplot per class for noisy dataset
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    scores_per_class = {i: [] for i in range(10)}
    for idx, label in enumerate(noisy_dataset.targets):
        # Convert label to int if it's a tensor
        lbl = int(label) if hasattr(label, 'item') else label
        scores_per_class[lbl].append(noisy_scores[idx])
    data_to_plot = [scores_per_class[i] for i in range(10)]
    plt.figure(figsize=(12, 6))
    bp = plt.boxplot(data_to_plot, labels=class_names, patch_artist=True,
                     showmeans=True, meanline=True, medianprops={'linewidth': 2})
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('LAVA Score')
    plt.title('LAVA Score Distribution per Class (Noisy Dataset)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    boxplot_path = os.path.join(out_dir, f'noisy_score_distribution_per_class_noise{args.noise_ratio}.png')
    plt.savefig(boxplot_path, dpi=150)
    print(f"Saved per‑class boxplot to {boxplot_path}")
    plt.show()

if __name__ == "__main__":
    main()