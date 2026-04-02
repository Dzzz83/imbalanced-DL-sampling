import sys
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from imbalanceddl.dataset.imbalance_cifar import IMBALANCECIFAR10
from imbalanceddl.strategy.selection_method.lava_selection import get_lava_selection_indices
from imbalanceddl.utils.config import get_args
import torchvision

def get_imbalanced_dataset(cfg):
    """Load the imbalanced CIFAR-10 training set with the given config."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = IMBALANCECIFAR10(
        root='./data',
        imb_type=cfg.imb_type,
        imb_factor=cfg.imb_factor,
        rand_number=cfg.rand_number,
        train=True,
        download=True,
        transform=transform
    )
    return train_dataset

def get_validation_dataset():
    """Return the original CIFAR-10 test set (balanced, clean)."""
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

def denormalize(tensor):
    """Convert normalized tensor to [0,1] numpy array for display."""
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img = tensor.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def main():
    # Load configuration (needed for dataset parameters)
    cfg = get_args()  # assumes you have a config file with imb_type, imb_factor, rand_number
    # Alternatively, set parameters manually:
    cfg.imb_type = 'exp'
    cfg.imb_factor = 0.01
    cfg.rand_number = 0
    cfg.dataset = 'cifar10'
    cfg.selection_method = 'lava'  # dummy

    # Load datasets
    train_dataset = get_imbalanced_dataset(cfg)
    val_dataset = get_validation_dataset()

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Compute or load LAVA scores using caching
    # We need to call get_lava_selection_indices with keep_ratio=1.0 to get all scores
    # The function will use the cache key based on dataset, imb_type, imb_factor, rand_number
    # We'll pass file_key manually.
    file_key = f"{cfg.dataset}_{cfg.imb_type}_{cfg.imb_factor}_{cfg.rand_number}"
    # get_lava_selection_indices returns indices for keep_ratio, but we want scores.
    # We can modify the function to also return scores, or we can call the internal caching logic directly.
    # Simpler: we call get_lava_selection_indices with keep_ratio=1.0, then retrieve scores from cache.
    # However, get_lava_selection_indices doesn't return scores. We'll extract scores from cache.
    from imbalanceddl.strategy.selection_method.lava_selection import get_saved_scores
    training_size = len(train_dataset)
    scores, _ = get_saved_scores(file_key, training_size)
    if scores is None:
        # Compute scores by calling get_lava_selection_indices with keep_ratio=1.0 (which will compute and cache)
        print("Computing LAVA scores (this may take a few minutes)...")
        indices = get_lava_selection_indices(
            train_dataset, val_dataset, keep_ratio=1.0, device='cuda', file_key=file_key
        )
        # After computation, load scores from cache
        scores, _ = get_saved_scores(file_key, training_size)
        if scores is None:
            raise RuntimeError("Failed to compute or load LAVA scores.")
    else:
        print("Loaded cached LAVA scores.")

    # Now scores is a numpy array of length training_size
    # Get indices sorted by score (ascending: best first)
    sorted_idx = np.argsort(scores)
    best_indices = sorted_idx[:10]   # 10 lowest scores (best quality)
    worst_indices = sorted_idx[-10:][::-1]  # 10 highest scores (worst quality)

    # Prepare output directory
    out_dir = 'lava_testing_results'
    os.makedirs(out_dir, exist_ok=True)

    # Class names for CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Save best images
    print("\n--- Top 10 Best (Lowest Scores) ---")
    for rank, idx in enumerate(best_indices):
        img_tensor, label = train_dataset[idx]
        img_np = denormalize(img_tensor)
        score = scores[idx]
        class_name = class_names[label]
        filename = f"best_{rank+1:02d}_score_{score:.2f}_class_{class_name}.png"
        filepath = os.path.join(out_dir, filename)
        plt.imsave(filepath, img_np)
        print(f"{rank+1:2}. Index {idx:5d} | Score {score:8.2f} | Class {class_name}")

    # Save worst images
    print("\n--- Top 10 Worst (Highest Scores) ---")
    for rank, idx in enumerate(worst_indices):
        img_tensor, label = train_dataset[idx]
        img_np = denormalize(img_tensor)
        score = scores[idx]
        class_name = class_names[label]
        filename = f"worst_{rank+1:02d}_score_{score:.2f}_class_{class_name}.png"
        filepath = os.path.join(out_dir, filename)
        plt.imsave(filepath, img_np)
        print(f"{rank+1:2}. Index {idx:5d} | Score {score:8.2f} | Class {class_name}")

    print(f"\nImages saved to '{out_dir}' directory.")

if __name__ == "__main__":
    main()