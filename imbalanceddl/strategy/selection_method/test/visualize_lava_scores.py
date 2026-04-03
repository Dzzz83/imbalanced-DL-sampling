import sys
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from unittest.mock import MagicMock

# ----------------------------------------------------------------------
# 1. Locate project root and change working directory
# ----------------------------------------------------------------------
# This file is at: imbalanced-DL-sampling/imbalanceddl/strategy/selection_method/test/visualize_lava_scores.py
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
def get_imbalanced_dataset(imb_type='exp', imb_factor=0.01, rand_number=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = IMBALANCECIFAR10(
        root='./data',   # now relative to project root
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

def denormalize(tensor):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2023, 0.1994, 0.2010])
    img = tensor.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

# ----------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------
def main():
    # Dataset parameters (match your training config)
    imb_type = 'exp'
    imb_factor = 0.01
    rand_number = 0
    dataset_name = 'cifar10'

    print("Loading imbalanced training set...")
    train_dataset = get_imbalanced_dataset(imb_type, imb_factor, rand_number)
    print("Loading validation set (original test set)...")
    val_dataset = get_validation_dataset()

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Cache key
    file_key = f"{dataset_name}_{imb_type}_{imb_factor}_{rand_number}"
    training_size = len(train_dataset)

    # Load or compute scores
    scores, _ = get_saved_scores(file_key, training_size)
    if scores is None:
        print("Computing LAVA scores (this may take ~6 minutes)...")
        _ = get_lava_selection_indices(
            train_dataset, val_dataset,
            keep_ratio=1.0,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            file_key=file_key
        )
        scores, _ = get_saved_scores(file_key, training_size)
        if scores is None:
            raise RuntimeError("Failed to compute or load LAVA scores.")
    else:
        print("Loaded cached LAVA scores.")

    num_images = 10

    # Sort indices by score
    sorted_idx = np.argsort(scores)
    best_indices = sorted_idx[:num_images]
    worst_indices = sorted_idx[-num_images:][::-1]

    out_dir = 'lava_testing_results'
    os.makedirs(out_dir, exist_ok=True)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Save best images
    print(f"\n--- Top {num_images} Best (Lowest Scores) ---")
    for rank, idx in enumerate(best_indices):
        img_tensor, label = train_dataset[idx]
        img_np = denormalize(img_tensor)
        score = scores[idx]
        class_name = class_names[label]
        filename = f"best_{rank+1:03d}_score_{score:.2f}_class_{class_name}.png"
        filepath = os.path.join(out_dir, filename)
        plt.imsave(filepath, img_np)
        print(f"{rank+1:3}. Index {idx:5d} | Score {score:8.2f} | Class {class_name} (idx {label})")

    # Save worst images
    print(f"\n--- Top {num_images} Worst (Highest Scores) ---")
    for rank, idx in enumerate(worst_indices):
        img_tensor, label = train_dataset[idx]
        img_np = denormalize(img_tensor)
        score = scores[idx]
        class_name = class_names[label]
        filename = f"worst_{rank+1:03d}_score_{score:.2f}_class_{class_name}.png"
        filepath = os.path.join(out_dir, filename)
        plt.imsave(filepath, img_np)
        print(f"{rank+1:3}. Index {idx:5d} | Score {score:8.2f} | Class {class_name} (idx {label})")

    print(f"\nImages saved to '{out_dir}' directory.")

        # --- 1. Boxplot with consistent color ---
    scores_per_class = {i: [] for i in range(10)}
    for idx, label in enumerate(train_dataset.targets):
        scores_per_class[label].append(scores[idx])

    data_to_plot = [scores_per_class[i] for i in range(10)]
    class_labels = class_names

    plt.figure(figsize=(12, 6))
    bp = plt.boxplot(data_to_plot, labels=class_labels, patch_artist=True,
                     showmeans=True, meanline=True, medianprops={'linewidth': 2})
    # Use a single color for all boxes
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('LAVA Score')
    plt.title('LAVA Score Distribution per Class (Imbalanced CIFAR-10)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'score_distribution_per_class.png'), dpi=150)
    plt.show()

if __name__ == "__main__":
    main()