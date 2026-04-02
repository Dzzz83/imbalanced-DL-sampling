import sys
import os
from unittest.mock import MagicMock

# Add the project root (three levels up from this file) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

lib = ["torchtext", "torchtext.data", "torchtext.data.utils",
       "torchtext.datasets", "torchtext.vocab", "vgg", "resnet"]
for mod in lib:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()


import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from imbalanceddl.strategy.selection_method.lava_selection import (
    get_lava_selection_indices, get_feature_extractor, dataset_prep
)
from torch.utils.data import DataLoader, Subset, TensorDataset

# 1. Load a small clean CIFAR-10 subset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
full_val = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Take a small random subset for testing
np.random.seed(42)
train_indices = np.random.choice(len(full_train), 50, replace=False)
val_indices = np.random.choice(len(full_val), 20, replace=False)

train_clean = Subset(full_train, train_indices)

# Build a balanced validation set (2 images per class, total 20)
np.random.seed(42)  # same seed for reproducibility
val_data = []
val_labels = []
for class_id in range(10):
    # Get all indices in full_val that belong to this class
    class_indices = [i for i, (_, lbl) in enumerate(full_val) if lbl == class_id]
    # Randomly pick 2 indices (without replacement)
    chosen = np.random.choice(class_indices, 2, replace=False)
    for idx in chosen:
        img, label = full_val[idx]
        val_data.append(img)
        val_labels.append(label)
val_dataset = TensorDataset(torch.stack(val_data), torch.tensor(val_labels))
val_dataset.targets = val_labels 

# 2. Create corrupted copies with balanced classes
def add_gaussian_noise(img, mean=0, std=0.5):
    noise = torch.randn_like(img) * std + mean
    return img + noise

# Take 10 images per class (100 total) to ensure enough samples after corruption
train_data = []
train_labels = []
corrupted_flags = []
np.random.seed(42)

for class_id in range(10):
    class_indices = [i for i, (_, lbl) in enumerate(full_train) if lbl == class_id]
    chosen = np.random.choice(class_indices, 10, replace=False)  # 10 per class
    for idx in chosen:
        img, label = full_train[idx]
        train_data.append(img)
        train_labels.append(label)

# Now corrupt: first 20 images (2 per class) with noise, next 20 (2 per class) with mislabel
# This leaves 6 clean images per class.
corrupted_flags = []
for idx in range(len(train_data)):
    img = train_data[idx]
    label = train_labels[idx]
    # Determine which class this image belongs to (original class)
    class_id = idx // 10   # because 10 images per class, in order
    # Noise for the first 2 images of each class (global indices 0-9,10-19,...,90-99)
    if idx % 10 < 2:
        img = add_gaussian_noise(img)
        corrupted_flags.append(True)
    # Mislabel for the next 2 images of each class (global indices 2-3 per block)
    elif idx % 10 < 4:
        new_label = np.random.choice([l for l in range(10) if l != label])
        label = new_label
        corrupted_flags.append(True)
    else:
        corrupted_flags.append(False)
    train_data[idx] = img
    train_labels[idx] = label

# Create dataset
train_dataset = TensorDataset(torch.stack(train_data), torch.tensor(train_labels))
train_dataset.targets = train_labels  

# 3. Compute LAVA scores (using your existing function)
# We need to wrap the dataset for OTDD (the wrapper expects a dataset that returns (img, label))
from imbalanceddl.strategy.selection_method.lava_selection import OTDDWrapper, dataset_prep
train_wrapper, val_wrapper = dataset_prep(train_dataset, val_dataset)

# Create data loaders (shuffle=False to keep order)
train_loader = DataLoader(train_wrapper, batch_size=128, shuffle=False)
val_loader = DataLoader(val_wrapper, batch_size=128, shuffle=False)

# Load feature extractor (use the same one as in your training)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
extractor = get_feature_extractor(device)

# Get dual potentials (the code returns [f, g])
from LAVA.lava import get_OT_dual_sol
dual_sol = get_OT_dual_sol(
    feature_extractor=extractor,
    trainloader=train_loader,
    testloader=val_loader,
    training_size=len(train_dataset),
    p=2,
    resize=32,
    device=device
)
print("dual_sol length:", len(dual_sol))
for i, item in enumerate(dual_sol):
    if torch.is_tensor(item):
        print(f"  dual_sol[{i}] shape: {item.shape}")
    else:
        print(f"  dual_sol[{i}] type: {type(item)}")
# Extract source potentials (f)
f = dual_sol[0].detach().cpu().numpy().flatten()
assert len(f) == len(train_dataset)

# 4. Analyze scores
N = len(f)
scores = (N / (N - 1)) * (f - np.mean(f))

# Print top 10 highest scores
print("Top 10 highest scores (worst quality):")
top10_idx = np.argsort(scores)[-10:][::-1]
for i, idx in enumerate(top10_idx):
    corrupted = "CORRUPTED" if corrupted_flags[idx] else "clean"
    print(f"{i+1:2}. Score {scores[idx]:.4f} | Index {idx} | {corrupted}")

# Histogram
clean_scores = [scores[i] for i in range(len(scores)) if not corrupted_flags[i]]
corrupted_scores = [scores[i] for i in range(len(scores)) if corrupted_flags[i]]

plt.hist(clean_scores, bins=20, alpha=0.5, label='Clean')
plt.hist(corrupted_scores, bins=20, alpha=0.5, label='Corrupted')
plt.xlabel('LAVA Score')
plt.ylabel('Count')
plt.legend()
plt.title('LAVA Score Distribution on Clean vs Corrupted Data')
plt.show()