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

# 1. Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
full_val = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 2. Balanced validation set: 5 images per class (total 50)
np.random.seed(42)
val_data, val_labels = [], []
for class_id in range(10):
    class_indices = [i for i, (_, lbl) in enumerate(full_val) if lbl == class_id]
    chosen = np.random.choice(class_indices, 5, replace=False)
    for idx in chosen:
        img, label = full_val[idx]
        val_data.append(img)
        val_labels.append(label)
val_dataset = TensorDataset(torch.stack(val_data), torch.tensor(val_labels))
val_dataset.targets = val_labels

# 3. Balanced training set: 10 images per class (total 100)
train_data, train_labels = [], []
for class_id in range(10):
    class_indices = [i for i, (_, lbl) in enumerate(full_train) if lbl == class_id]
    chosen = np.random.choice(class_indices, 10, replace=False)
    for idx in chosen:
        img, label = full_train[idx]
        train_data.append(img)
        train_labels.append(label)

# 4. Corruption: strong noise (std=2.0) and mislabel between distant classes
def add_gaussian_noise(img, std=2.0):
    noise = torch.randn_like(img) * std
    return img + noise

# Define distant class mapping for mislabeling (cat->airplane, dog->truck, etc.)
# We'll use a fixed mapping for classes 0-9 to ensure large label distance
# For simplicity, map each class to a class that is very different:
# 0: airplane -> 3: cat (but we want cat->airplane, so we need to map)
# We'll create a mapping dictionary:
# Original class -> new class (very different)
# 0 airplane -> 9 truck (different)
# 1 automobile -> 5 dog (different)
# 2 bird -> 1 automobile
# 3 cat -> 0 airplane
# 4 deer -> 9 truck
# 5 dog -> 1 automobile
# 6 frog -> 2 bird
# 7 horse -> 3 cat
# 8 ship -> 0 airplane
# 9 truck -> 3 cat
mislabel_map = {
    0: 9,  # airplane -> truck
    1: 5,  # automobile -> dog
    2: 1,  # bird -> automobile
    3: 0,  # cat -> airplane
    4: 9,  # deer -> truck
    5: 1,  # dog -> automobile
    6: 2,  # frog -> bird
    7: 3,  # horse -> cat
    8: 0,  # ship -> airplane
    9: 3   # truck -> cat
}

corrupted_flags = []
for idx in range(len(train_data)):
    img = train_data[idx]
    label = train_labels[idx]
    # Corrupt first 2 of each class with noise
    if idx % 10 < 2:
        img = add_gaussian_noise(img, std=2.0)
        corrupted_flags.append(True)
    # Next 2 of each class with mislabel to distant class
    elif idx % 10 < 4:
        new_label = mislabel_map[label]
        label = new_label
        corrupted_flags.append(True)
    else:
        corrupted_flags.append(False)
    train_data[idx] = img
    train_labels[idx] = label

train_dataset = TensorDataset(torch.stack(train_data), torch.tensor(train_labels))
train_dataset.targets = train_labels

# 5. Compute LAVA scores
train_wrapper, val_wrapper = dataset_prep(train_dataset, val_dataset)
train_loader = DataLoader(train_wrapper, batch_size=128, shuffle=False)
val_loader = DataLoader(val_wrapper, batch_size=128, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
extractor = get_feature_extractor(device)

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

# 6. Calibrated scores
f = dual_sol[0].detach().cpu().numpy().flatten()
N = len(f)
scores = (N / (N - 1)) * (f - np.mean(f))

# 7. Results
print("Top 10 highest scores (worst quality):")
top10_idx = np.argsort(scores)[-10:][::-1]
for i, idx in enumerate(top10_idx):
    corrupted = "CORRUPTED" if corrupted_flags[idx] else "clean"
    print(f"{i+1:2}. Score {scores[idx]:.2f} | Index {idx} | {corrupted}")

# Histogram
clean_scores = [scores[i] for i in range(len(scores)) if not corrupted_flags[i]]
corrupted_scores = [scores[i] for i in range(len(scores)) if corrupted_flags[i]]

plt.figure(figsize=(10,5))
plt.hist(clean_scores, bins=20, alpha=0.5, label='Clean')
plt.hist(corrupted_scores, bins=20, alpha=0.5, label='Corrupted')
plt.xlabel('LAVA Score')
plt.ylabel('Count')
plt.legend()
plt.title('LAVA Score Distribution on Clean vs Corrupted Data (strong noise, distant mislabel)')
plt.show()