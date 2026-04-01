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
val_clean = Subset(full_val, val_indices)

# 2. Create corrupted copies
def add_gaussian_noise(img, mean=0, std=0.5):
    noise = torch.randn_like(img) * std + mean
    return img + noise

train_data = []
train_labels = []
corrupted_flags = []

for idx in range(len(train_clean)):
    img, label = train_clean[idx]
    # Mark first 10 as noise-corrupted
    if idx < 10:
        img = add_gaussian_noise(img)
        corrupted_flags.append(True)
    # Next 10 as mislabeled
    elif idx < 20:
        new_label = np.random.choice([l for l in range(10) if l != label])
        label = new_label
        corrupted_flags.append(True)
    else:
        corrupted_flags.append(False)
    train_data.append(img)
    train_labels.append(label)

# Create a custom dataset for training
train_dataset = TensorDataset(torch.stack(train_data), torch.tensor(train_labels))

# 3. Compute LAVA scores (using your existing function)
# We need to wrap the dataset for OTDD (the wrapper expects a dataset that returns (img, label))
from imbalanceddl.strategy.selection_method.lava_selection import OTDDWrapper, dataset_prep
train_wrapper, val_wrapper = dataset_prep(train_dataset, val_clean)

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

# Extract source potentials (f)
f = dual_sol[0].detach().cpu().numpy().flatten()
assert len(f) == len(train_dataset)

# 4. Analyze scores
scores = f   # these are the LAVA scores (higher = worse)

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