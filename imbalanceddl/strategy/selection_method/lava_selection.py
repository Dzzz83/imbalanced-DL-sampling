import sys
import os 
from unittest.mock import MagicMock

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from LAVA import lava
from LAVA.lava import compute_dual
print("Successfully imported LAVA.lava")

# fake all the unncessary libraries
lib = ["torchtext", "torchtext.data", "torchtext.data.utils", 
            "torchtext.datasets", "torchtext.vocab", "vgg", "resnet"]
for mod in lib:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

from LAVA.lava import compute_dual, compute_values_and_visualize


# replace the original compute_dual with the new compute_dual_1 
def compute_dual_1(feature_extractor, trainloader, testloader, training_size, shuffle_ind, p=2, resize=32, device='cuda'):
    # train_indices = lava.get_indices(trainloader)
    train_indices = list(range(training_size))
    trained_with_flag = lava.train_with_corrupt_flag(trainloader, shuffle_ind, train_indices)

    dual_sol = lava.get_OT_dual_sol(
        feature_extractor,
        trainloader,
        testloader,
        training_size=training_size,
        p=p,
        resize=resize,
        device=device
    )
    return dual_sol, trained_with_flag

lava.compute_dual = compute_dual_1

# wrap the ResNet model to extract the features
class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.base_model.fc = nn.Identity()

    def forward(self, x):
        features = self.base_model(x)
        return features
# OTDD expects (image, label) | PyTorch returns (image, label, index)
# this class wraps the dataset and returns (image, label)
class OTDDWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        if hasattr(dataset, 'targets'):
            self.targets = dataset.targets
        if hasattr(dataset, 'indices'):
            self.indices = dataset.indices

    def __getitem__(self, index):
        item = self.dataset[index]

        return item[0], item[1]
    
    def __len__(self):
        return len(self.dataset)
    
# make the "targets" become Tensor for calculation
def dataset_prep(train_dataset, val_dataset):
    dataset_list = [train_dataset, val_dataset]
    for ds in dataset_list:
        # if there is "targets" and not in tensor form
        if hasattr(ds, 'targets') and not isinstance(ds, torch.Tensor):
            # transform it to tensor
            ds.targets = torch.LongTensor(ds.targets)
    
    return OTDDWrapper(train_dataset), OTDDWrapper(val_dataset)

# load the IMAGENET1K ResNet18 model to extract feature
def get_feature_extractor(device):
    base_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    model = FeatureExtractor(base_resnet)
    model.eval()
    return model

def select_indices(lava_values, training_size, keep_ratio):
    train_scores = lava_values[:training_size]          # only training duals
    selected_sample_size = int(len(train_scores) * keep_ratio)
    # sort ascending, take first selected_sample_size (smallest scores)
    selected_indices = np.argsort(train_scores)[:selected_sample_size]
    return selected_indices.tolist()

def get_lava_selection_indices(train_dataset, val_dataset, keep_ratio=0.7, device='cuda'):
    # Prepare datasets
    train_wrapper, val_wrapper = dataset_prep(train_dataset, val_dataset)

    # Create dataloaders (shuffle=False to preserve order)
    train_loader = DataLoader(train_wrapper, batch_size=128, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_wrapper, batch_size=128, shuffle=False)

    # Feature extractor
    extractor = get_feature_extractor(device)
    training_size = len(train_dataset)

    print(f"--- LAVA Selection Started ---")
    print(f"Total training samples to evaluate: {training_size}")

    # Compute OT duals
    dual_sol, _ = lava.compute_dual(
        feature_extractor=extractor,
        trainloader=train_loader,
        testloader=val_loader,
        training_size=training_size,
        shuffle_ind=[],
        device=device
    )

    # --- DEBUG: Inspect dual_sol structure ---
    print("\n--- dual_sol structure ---")
    print(f"type(dual_sol): {type(dual_sol)}")
    if isinstance(dual_sol, (tuple, list)):
        print(f"len(dual_sol): {len(dual_sol)}")
        for i, elem in enumerate(dual_sol):
            if isinstance(elem, torch.Tensor):
                print(f"  elem[{i}] shape: {elem.shape}")
            else:
                print(f"  elem[{i}] type: {type(elem)}")
    else:
        print(f"dual_sol is not a tuple/list; shape: {dual_sol.shape if hasattr(dual_sol, 'shape') else 'scalar'}")

    # --- Extract source potentials (f) ---
    f_tensor = None
    if isinstance(dual_sol, (tuple, list)):
        # dual_sol[1] should be source potentials (size = training_size)
        if len(dual_sol) >= 2 and isinstance(dual_sol[1], torch.Tensor) and dual_sol[1].numel() == training_size:
            f_tensor = dual_sol[1]
            print("Extracted source potentials from dual_sol[1]")
        else:
            # Fallback: search for any tensor with correct length
            for i, elem in enumerate(dual_sol):
                if isinstance(elem, torch.Tensor) and elem.numel() == training_size:
                    f_tensor = elem
                    print(f"Found source potentials at index {i}")
                    break
    elif isinstance(dual_sol, torch.Tensor):
        f_tensor = dual_sol

    if f_tensor is None:
        raise RuntimeError(f"Could not find source potentials. dual_sol structure: {dual_sol}")

    # Convert to numpy
    lava_values = f_tensor.detach().cpu().numpy().flatten()
    if len(lava_values) != training_size:
        raise ValueError(f"Source potentials have length {len(lava_values)}, expected {training_size}")

    # --- Per‑class statistics for debugging ---
    targets = np.array(train_dataset.targets)
    print("\n--- LAVA Debug: Per-Class Value Stats ---")
    for i in range(10):  # adjust for your number of classes
        class_idx = np.where(targets == i)[0]
        if len(class_idx) > 0:
            class_vals = lava_values[class_idx]
            print(f"Class {i} | Size: {len(class_idx):>4} | "
                  f"Mean Val: {class_vals.mean():.6f} | "
                  f"Max: {class_vals.max():.4f} | Min: {class_vals.min():.4f}")

    # --- Select top keep_ratio% (lowest scores = highest quality) ---
    selected_sample_size = int(training_size * keep_ratio)
    selected_indices = np.argsort(lava_values)[:selected_sample_size].tolist()
    print(f"Selected {len(selected_indices)} samples (keep_ratio = {keep_ratio})")
    return selected_indices