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
    """
    Implements LAVA Calibration (Eq. 1) and Gradient-based selection.
    """
    # 1. Extract dual variables for training set
    f_star = lava_values[:training_size]
    N = len(f_star)
    
    # 2. Calibrate Gradients: f_i - [avg of all other f_j]
    # Mathematically equivalent to: f_i - (Total_Sum - f_i) / (N - 1)
    total_sum = np.sum(f_star)
    calibrated_gradients = (f_star - (total_sum - f_star) / (N - 1))
    
    # 3. Selection Strategy (Section 3.1):
    # To reduce OT distance (make training look like validation),
    # we discard samples with LARGE POSITIVE gradients.
    # Therefore, we sort ASCENDING and keep the smallest/most negative.
    num_to_keep = int(N * keep_ratio)
    selected_indices = np.argsort(calibrated_gradients)[:num_to_keep]
    
    return selected_indices.tolist()

def get_lava_selection_indices(train_dataset, val_dataset, keep_ratio=0.7, device='cuda'):
    # prepare the val_dataset and train_dataset
    train_wrapper, val_wrapper = dataset_prep(train_dataset, val_dataset)

    # dataset_indices = getattr(train_dataset, 'indices', list(range(len(train_dataset))))
    # train_sampler = SubsetRandomSampler(dataset_indices)

    # create dataloader
    train_loader = DataLoader(train_wrapper, batch_size=128, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_wrapper, batch_size=128, shuffle=False)

    # get feature extractor
    extractor = get_feature_extractor(device)
    # get training size
    training_size = len(train_dataset)

    print(f"--- LAVA Selection Started ---")
    print(f"Total training samples to evaluate: {training_size}")

    # Diagnostic check in get_lava_selection_indices
    train_classes = set(np.array(train_dataset.targets))
    val_classes = set(np.array(val_dataset.targets))

    print(f"Classes in Training: {len(train_classes)}")
    print(f"Classes in Validation: {len(val_classes)}")

    if train_classes != val_classes:
        raise ValueError(f"Mismatch! Train has {train_classes}, but Val has {val_classes}. OTDD requires both to have the same labels.")
    # calculate OT score
    dual_sol, _ = lava.compute_dual(
        feature_extractor=extractor,
        trainloader=train_loader,
        testloader=val_loader,
        training_size=training_size,
        shuffle_ind=[],
        device=device
    )

    if isinstance(dual_sol, (list, tuple)):
        # LAVA returns a list where the first element is usually the training duals
        lava_values = dual_sol[0].detach().cpu().numpy().flatten()
    else:
        lava_values = dual_sol.detach().cpu().numpy().flatten()

    # --- DEBUG START ---
    targets = np.array(train_dataset.targets) # Ensure this matches your train_dataset
    print("\n--- LAVA Debug: Per-Class Value Stats ---")
    for i in range(10): # For CIFAR-10
        class_idx = np.where(targets == i)[0]
        if len(class_idx) > 0:
            class_vals = lava_values[class_idx]
            print(f"Class {i} | Size: {len(class_idx):>4} | Mean Val: {class_vals.mean():.6f} | Max: {class_vals.max():.4f} | Min: {class_vals.min():.4f}")
    # --- DEBUG END ---

    selected_indices = select_indices(lava_values, training_size, keep_ratio)

    return selected_indices

