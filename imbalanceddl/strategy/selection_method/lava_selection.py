import sys
import os 
from unittest.mock import MagicMock
from torch.utils.data import Subset

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
import torch.nn.functional as F

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
        features = F.normalize(features, p=2, dim=1)
        return features
    
class OTDDWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        print(f"\n[DEBUG] OTDDWrapper: Probing {type(dataset)}")
        
        # Trace recursion for nested subsets
        def find_targets(ds, depth=0):
            indent = "  " * depth
            print(f"{indent}Level {depth}: {type(ds)}")
            if hasattr(ds, 'targets'):
                print(f"{indent}-> Found .targets at this level!")
                return ds.targets
            if hasattr(ds, 'dataset'):
                print(f"{indent}-> No .targets here, looking inside .dataset...")
                return find_targets(ds.dataset, depth + 1)
            return None

        targets = find_targets(dataset)

        if targets is not None:
            # If we found targets but are in a Subset, we must slice them
            curr = dataset
            final_indices = None
            while hasattr(curr, 'indices'):
                if final_indices is None:
                    final_indices = np.array(curr.indices)
                else:
                    final_indices = final_indices[np.array(curr.indices)]
                curr = curr.dataset
            
            if final_indices is not None:
                print(f"[DEBUG] Slicing targets with accumulated indices (Length: {len(final_indices)})")
                targets = np.array(targets)[final_indices]

            if not isinstance(targets, torch.Tensor):
                self.targets = torch.tensor(targets, dtype=torch.long)
            else:
                self.targets = targets.long()
            
            self.classes = torch.unique(self.targets).tolist()
            print(f"[DEBUG] Success: Found {len(self.targets)} targets and {len(self.classes)} classes.")
        else:
            # List all available attributes to see where they might be stored
            print(f"[ERROR] Could not find 'targets' in {type(dataset)} or its parents.")
            print(f"[DEBUG] Available attributes: {dir(dataset)}")
            raise ValueError("OTDDWrapper could not find targets in the provided dataset.")

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
    # 1. Create a SHUFFLED map of indices so OTDD sees all classes early
    original_indices = np.arange(len(train_dataset))
    shuffled_indices = np.copy(original_indices)
    np.random.seed(42)  
    np.random.shuffle(shuffled_indices)
    
    # 2. Create a Subset using these shuffled indices
    shuffled_train_dataset = Subset(train_dataset, shuffled_indices)
    
    if hasattr(train_dataset, 'targets'):
        # Map original targets to the new shuffled order
        shuffled_train_dataset.targets = torch.tensor(np.array(train_dataset.targets)[shuffled_indices])
            
    # 3. Wrap and load (Keep shuffle=False here, the Subset is already mixed!)
    train_wrapper, val_wrapper = dataset_prep(shuffled_train_dataset, val_dataset)
    train_loader = DataLoader(train_wrapper, batch_size=128, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_wrapper, batch_size=128, shuffle=False)

    # # prepare the val_dataset and train_dataset
    # train_wrapper, val_wrapper = dataset_prep(train_dataset, val_dataset)

    # # dataset_indices = getattr(train_dataset, 'indices', list(range(len(train_dataset))))
    # # train_sampler = SubsetRandomSampler(dataset_indices)

    # # create dataloader
    # train_loader = DataLoader(train_wrapper, batch_size=128, shuffle=False, num_workers=4)
    # val_loader = DataLoader(val_wrapper, batch_size=128, shuffle=False)

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

    # --- DEBUG FOR OTDD VALUEERROR ---
    print("\n--- OTDD Label Diagnostic ---")
    train_targets = []
    for _, target in train_loader:
        train_targets.append(target)
    train_targets = torch.cat(train_targets)
    unique_train = torch.unique(train_targets).tolist()
    
    val_targets = []
    for _, target in val_loader:
        val_targets.append(target)
    val_targets = torch.cat(val_targets)
    unique_val = torch.unique(val_targets).tolist()

    print(f"Unique Training Labels: {unique_train}")
    print(f"Unique Validation Labels: {unique_val}")
    print(f"Number of samples per class (Train): {torch.bincount(train_targets).tolist()}")
    print(f"Number of samples per class (Val): {torch.bincount(val_targets).tolist()}")
    
    if len(unique_train) != len(unique_val):
        print("CRITICAL: Label count mismatch between sets!")
    # ---------------------------------

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

    # 6. Extract results from the shuffled run
    if isinstance(dual_sol, (list, tuple)):
        lava_values_shuffled = dual_sol[0].detach().cpu().numpy().flatten()
    else:
        lava_values_shuffled = dual_sol.detach().cpu().numpy().flatten()
    
    # 5. RE-MAP the values back to original order before selection
    # This ensures lava_values[0] is the score for the 1st image in train_dataset
    lava_values_original_order = np.zeros_like(lava_values_shuffled)
    lava_values_original_order[shuffled_indices] = lava_values_shuffled
    

    # --- DEBUG START ---
    targets = np.array(train_dataset.targets) # Ensure this matches your train_dataset
    print("\n--- LAVA Debug: Per-Class Value Stats ---")
    for i in range(10): # For CIFAR-10
        class_idx = np.where(targets == i)[0]
        if len(class_idx) > 0:
            class_vals = lava_values_original_order[class_idx]
            print(f"Class {i} | Size: {len(class_idx):>4} | Mean Val: {class_vals.mean():.6f} | Max: {class_vals.max():.4f} | Min: {class_vals.min():.4f}")
    # --- DEBUG END ---

    # 6. Now perform selection on the correctly ordered values
    selected_indices = select_indices(lava_values_original_order, len(train_dataset), keep_ratio)

    return selected_indices

