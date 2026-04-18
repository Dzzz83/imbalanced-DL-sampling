# evaluate_lava_models.py
import sys
import os
from unittest.mock import MagicMock

# Mock problematic modules
lib = ["torchtext", "torchtext.data", "torchtext.data.utils",
       "torchtext.datasets", "torchtext.vocab", "vgg", "resnet"]
for mod in lib:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

lava_folder = os.path.join(project_root, 'LAVA')
if lava_folder not in sys.path:
    sys.path.insert(0, lava_folder)

import otdd
print("otdd location:", otdd.__file__)

from LAVA import lava
from LAVA.lava import PreActResNet18, values
print("Successfully imported LAVA.lava")

import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter
from torchvision import datasets

from imbalanceddl.dataset.imbalance_dataset import ImbalancedDataset
from imbalanceddl.utils.logging import setup_logger, create_distribution_table
from imbalanceddl.utils._augmentation import get_weak_augmentation

# ---------- Helper functions (copied from lava_selection) ----------
class OTDDWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        if hasattr(dataset, 'targets'):
            self.targets = dataset.targets
    def __getitem__(self, index):
        item = self.dataset[index]
        return item[0], item[1]
    def __len__(self):
        return len(self.dataset)

def dataset_prep(train_dataset, val_dataset):
    for ds in [train_dataset, val_dataset]:
        if hasattr(ds, 'targets') and not isinstance(ds.targets, torch.Tensor):
            ds.targets = torch.LongTensor(ds.targets)
    return OTDDWrapper(train_dataset), OTDDWrapper(val_dataset)

def get_custom_feature_extractor(device, checkpoint_path, num_classes=10):
    print(f"Loading feature extractor from {checkpoint_path}")
    model = PreActResNet18(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    # Remove the final linear layer to obtain a pure feature extractor
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model = model.to(device)
    model.eval()
    return model

def compute_lava_scores(train_dataset, val_dataset, feature_extractor, device='cuda'):
    """Compute LAVA scores without caching."""
    training_size = len(train_dataset)
    train_wrapper, val_wrapper = dataset_prep(train_dataset, val_dataset)
    train_loader = DataLoader(train_wrapper, batch_size=128, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_wrapper, batch_size=128, shuffle=False)

    # Override compute_dual to use sequential indices
    def compute_dual_1(feature_extractor, trainloader, testloader, training_size, shuffle_ind, p=2, resize=32, device='cuda'):
        train_indices = list(range(training_size))
        trained_with_flag = lava.train_with_corrupt_flag(trainloader, shuffle_ind, train_indices)
        dual_sol = lava.get_OT_dual_sol(
            feature_extractor, trainloader, testloader,
            training_size=training_size, p=p, resize=resize, device=device
        )
        return dual_sol, trained_with_flag

    dual_sol, _ = compute_dual_1(
        feature_extractor, train_loader, val_loader,
        training_size=training_size, shuffle_ind=[], device=device
    )
    calibrated = values(dual_sol, training_size)
    lava_values = np.array(calibrated)
    return lava_values

# ---------- Main ----------
def main():
    # Configuration – adjust for your dataset
    class Config:
        dataset = 'cifar10'          # CIFAR-10
        imb_type = 'exp'
        imb_factor = 0.01
        num_classes = 10
        rand_number = 0
        augmentation = 'none'        # LAVA scoring uses no augmentation
        root = './data'
        workers = 4
        gpu = 1                      # Use GPU 1 (change if needed)
    cfg = Config()

    # Load validation set (balanced CIFAR-10 test set)
    _, val_transform = get_weak_augmentation()
    val_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)

    # Load imbalanced training set (no augmentation, because LAVA scoring uses raw images)
    train_ds_raw = ImbalancedDataset(cfg, cfg.dataset, augmentation='none')
    train_dataset, _ = train_ds_raw.train_val_sets

    # Original class counts (for the distribution table)
    if hasattr(train_dataset, 'targets'):
        original_targets = train_dataset.targets
    else:
        original_targets = [train_dataset[i][1] for i in range(len(train_dataset))]
    original_counts = Counter(original_targets)
    all_classes = list(range(cfg.num_classes))
    orig_dict = {c: original_counts.get(c, 0) for c in all_classes}

    # Directory containing the trained models
    models_dir = "/home/phatht/phat/imbalanced-DL-sampling/models1"
    if not os.path.isdir(models_dir):
        print(f"Directory not found: {models_dir}")
        return

    # Find all .pth files in that directory
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    if not model_files:
        print(f"No .pth files found in {models_dir}")
        return

    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"
    keep_ratio = 0.7
    out_dir = 'lava_test'
    os.makedirs(out_dir, exist_ok=True)

    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        model_name = os.path.splitext(model_file)[0]   # remove .pth extension

        print(f"\n========== Evaluating {model_name} ==========")
        try:
            extractor = get_custom_feature_extractor(device, model_path, num_classes=10)

            # Compute LAVA scores (no caching)
            lava_values = compute_lava_scores(train_dataset, val_ds, extractor, device=device)

            # Select top 70% samples (lowest scores = best quality)
            selected_sample_size = int(len(train_dataset) * keep_ratio)
            indices = np.argsort(lava_values)[:selected_sample_size].tolist()

            # Compute selected class counts
            selected_targets = [original_targets[i] for i in indices]
            selected_counts = Counter(selected_targets)
            sel_dict = {c: selected_counts.get(c, 0) for c in all_classes}

            # Log distribution table
            log_path = os.path.join(out_dir, f"{model_name}_distribution.log")
            logger, _ = setup_logger(log_path)
            logger.info(f"Model: {model_name}\n")
            create_distribution_table(logger, orig_dict, sel_dict)
            logger.info("\n" + "="*60 + "\n")
            print(f"Distribution table saved to {log_path}")

        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue

if __name__ == "__main__":
    main()