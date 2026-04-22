import sys
import os
from unittest.mock import MagicMock

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

lava_folder = os.path.join(project_root, 'LAVA')
if lava_folder not in sys.path:
    sys.path.insert(0, lava_folder)

import otdd
print("otdd location:", otdd.__file__)

from LAVA import lava
from LAVA.lava import compute_dual, PreActResNet18, values
print("Successfully imported LAVA.lava")

lib = ["torchtext", "torchtext.data", "torchtext.data.utils",
       "torchtext.datasets", "torchtext.vocab", "vgg", "resnet"]
for mod in lib:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

def get_saved_scores(file_key, training_size, recompute=False):
    """
    Load cached LAVA scores if available and match training size.
    Returns (scores, saved_file_path) or (None, None) if not found or recompute=True.
    """
    if recompute or file_key is None:
        return None, None
    cache_dir = 'lava_selection_results'
    os.makedirs(cache_dir, exist_ok=True)
    score_file = os.path.join(cache_dir, f"{file_key}_scores.npy")
    if os.path.exists(score_file):
        scores = np.load(score_file)
        if len(scores) == training_size:
            print(f"Loaded cached LAVA scores from {score_file}")
            return scores, score_file
        else:
            print(f"Cached scores length {len(scores)} != training size {training_size}. Recomputing.")
    return None, score_file 

def save_lava_scores(scores, score_file):
    """Save scores to the given file path."""
    np.save(score_file, scores)
    print(f"Saved LAVA scores to {score_file}")

# replace the original compute_dual with the new compute_dual_1 
def compute_dual_1(feature_extractor, trainloader, testloader, training_size, shuffle_ind, p=2, resize=32, device='cuda'):
    # Use sequential indices because the DataLoader returns samples in dataset order.
    train_indices = list(range(training_size))
    # not used in selection
    trained_with_flag = lava.train_with_corrupt_flag(trainloader, shuffle_ind, train_indices)

    # compute OT dual potentials
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

def get_feature_extractor(device, dataset_name):
    if dataset_name == "cifar10":
        print("Using PreActResnet18-Cifar10 as a feature extractor")
        model = PreActResNet18(num_classes=10)
        checkpoint = torch.load('models/cifar10_embedder_preact_resnet18.pth', map_location='cpu')
        model.load_state_dict(checkpoint)
        # remove the final linear layer
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model = model.to(device)
        model.eval()
        return model
    elif dataset_name == "cifar100":
        print("Using PreActResnet18-Cifar100 as a feature extractor")
        model = PreActResNet18()
        in_features = model.linear.in_features
        model.linear = nn.Linear(in_features, 100)
        checkpoint = torch.load('models/cifar100_embedder_preact_resnet18.pth', map_location='cpu')
        model.load_state_dict(checkpoint)
        # remove the final linear layer
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model = model.to(device)
        model.eval()
        return model

def get_lava_selection_indices(train_dataset, val_dataset, keep_ratio=0.7, device='cuda', file_key=None):
    # get training size
    training_size = len(train_dataset)
    print(f"[DEBUG] training_size = {training_size}", flush=True)
    # get the lava scores from the file
    lava_values, saved_file = get_saved_scores(file_key, training_size)
    print(f"[DEBUG] saved_file = {saved_file}", flush=True)

    # if no scores available
    if lava_values is None:
        # prepare the dataset, shuffle=False to map correctly scores to each sample
        print("[DEBUG] Preparing train_wrapper, val_wrapper...", flush=True)
        train_wrapper, val_wrapper = dataset_prep(train_dataset, val_dataset)
        print("[DEBUG] Creating DataLoaders...", flush=True)
        train_loader = DataLoader(train_wrapper, batch_size=128, shuffle=False, num_workers=4)
        val_loader = DataLoader(val_wrapper, batch_size=128, shuffle=False)

        # load the feature extractor
        if file_key is None:
            raise ValueError("file_key must be provided to determine dataset name")
        dataset_name = file_key.split('_')[0] 
        print(f"[DEBUG] Loading feature extractor for {dataset_name}...", flush=True)
        extractor = get_feature_extractor(device, dataset_name)
        print("[DEBUG] Feature extractor loaded.", flush=True)

        print(f"--- LAVA Selection Started ---")
        print(f"Total training samples to evaluate: {training_size}")

        # compute the scores
        print("[DEBUG] Calling lava.compute_dual (this may take a long time)...", flush=True)
        dual_sol, _ = lava.compute_dual(
            feature_extractor=extractor,
            trainloader=train_loader,
            testloader=val_loader,
            training_size=training_size,
            shuffle_ind=[], # passs in empty array
            device=device
        )
        print("[DEBUG] lava.compute_dual finished.", flush=True)

        # dual_sol[0] is f (source potentials)
        print("[DEBUG] Computing calibrated values...", flush=True)
        calibrated = values(dual_sol, training_size)
        lava_values = np.array(calibrated)
        print("[DEBUG] Calibration done.", flush=True)

        # save the computation
        if saved_file is not None:
            print(f"[DEBUG] Saving scores to {saved_file}...", flush=True)
            save_lava_scores(lava_values, saved_file)
            print("[DEBUG] Scores saved.", flush=True)
    else:
        print("Using cached LAVA scores.")

    # select lowest scores (best quality)
    selected_sample_size = int(training_size * keep_ratio)
    print(f"[DEBUG] Selecting top {selected_sample_size} samples...", flush=True)
    selected_indices = np.argsort(lava_values)[:selected_sample_size].tolist()
    print(f"Selected {len(selected_indices)} samples (keep_ratio = {keep_ratio})")

    return selected_indices