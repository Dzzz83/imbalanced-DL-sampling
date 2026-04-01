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

from LAVA.lava import compute_dual, compute_values_and_visualize, PreActResNet18


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
    model = PreActResNet18()
    checkpoint = torch.load('models/cifar10_embedder_preact_resnet18.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model = FeatureExtractor(model)
    model.eval()
    return model.to(device)

def select_indices(lava_values, training_size, keep_ratio):
    train_scores = lava_values[:training_size]          # only training duals
    selected_sample_size = int(len(train_scores) * keep_ratio)
    # sort ascending, take first selected_sample_size (smallest scores)
    selected_indices = np.argsort(train_scores)[:selected_sample_size]
    return selected_indices.tolist()

def get_lava_selection_indices(train_dataset, val_dataset, keep_ratio=0.7, device='cuda'):
    train_wrapper, val_wrapper = dataset_prep(train_dataset, val_dataset)
    train_loader = DataLoader(train_wrapper, batch_size=128, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_wrapper, batch_size=128, shuffle=False)

    extractor = get_feature_extractor(device)
    training_size = len(train_dataset)

    print(f"--- LAVA Selection Started ---")
    print(f"Total training samples to evaluate: {training_size}")

    dual_sol, _ = lava.compute_dual(
        feature_extractor=extractor,
        trainloader=train_loader,
        testloader=val_loader,
        training_size=training_size,
        shuffle_ind=[],
        device=device
    )

    # dual_sol[0] is f (source potentials)
    f = dual_sol[0]
    if isinstance(f, torch.Tensor):
        lava_values = f.detach().cpu().numpy().flatten()
    else:
        lava_values = np.array(f).flatten()

    if len(lava_values) != training_size:
        raise ValueError(f"Expected {training_size} scores, got {len(lava_values)}")

    # Select lowest scores (best quality)
    selected_sample_size = int(training_size * keep_ratio)
    selected_indices = np.argsort(lava_values)[:selected_sample_size].tolist()
    print(f"Selected {len(selected_indices)} samples (keep_ratio = {keep_ratio})")

    return selected_indices