import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

SAVA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../sava')

if SAVA_ROOT not in sys.path:
    sys.path.insert(0, SAVA_ROOT)
otdd_path = os.path.join(SAVA_ROOT, 'otdd')
if otdd_path not in sys.path:
    sys.path.insert(0, otdd_path)
modules_to_remove = [m for m in sys.modules if m.startswith('otdd')]
for m in modules_to_remove:
    del sys.modules[m]
models_path = os.path.join(SAVA_ROOT, 'models')
if models_path not in sys.path:
    sys.path.insert(0, models_path)

from api import hierarchical_ot_experiment
from preact_resnet import PreActResNet18

def get_sava_sorted_indices(train_dataset, val_dataset, device='cuda',
                            batch_size=1024, num_classes=10, feat_repr=True,
                            parallel=False, cuda_num=0, n_gpu=1, resize=32,
                            cache_label_distances=True, model_path=None,
                            corrupt_por=0.01):
    """Compute SAVA scores and return training indices sorted by increasing value."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    training_size = len(train_dataset)

    if feat_repr:
        if model_path is None:
            if num_classes == 10:
                model_path = '/home/phatht/phat/imbalanced-DL-sampling/sava/checkpoint/cifar10_embedder_preact_resnet18.pth'
            else:
                model_path = '/home/phatht/phat/imbalanced-DL-sampling/sava/checkpoint/cifar100_embedder_preact_resnet18.pth'
        print(f"Loading feature extractor from: {model_path}")
        model = PreActResNet18(num_classes=num_classes)
        model = model.to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.linear = torch.nn.Identity()
        model.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 32, 32).to(device)
            out = model(dummy)
        print(f"Feature dimension: {out.shape[-1]}")
    else:
        print("WARNING: feat_repr=False uses raw pixels (slow).")
        class IdentityExtractor(torch.nn.Module):
            def forward(self, x):
                return x
        model = IdentityExtractor().to(device)
        model.eval()

    # Create non‑empty shuffle_ind (simulate a small corruption)
    n_corrupt = int(training_size * corrupt_por)
    if n_corrupt > 0:
        shuffle_ind = list(range(n_corrupt))
        print(f"Using shuffle_ind with {len(shuffle_ind)} samples (corrupt_por={corrupt_por})")
    else:
        shuffle_ind = []
        print("Warning: shuffle_ind is empty (corrupt_por=0)")

    print(f"DEBUG: feat_repr = {feat_repr}, model_path = {model_path}")
    result = hierarchical_ot_experiment(
        feature_extractor=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_size=training_size,
        batch_size=batch_size,
        shuffle_ind=shuffle_ind,
        resize=resize,
        portion=0.0,                     # <--- Changed from None
        device=device,
        cache_label_distances=cache_label_distances,
        visualise_hot=False,
        tag="",
        feat_repr=feat_repr,
        num_classes=num_classes,
        parallel=parallel,
        cuda_num=cuda_num,
        n_gpu=n_gpu,
    )
        # Convert to a 1D integer array
    if isinstance(result, tuple):
        sorted_indices = result[0]   # first element is the sorted indices
    else:
        sorted_indices = result

        # Convert to a 1D integer array
    if isinstance(sorted_indices, list):
        if len(sorted_indices) > 0 and hasattr(sorted_indices[0], '__len__') and len(sorted_indices[0]) == 1:
            sorted_indices = np.array([int(x[0]) for x in sorted_indices], dtype=int)
        else:
            sorted_indices = np.array(sorted_indices, dtype=int)
    else:
        sorted_indices = np.asarray(sorted_indices).ravel().astype(int)

    print(f"Converted sorted_indices shape: {sorted_indices.shape}, dtype: {sorted_indices.dtype}")
    return sorted_indices