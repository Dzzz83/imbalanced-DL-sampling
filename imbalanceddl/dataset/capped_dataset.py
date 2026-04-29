import numpy as np
from torch.utils.data import Dataset, Subset

class CappedDataset(Dataset):
    """
    Wraps an existing dataset and returns a random subset where each class
    is capped to a specified number of samples.
    """
    def __init__(self, dataset, cap_per_class, num_classes=None):
        """
        Args:
            dataset: A torch Dataset with a `targets` attribute (list or array of labels).
            cap_per_class: int or list of ints. If int, same cap for all classes.
            num_classes: Total number of classes (if not provided, inferred from dataset).
        """
        self.dataset = dataset
        if isinstance(cap_per_class, int):
            if num_classes is None:
                # Infer from unique targets
                targets = np.array(dataset.targets)
                num_classes = len(np.unique(targets))
            self.caps = [cap_per_class] * num_classes
        else:
            self.caps = cap_per_class

        # Compute indices to keep
        keep_indices = []
        targets = np.array(dataset.targets)
        for c, cap in enumerate(self.caps):
            idx = np.where(targets == c)[0]
            if len(idx) > cap:
                selected = np.random.choice(idx, cap, replace=False)
            else:
                selected = idx
            keep_indices.extend(selected)
        self.keep_indices = keep_indices
        self.subset = Subset(dataset, keep_indices)
        # Replicate targets for compatibility
        self.targets = [dataset.targets[i] for i in keep_indices]
        self.cls_num_list = [np.sum(np.array(self.targets) == c) for c in range(num_classes)]
        # debug print
        print(f"[CappedDataset] Original dataset size: {len(dataset)}")
        print(f"[CappedDataset] Capped dataset size: {len(self.keep_indices)}")
        print(f"[CappedDataset] New class distribution: {self.cls_num_list}")

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.subset[idx]

    def get_cls_num_list(self):
        return self.cls_num_list

    @property
    def train_val_sets(self):
        # For compatibility with ImbalancedDataset interface
        return self, None  # only training set; validation must be handled separately
