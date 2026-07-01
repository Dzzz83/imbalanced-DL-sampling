import numpy as np
from torch.utils.data import Dataset, Subset

class CappedDataset(Dataset):
    def __init__(self, dataset, cap_per_class, num_classes=None):
        self.dataset = dataset
        if isinstance(cap_per_class, int):
            if num_classes is None:
                targets = np.array(dataset.targets)
                num_classes = len(np.unique(targets))
            self.caps = [cap_per_class] * num_classes
        else:
            self.caps = cap_per_class

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
        self.targets = [dataset.targets[i] for i in keep_indices]
        self.cls_num_list = [np.sum(np.array(self.targets) == c) for c in range(num_classes)]
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
        return self, None

    def get_class_idxs2(self):
        targets_np = np.array(self.targets, dtype=np.int64)
        class_idxs = []
        for c in range(len(self.caps)):
            idxs = np.where(targets_np == c)[0].tolist()
            class_idxs.append(idxs)
        return class_idxs

    def get_sample_weights(self):
        cls_counts = np.bincount(self.targets, minlength=len(self.caps))
        cls_counts = np.maximum(cls_counts, 1)
        total = len(self.targets)
        class_weights = total / (len(self.caps) * cls_counts)
        return [class_weights[t] for t in self.targets]

    def get_weights(self):
        cls_counts = np.bincount(self.targets, minlength=len(self.caps))
        cls_counts = np.maximum(cls_counts, 1)
        total = len(self.targets)
        class_weights = total / (len(self.caps) * cls_counts)
        return class_weights.tolist()