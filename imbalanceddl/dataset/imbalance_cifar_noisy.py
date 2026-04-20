import torchvision
import numpy as np
from .imbalance_cifar import IMBALANCECIFAR10
from collections import Counter

class IMBALANCECIFAR10_NOISY(IMBALANCECIFAR10):
    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0,
                 train=True, transform=None, target_transform=None, download=False,
                 noise_ratio=0.25, num_classes=10, seed=42):
        super().__init__(root, imb_type, imb_factor, rand_number, train, transform, target_transform, download)
        self.noise_ratio = noise_ratio
        self.num_classes = num_classes
        self.rng = np.random.RandomState(seed if seed is not None else rand_number)
        self._add_label_noise()

    def _add_label_noise(self):
        # Print original class distribution
        original_counts = dict(Counter(self.targets))
        print(f"[NOISY] Original class distribution: {original_counts}")

        targets = np.array(self.targets)
        # get the number of intended noisy samples
        noisy_sample_num = int(len(targets) * self.noise_ratio)
        print(f"[NOISY] Flipping {noisy_sample_num} out of {len(targets)} labels (ratio={self.noise_ratio})")

        # get a list of random indexs to choose samples to inject noise without replacement
        noisy_idx = self.rng.choice(len(targets), noisy_sample_num, replace=False)
        for idx in noisy_idx:
            # get the current label
            current_idx = targets[idx]
            # get list of possible new labels
            possible_new_label = []
            for i in range(self.num_classes):
                if i != current_idx:
                    possible_new_label.append(i)
            # choose randomly the new label
            new_label = self.rng.choice(possible_new_label)
            targets[idx] = new_label
        self.targets = targets.tolist()

        # recompute per-class counts
        self.num_per_cls_dict = dict(Counter(self.targets))

        # Print new class distribution
        print(f"[NOISY] New class distribution: {self.num_per_cls_dict}")
        