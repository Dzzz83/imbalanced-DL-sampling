import numpy as np
import random
import torchvision.transforms as transforms
from torch.utils.data import Subset
from .trainer import Trainer
from imbalanceddl.strategy.selection_method.lava_selection import get_lava_selection_indices
from imbalanceddl.utils.deep_smote_data_loader import CustomImageDataset, inject_label_noise
from imbalanceddl.utils._augmentation import get_weak_augmentation, get_trivial_augmentation
from imbalanceddl.strategy.build_trainer import build_trainer
from torchvision import datasets
from imbalanceddl.utils.key_generation import LavaCacheKey
from imbalanceddl.dataset.imbalance_cifar import IMBALANCECIFAR10
from imbalanceddl.dataset.capped_dataset import CappedDataset
import torch

class SelectionRandomOversamplingTrainer(Trainer):
    def __init__(self, cfg, dataset, model, strategy="Selection_RandomOversampling"):
        print("\n" + "="*60)
        print("SelectionRandomOversamplingTrainer Initialization")
        print("="*60)

        # Validation dataset
        _, val_transform = get_weak_augmentation()
        print(f"1. Loading validation dataset: {cfg.dataset}")
        if cfg.dataset == 'cifar10':
            val_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
        elif cfg.dataset == 'cifar100':
            val_ds = datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)
        else:
            raise NotImplementedError
        print(f"   Validation set size: {len(val_ds)}")

        # 2. Load original imbalanced dataset (clean base)
        print(f"\n2. Loading original imbalanced dataset for {cfg.dataset}, imb_type={cfg.imb_type}, imb_factor={cfg.imb_factor}")
        base_dataset = IMBALANCECIFAR10(
            root='./data',
            imb_type=cfg.imb_type,
            imb_factor=cfg.imb_factor,
            rand_number=cfg.rand_number,
            train=True,
            download=True,
            transform=None
        )
        X = base_dataset.data
        Y = np.array(base_dataset.targets).astype(int)
        print(f"[DEBUG] Loaded clean dataset: X.shape={X.shape}, Y.shape={Y.shape}")
        print(f"[DEBUG] Original class distribution: {dict(zip(*np.unique(Y, return_counts=True)))}")

        # 3. Inject label noise first (if configured)
        if hasattr(cfg, 'noise_ratio') and cfg.noise_ratio > 0:
            print(f"Applying {cfg.noise_ratio*100}% label noise to original dataset (before selection)")
            Y = inject_label_noise(Y, cfg.noise_ratio, cfg.num_classes, seed=cfg.rand_number)
            print(f"[DEBUG] After noise: class distribution: {dict(zip(*np.unique(Y, return_counts=True)))}")

        # 4. Create plain and augmented datasets from the (noisy) imbalanced data
        plain_transform = val_transform
        plain_dataset = CustomImageDataset(X, Y, transform=plain_transform)
        print(f"\n3. Plain dataset (for scoring) created with {len(plain_dataset)} samples (imbalanced)")
        print(f"   Transform: ToTensor + Normalize (no augmentation)")
        print(f"[DEBUG] Plain dataset class distribution: {dict(zip(*np.unique(plain_dataset.Y, return_counts=True)))}")

        # Determine training transform
        print(f"\n4. Training transform: cfg.augmentation = {cfg.augmentation}")
        if cfg.augmentation == 'weak':
            train_transform, _ = get_weak_augmentation()
            print("   Using weak augmentation (RandomCrop + RandomHorizontalFlip)")
        elif cfg.augmentation == 'trivial':
            train_transform, _ = get_trivial_augmentation()
            print("   Using trivial augmentation (only ToTensor + Normalize)")
        elif cfg.augmentation == 'none':
            if cfg.dataset == 'cifar10':
                normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            elif cfg.dataset == 'cifar100':
                normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            else:
                raise NotImplementedError(f"Normalization for {cfg.dataset} not defined")
            train_transform = transforms.Compose([transforms.ToTensor(), normalize])
            print("   Using no augmentation (only ToTensor + Normalize)")
        else:
            raise NotImplementedError(f"Augmentation {cfg.augmentation} not supported")

        aug_dataset = CustomImageDataset(X, Y, transform=train_transform)
        original_cls_num_list = aug_dataset.get_cls_num_list()
        cfg.original_cls_num_list = original_cls_num_list
        print(f"\n5. Augmented dataset (for training) created with {len(aug_dataset)} samples (imbalanced)")

        # 5. Apply selection (LAVA or random) on the plain dataset
        print(f"\n6. Selection: method={cfg.selection_method}, ratio={cfg.selection_ratio}")
        if cfg.selection_ratio < 1.0:
            if cfg.selection_method == 'lava':
                print("   Computing LAVA scores...")
                is_noisy = hasattr(cfg, 'noise_ratio') and cfg.noise_ratio > 0
                # Cache key: selection first, oversampled later, no noise_first
                key_gen = LavaCacheKey(config=cfg, is_deepsmote=False, is_noisy=is_noisy,
                                       is_oversampled=False, is_selection_first=True)
                file_key = key_gen.generate()
                print(f"[DEBUG] LAVA file_key = {file_key}")
                indices = get_lava_selection_indices(
                    plain_dataset,
                    val_ds,
                    keep_ratio=cfg.selection_ratio,
                    device=cfg.device,
                    file_key=file_key
                )
                print(f"[DEBUG] Selected {len(indices)} indices")
                selected_targets = [plain_dataset.Y[i] for i in indices]
                unique, counts = np.unique(selected_targets, return_counts=True)
                print(f"[DEBUG] Selected class distribution (from plain dataset): {dict(zip(unique, counts))}")
                print(f"   LAVA selection completed. Kept {len(indices)} indices.")
            elif cfg.selection_method == 'random':
                print("   Randomly selecting samples...")
                total = len(plain_dataset)
                n_keep = int(total * cfg.selection_ratio)
                indices = random.sample(range(total), n_keep)
                print(f"[DEBUG] Randomly selected {len(indices)} indices")
                selected_targets = [plain_dataset.Y[i] for i in indices]
                unique, counts = np.unique(selected_targets, return_counts=True)
                print(f"[DEBUG] Random selection class distribution: {dict(zip(unique, counts))}")
                print(f"   Random selection kept {len(indices)} samples out of {total}")
            else:
                raise ValueError(f"Unknown selection_method: {cfg.selection_method}")

            # Create subset of augmented dataset from selected indices
            selected_subset = Subset(aug_dataset, indices)
            selected_targets = [aug_dataset.Y[i] for i in indices]
            selected_counts = np.bincount(selected_targets, minlength=cfg.num_classes).tolist()
            cfg.selected_cls_num_list = selected_counts   # store for logger
            print(f"   Selected subset class distribution: {dict(enumerate(selected_counts))}")
            # Now oversample the selected subset to cap_per_class
            if not hasattr(cfg, 'cap_per_class') or cfg.cap_per_class is None:
                raise ValueError("For Selection_RandomOversampling, cap_per_class must be specified.")
            target_per_class = cfg.cap_per_class
            print(f"   Oversampling selected subset to {target_per_class} samples per class...")

            # Extract targets from the subset
            subset_targets = [aug_dataset.Y[i] for i in indices]
            subset_targets_np = np.array(subset_targets)

            # Compute oversampled indices (duplicate with replacement)
            oversampled_indices = []
            for c in range(cfg.num_classes):
                idx = np.where(subset_targets_np == c)[0]
                n = len(idx)
                if n == 0:
                    # If a class is completely absent, cannot generate samples.
                    print(f"[WARNING] Class {c} has no samples after selection; no oversampling possible.")
                    continue
                # Randomly choose indices (with replacement) to reach target_per_class
                chosen = np.random.choice(idx, size=target_per_class, replace=True)
                oversampled_indices.extend(chosen)
            # Map back to original subset indices
            final_indices = [indices[i] for i in oversampled_indices]
            final_train = Subset(aug_dataset, final_indices)
            print(f"   Oversampling complete. Final training set size: {len(final_train)}")
        else:
            # No selection: use full dataset (imbalanced) – then oversample to cap_per_class
            final_train = aug_dataset
            print(f"\n7. Final training set: all {len(final_train)} samples (no selection)")
            selected_counts = np.bincount(aug_dataset.Y, minlength=cfg.num_classes).tolist()
            cfg.selected_cls_num_list = selected_counts
            if hasattr(cfg, 'cap_per_class') and cfg.cap_per_class is not None:
                target_per_class = cfg.cap_per_class
                print(f"   Oversampling full dataset to {target_per_class} samples per class...")
                full_targets = aug_dataset.Y
                oversampled_indices = []
                for c in range(cfg.num_classes):
                    idx = np.where(full_targets == c)[0]
                    n = len(idx)
                    if n == 0:
                        continue
                    chosen = np.random.choice(idx, size=target_per_class, replace=True)
                    oversampled_indices.extend(chosen)
                final_train = Subset(aug_dataset, oversampled_indices)
                print(f"   Oversampling complete. Final training set size: {len(final_train)}")
            else:
                raise ValueError("For Selection_RandomOversampling, cap_per_class must be specified even for ratio=1.0.")

        # 8. Wrap for inner trainer
        class SimpleWrapper:
            def __init__(self, train, val, cfg):
                self.train_val_sets = (train, val)
                self.cfg = cfg
                # Correctly get targets from a Subset (oversampled set)
                if hasattr(train, 'dataset') and hasattr(train, 'indices'):
                    targets = np.array(train.dataset.Y)[train.indices]
                else:
                    targets = train.Y
                self.cls_num_list = np.bincount(targets, minlength=cfg.num_classes).tolist()
                print(f"   Wrapper class counts: {self.cls_num_list}")
        wrapper = SimpleWrapper(final_train, val_ds, cfg)
        cfg.cls_num_list = wrapper.cls_num_list

        # 9. Inner trainer
        base_strategy = getattr(cfg, 'base_strategy', 'ERM')
        print(f"\n8. Building inner trainer with base_strategy={base_strategy}")
        self.inner_trainer = build_trainer(cfg, wrapper, model, base_strategy)
        print("   Inner trainer initialized successfully")

        # Delegate attributes
        self.cfg = cfg
        self.model = model
        self.epoch = 0
        self.best_acc1 = 0
        self.train_loader = self.inner_trainer.train_loader
        self.val_loader = self.inner_trainer.val_loader
        self.optimizer = self.inner_trainer.optimizer
        self.logger = self.inner_trainer.logger
        self.log_training = self.inner_trainer.log_training
        self.log_testing = self.inner_trainer.log_testing
        self.tf_writer = self.inner_trainer.tf_writer

        print("="*60)
        print("SelectionRandomOversamplingTrainer initialization complete.\n")
        print("="*60)

    def get_criterion(self):
        return self.inner_trainer.get_criterion()

    def train_one_epoch(self):
        self.inner_trainer.train_one_epoch()

    def do_train_val(self):
        self.inner_trainer.do_train_val()

    def validate(self):
        return self.inner_trainer.validate()