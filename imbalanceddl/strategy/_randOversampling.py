import numpy as np
import random
import torchvision.transforms as transforms
from torch.utils.data import Subset
from .trainer import Trainer

# Import SAVA method
from imbalanceddl.strategy.selection_method.sava_selection import get_sava_selection_indices

from imbalanceddl.utils.deep_smote_data_loader import CustomImageDataset, inject_label_noise
from imbalanceddl.utils._augmentation import get_weak_augmentation, get_trivial_augmentation
from imbalanceddl.strategy.build_trainer import build_trainer
from torchvision import datasets

# Import cache key generator
from imbalanceddl.utils.sava_key_generation import SavaCacheKey

from imbalanceddl.dataset.imbalance_cifar import IMBALANCECIFAR10
import torch

class RandomOversamplingTrainer(Trainer):
    def __init__(self, cfg, dataset, model, strategy="RandomOversampling_Selection"):
        print("\n" + "="*60)
        print("RandomOversamplingSelectionTrainer Initialization (No Capping)")
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

        # 2. Load original clean imbalanced dataset (always clean as base)
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
        X = base_dataset.data          # numpy array (N, H, W, C) uint8
        Y = np.array(base_dataset.targets).astype(int)
        print(f"[DEBUG] Loaded clean dataset: X.shape={X.shape}, Y.shape={Y.shape}")
        print(f"[DEBUG] Original class distribution: {dict(zip(*np.unique(Y, return_counts=True)))}")

        # Determine the pipeline order based on cfg.noise_first
        noise_first = hasattr(cfg, 'noise_first') and cfg.noise_first
        print(f"[DEBUG] noise_first = {noise_first}")

        if noise_first:
            # ------------------------------------------------------------
            # Pipeline 2: Noise → Oversample (No Capping)
            # ------------------------------------------------------------
            clean_counts = np.bincount(Y, minlength=cfg.num_classes)
            orig_majority = max(clean_counts)

            # 3a. Inject noise first (if requested)
            if hasattr(cfg, 'noise_ratio') and cfg.noise_ratio > 0:
                print(f"Applying {cfg.noise_ratio*100}% label noise to original dataset (before oversampling)")
                Y = inject_label_noise(Y, cfg.noise_ratio, cfg.num_classes, seed=cfg.rand_number)
                print(f"[DEBUG] After noise injection: class distribution: {dict(zip(*np.unique(Y, return_counts=True)))}")

            majority_count = orig_majority
            print(f"Majority class size (original): {majority_count}")

            # 5a. Random oversample each class to majority_count (with replacement)
            print("[DEBUG] Starting random oversampling...")
            oversampled_indices = []
            for c in range(cfg.num_classes):
                idx = np.where(Y == c)[0]
                if len(idx) == 0:
                    continue
                chosen = np.random.choice(idx, size=majority_count, replace=True)
                oversampled_indices.extend(chosen)
            oversampled_indices = np.array(oversampled_indices)
            X_bal = X[oversampled_indices]
            Y_bal = Y[oversampled_indices]
            print(f"[DEBUG] Oversampled dataset size: X_bal.shape={X_bal.shape}, Y_bal.shape={Y_bal.shape}")
            print(f"[DEBUG] Oversampled class distribution: {dict(zip(*np.unique(Y_bal, return_counts=True)))}")

        else:
            # ------------------------------------------------------------
            # Pipeline 1: Oversample → Noise (No Capping)
            # ------------------------------------------------------------
            # 3b. Compute majority count from original clean labels
            original_counts = np.bincount(Y, minlength=cfg.num_classes)
            majority_count = max(original_counts)
            print(f"Original class distribution: {dict(enumerate(original_counts))}")
            print(f"Majority class size: {majority_count}")

            # 4b. Random oversample each class to majority_count (with replacement)
            print("[DEBUG] Starting random oversampling...")
            oversampled_indices = []
            for c in range(cfg.num_classes):
                idx = np.where(Y == c)[0]
                if len(idx) == 0:
                    continue
                chosen = np.random.choice(idx, size=majority_count, replace=True)
                oversampled_indices.extend(chosen)
            oversampled_indices = np.array(oversampled_indices)
            X_bal = X[oversampled_indices]
            Y_bal = Y[oversampled_indices]
            print(f"[DEBUG] Oversampled dataset size: X_bal.shape={X_bal.shape}, Y_bal.shape={Y_bal.shape}")
            print(f"[DEBUG] Oversampled class distribution: {dict(zip(*np.unique(Y_bal, return_counts=True)))}")

            # 6b. Inject label noise after oversampling (if configured)
            if hasattr(cfg, 'noise_ratio') and cfg.noise_ratio > 0:
                print(f"Applying {cfg.noise_ratio*100}% label noise to balanced dataset")
                Y_bal = inject_label_noise(Y_bal, cfg.noise_ratio, cfg.num_classes, seed=cfg.rand_number)
                print(f"[DEBUG] After noise injection: class distribution: {dict(zip(*np.unique(Y_bal, return_counts=True)))}")

        # 7. Create plain dataset (no augmentation, only normalization)
        plain_transform = val_transform   # ToTensor + Normalize
        plain_dataset = CustomImageDataset(X_bal, Y_bal, transform=plain_transform)
        print(f"\n3. Plain dataset (for scoring) created with {len(plain_dataset)} samples")
        print(f"   Transform: ToTensor + Normalize (no augmentation)")
        print(f"[DEBUG] Plain dataset class distribution: {dict(zip(*np.unique(plain_dataset.Y, return_counts=True)))}")

        # 8. Determine training transform
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

        # Create augmented dataset (with augmentation)
        aug_dataset = CustomImageDataset(X_bal, Y_bal, transform=train_transform)
        original_cls_num_list = aug_dataset.get_cls_num_list()
        cfg.original_cls_num_list = original_cls_num_list
        print(f"\n5. Augmented dataset (for training) created with {len(aug_dataset)} samples")
        print(f"[DEBUG] Augmented dataset class distribution: {dict(zip(*np.unique(aug_dataset.Y, return_counts=True)))}")

        # 9. Apply selection (SAVA or random) on the plain dataset
        print(f"\n6. Selection: method={cfg.selection_method}, ratio={cfg.selection_ratio}")
        if cfg.selection_ratio < 1.0:
            is_noisy = hasattr(cfg, 'noise_ratio') and cfg.noise_ratio > 0
            is_noise_first = noise_first and is_noisy

            # --- SUB-ROUTE A: SAVA CALCULATOR ---
            if cfg.selection_method == 'sava':
                print("   Computing SAVA scores...")
                key_gen = SavaCacheKey(config=cfg, is_deepsmote=False, is_noisy=is_noisy,
                                       is_oversampled=True, is_noise_first=is_noise_first)
                file_key = key_gen.generate()
                print(f"[DEBUG] SAVA file_key = {file_key}")
                
                # Fetch SAVA arguments from config or safely assign expected backend defaults
                sava_batch_size = getattr(cfg, 'sava_batch_size', 1024)
                
                indices = get_sava_selection_indices(
                    train_dataset=plain_dataset,
                    val_dataset=val_ds,
                    keep_ratio=cfg.selection_ratio,
                    device=cfg.device,
                    file_key=file_key,
                    batch_size=sava_batch_size,
                    num_classes=cfg.num_classes
                )
                print(f"[DEBUG] Selected {len(indices)} indices via SAVA")

            # --- SUB-ROUTE B: RANDOM SELECTION ---
            elif cfg.selection_method == 'random':
                print("   Randomly selecting samples...")
                total = len(plain_dataset)
                n_keep = int(total * cfg.selection_ratio)
                indices = random.sample(range(total), n_keep)
                print(f"[DEBUG] Randomly selected {len(indices)} indices")
            
            else:
                raise ValueError(f"Unknown selection_method: {cfg.selection_method}")

            # Print tracking distributions for diagnostic monitoring
            selected_targets = [plain_dataset.Y[i] for i in indices]
            unique, counts = np.unique(selected_targets, return_counts=True)
            print(f"[DEBUG] Selected class distribution: {dict(zip(unique, counts))}")
            print(f"   Selection completed. Kept {len(indices)} indices out of {len(plain_dataset)}.")
            
            final_train = Subset(aug_dataset, indices)
            print(f"\n7. Final training set: {len(final_train)} samples (selected subset)")
        else:
            final_train = aug_dataset
            print(f"\n7. Final training set: all {len(final_train)} samples (no selection)")

        # 10. Wrap for inner trainer
        class SimpleWrapper:
            def __init__(self, train, val, cfg):
                self.train_val_sets = (train, val)
                self.cfg = cfg
                if hasattr(train, 'indices'):
                    # It's a Subset! Only count the labels of the SELECTED indices.
                    targets = [train.dataset.Y[i] for i in train.indices]
                elif hasattr(train, 'dataset'):
                    targets = train.dataset.Y
                else:
                    targets = train.Y
                self.cls_num_list = np.bincount(targets, minlength=cfg.num_classes).tolist()
                print(f"   Wrapper class counts: {self.cls_num_list}")
        wrapper = SimpleWrapper(final_train, val_ds, cfg)
        cfg.cls_num_list = wrapper.cls_num_list

        # 11. Inner trainer
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
        print("RandomOversamplingSelectionTrainer initialization complete.\n")
        print("="*60)

    def get_criterion(self):
        return self.inner_trainer.get_criterion()

    def train_one_epoch(self):
        self.inner_trainer.train_one_epoch()

    def do_train_val(self):
        self.inner_trainer.do_train_val()

    def validate(self):
        return self.inner_trainer.validate()