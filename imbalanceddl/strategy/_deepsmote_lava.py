import numpy as np
import random
import torchvision.transforms as transforms
from torch.utils.data import Subset
from .trainer import Trainer
from imbalanceddl.strategy.selection_method.lava_selection import get_lava_selection_indices
from imbalanceddl.utils.deep_smote_data_loader import (
    CustomImageDataset,
    load_and_cap_deepsmote,
    load_deepsmote_raw,
    inject_label_noise
)
from imbalanceddl.utils._augmentation import get_weak_augmentation, get_trivial_augmentation
from imbalanceddl.strategy.build_trainer import build_trainer
from torchvision import datasets
from imbalanceddl.utils.key_generation import LavaCacheKey
import torch

class DeepSMOTESelectionTrainer(Trainer):
    def __init__(self, cfg, dataset, model, strategy="DeepSMOTELAVA"):
        print("\n" + "="*60)
        print("DeepSMOTESelectionTrainer Initialization")
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

        # Determine pipeline order
        noise_first = hasattr(cfg, 'noise_first') and cfg.noise_first
        print(f"   noise_first = {noise_first}")

        # ============================================================
        # Load and prepare DeepSMOTE data (raw or capped)
        # ============================================================
        if noise_first:
            # Pipeline A: Load raw (5000 per class) → inject noise → cap
            print(f"\n2. Loading raw DeepSMOTE data for {cfg.dataset}, imb_type={cfg.imb_type}, imb_factor={cfg.imb_factor}")
            X_raw, Y_raw = load_deepsmote_raw(cfg.dataset, cfg.imb_type, cfg.imb_factor)
            print(f"[VERIFY] Raw data shape: X={X_raw.shape}, Y={Y_raw.shape}")
            print(f"[VERIFY] Raw class distribution: {dict(zip(*np.unique(Y_raw, return_counts=True)))}")

            # Inject noise first (if configured)
            if hasattr(cfg, 'noise_ratio') and cfg.noise_ratio > 0:
                print(f"Applying {cfg.noise_ratio*100}% label noise to raw DeepSMOTE data (before capping)")
                Y_raw = inject_label_noise(Y_raw, cfg.noise_ratio, cfg.num_classes, seed=cfg.rand_number)
                print(f"[VERIFY] After noise injection: class distribution: {dict(zip(*np.unique(Y_raw, return_counts=True)))}")

            # Then cap (if cap_per_class is set)
            X_capped = X_raw
            Y_capped = Y_raw
            if hasattr(cfg, 'cap_per_class') and cfg.cap_per_class is not None:
                print(f"Capping dataset to {cfg.cap_per_class} samples per class")
                from imbalanceddl.dataset.capped_dataset import CappedDataset
                temp_dataset = CustomImageDataset(X_capped, Y_capped, transform=None)
                capped_dataset = CappedDataset(temp_dataset, cfg.cap_per_class, num_classes=cfg.num_classes)
                subset_indices = capped_dataset.keep_indices
                X_capped = X_capped[subset_indices]
                Y_capped = Y_capped[subset_indices]
                print(f"[VERIFY] After capping: X.shape={X_capped.shape}, Y.shape={Y_capped.shape}")
                print(f"[VERIFY] Capped class distribution: {dict(zip(*np.unique(Y_capped, return_counts=True)))}")
            else:
                print("[DEBUG] No capping applied, using raw data (50000 samples)")

        else:
            # Pipeline B (original order): cap → noise
            print(f"\n2. Loading DeepSMOTE data for {cfg.dataset}, imb_type={cfg.imb_type}, imb_factor={cfg.imb_factor}")
            X_capped, Y_capped = load_and_cap_deepsmote(
                dataset=cfg.dataset,
                imb_type=cfg.imb_type,
                imb_factor=cfg.imb_factor,
                class_caps=None  # Uses default caps (e.g., [4000]*10)
            )
            print(f"[VERIFY] After capping: X.shape={X_capped.shape}, Y.shape={Y_capped.shape}")
            print(f"[VERIFY] Capped class distribution: {dict(zip(*np.unique(Y_capped, return_counts=True)))}")

            # Inject noise after capping
            if hasattr(cfg, 'noise_ratio') and cfg.noise_ratio > 0:
                print(f"Applying {cfg.noise_ratio*100}% label noise to capped dataset")
                Y_capped = inject_label_noise(Y_capped, cfg.noise_ratio, cfg.num_classes, seed=cfg.rand_number)
                print(f"[VERIFY] After noise injection: class distribution: {dict(zip(*np.unique(Y_capped, return_counts=True)))}")

        print(f"   Final data shape: X={X_capped.shape}, Y={Y_capped.shape}")
        unique, counts = np.unique(Y_capped, return_counts=True)
        print(f"   Final class distribution: {dict(zip(unique, counts))}")

        # ============================================================
        # Create plain and augmented datasets (same as before)
        # ============================================================
        plain_transform = val_transform   # ToTensor + Normalize
        plain_dataset = CustomImageDataset(X_capped, Y_capped, transform=plain_transform)
        print(f"\n3. Plain dataset (for scoring) created with {len(plain_dataset)} samples")
        print(f"   Transform: ToTensor + Normalize (no augmentation)")
        # Note: CustomImageDataset already prints its class distribution on init

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

        # Create augmented dataset
        aug_dataset = CustomImageDataset(X_capped, Y_capped, transform=train_transform)
        original_cls_num_list = aug_dataset.get_cls_num_list()
        cfg.original_cls_num_list = original_cls_num_list
        print(f"\n5. Augmented dataset (for training) created with {len(aug_dataset)} samples")
        # Again, CustomImageDataset prints its distribution

        # ============================================================
        # Apply selection (LAVA or random)
        # ============================================================
        print(f"\n6. Selection: method={cfg.selection_method}, ratio={cfg.selection_ratio}")
        if cfg.selection_ratio < 1.0:
            if cfg.selection_method == 'lava':
                print("   Computing LAVA scores...")
                is_noisy = hasattr(cfg, 'noise_ratio') and cfg.noise_ratio > 0
                # Include noise_first in cache key
                key_gen = LavaCacheKey(config=cfg, is_deepsmote=True, is_noisy=is_noisy, is_noise_first=noise_first)
                file_key = key_gen.generate()
                print(f"[DEBUG] LAVA cache key: {file_key}")
                indices = get_lava_selection_indices(
                    plain_dataset,
                    val_ds,
                    keep_ratio=cfg.selection_ratio,
                    device=cfg.device,
                    file_key=file_key
                )
                print(f"   LAVA selection completed. Kept {len(indices)} indices.")
                # Show selected class distribution from plain_dataset
                selected_targets = [plain_dataset.Y[i] for i in indices]
                unique, counts = np.unique(selected_targets, return_counts=True)
                print(f"[DEBUG] Selected class distribution (from plain dataset): {dict(zip(unique, counts))}")
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
            final_train = Subset(aug_dataset, indices)
            print(f"\n7. Final training set: {len(final_train)} samples (selected subset)")
            # Optionally, verify selected distribution from the subset
            subset_targets = [aug_dataset.Y[i] for i in indices]
            unique, counts = np.unique(subset_targets, return_counts=True)
            print(f"[VERIFY] Final selected class distribution: {dict(zip(unique, counts))}")
        else:
            final_train = aug_dataset
            print(f"\n7. Final training set: all {len(final_train)} samples (no selection)")

        # ============================================================
        # Wrapper, inner trainer, delegation (unchanged)
        # ============================================================
        class SimpleWrapper:
            def __init__(self, train, val, cfg):
                self.train_val_sets = (train, val)
                self.cfg = cfg
                if hasattr(train, 'dataset'):
                    targets = train.dataset.Y
                else:
                    targets = train.Y
                self.cls_num_list = np.bincount(targets, minlength=cfg.num_classes).tolist()
                print(f"   Wrapper class counts: {self.cls_num_list}")
        wrapper = SimpleWrapper(final_train, val_ds, cfg)
        cfg.cls_num_list = wrapper.cls_num_list

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
        print("DeepSMOTESelectionTrainer initialization complete.\n")
        print("="*60)

    def get_criterion(self):
        return self.inner_trainer.get_criterion()

    def train_one_epoch(self):
        self.inner_trainer.train_one_epoch()

    def do_train_val(self):
        self.inner_trainer.do_train_val()

    def validate(self):
        return self.inner_trainer.validate()