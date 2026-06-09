import numpy as np
import random
import torchvision.transforms as transforms
from torch.utils.data import Subset
from .trainer import Trainer
from imbalanceddl.strategy.selection_method.sava_selection import get_sava_selection_indices
import os
from imbalanceddl.utils.deep_smote_data_loader import (
    CustomImageDataset,
    load_deepsmote_raw,
    inject_label_noise,
)
from imbalanceddl.utils._augmentation import get_weak_augmentation, get_trivial_augmentation
from imbalanceddl.strategy.build_trainer import build_trainer
from torchvision import datasets
from imbalanceddl.utils.sava_key_generation import SavaCacheKey
import torch

class DeepSMOTESavaTrainer(Trainer):
    def __init__(self, cfg, dataset, model, strategy="DeepSMOTESava"):
        print("\n" + "="*60)
        print("DeepSMOTESavaTrainer Initialization")
        print("="*60)

        # Validation dataset - pass cfg.dataset to augmentation function
        _, val_transform = get_weak_augmentation(cfg.dataset)
        print(f"1. Loading validation dataset: {cfg.dataset}")
        if cfg.dataset == 'cifar10':
            val_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
        elif cfg.dataset == 'cifar100':
            val_ds = datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)
        else:
            raise NotImplementedError
        print(f"   Validation set size: {len(val_ds)}")

        noise_first = hasattr(cfg, 'noise_first') and cfg.noise_first
        noise_ratio = getattr(cfg, 'noise_ratio', 0.0)
        print(f"   noise_first = {noise_first}, noise_ratio = {noise_ratio}")

        # ============================================================
        # Load data according to noise_first flag
        # ============================================================
        if noise_first and noise_ratio > 0:
            # Load raw imbalanced data (without any oversampling)
            print(f"\n2. Loading raw imbalanced data for {cfg.dataset}, imb_type={cfg.imb_type}, imb_factor={cfg.imb_factor}")
            from imbalanceddl.dataset.imbalance_cifar import IMBALANCECIFAR10
            raw_ds = IMBALANCECIFAR10(
                root='./data',
                imb_type=cfg.imb_type,
                imb_factor=cfg.imb_factor,
                rand_number=cfg.rand_number,
                train=True,
                download=True,
                transform=None   # no transform, we want raw arrays
            )
            X_raw = raw_ds.data          # numpy array (N, 32, 32, 3)
            Y_raw = np.array(raw_ds.targets)
            print(f"[VERIFY] Raw imbalanced data shape: X={X_raw.shape}, Y={Y_raw.shape}")
            print(f"[VERIFY] Raw class distribution: {dict(zip(*np.unique(Y_raw, return_counts=True)))}")

            # Inject noise into imbalanced labels
            print(f"Applying {noise_ratio*100}% label noise to imbalanced data (noise_first=True)")
            Y_noisy = inject_label_noise(Y_raw, noise_ratio, cfg.num_classes, seed=cfg.rand_number)
            print(f"[VERIFY] After noise injection: class distribution: {dict(zip(*np.unique(Y_noisy, return_counts=True)))}")


            deepsmote_folder = 'deepsmote_models'
            noise_key = f"noise{noise_ratio}_seed{cfg.rand_number}"
            data_file = f"./{deepsmote_folder}/{cfg.dataset}/{cfg.dataset}_{cfg.imb_type}_R{int(1/cfg.imb_factor)}_{noise_key}_train_data.txt"
            label_file = f"./{deepsmote_folder}/{cfg.dataset}/{cfg.dataset}_{cfg.imb_type}_R{int(1/cfg.imb_factor)}_{noise_key}_train_label.txt"

            if not os.path.exists(data_file):
                raise FileNotFoundError(
                    f"DeepSMOTE data for noise-first not found.\n"
                    f"Please pre‑generate it using Deepsmote_Generate_Balance.py with the same noise_ratio and seed.\n"
                    f"Missing: {data_file}"
                )
            # Load the pre‑generated balanced DeepSMOTE data
            X_balanced = np.loadtxt(data_file)
            Y_balanced = np.loadtxt(label_file).astype(int)
            # Reshape to image format
            X_balanced = X_balanced.reshape(-1, 3, 32, 32)
            X_balanced = np.transpose(X_balanced, (0, 2, 3, 1))
            X_balanced = np.clip(X_balanced * 255, 0, 255).astype(np.uint8)
            print(f"[VERIFY] Loaded balanced DeepSMOTE data: X={X_balanced.shape}, Y={Y_balanced.shape}")
            print(f"[VERIFY] Class distribution: {dict(zip(*np.unique(Y_balanced, return_counts=True)))}")
            X_final = X_balanced
            Y_final = Y_balanced
        else:
            # Original behaviour: load pre‑generated balanced DeepSMOTE data (no noise or noise after balance)
            print(f"\n2. Loading balanced DeepSMOTE data for {cfg.dataset}, imb_type={cfg.imb_type}, imb_factor={cfg.imb_factor}")
            X_raw, Y_raw = load_deepsmote_raw(cfg.dataset, cfg.imb_type, cfg.imb_factor)
            print(f"[VERIFY] Balanced data shape: X={X_raw.shape}, Y={Y_raw.shape}")
            if noise_ratio > 0 and not noise_first:
                # Inject noise after balancing
                print(f"Applying {noise_ratio*100}% label noise to balanced data (noise_first=False)")
                Y_noisy = inject_label_noise(Y_raw, noise_ratio, cfg.num_classes, seed=cfg.rand_number)
                Y_final = Y_noisy
                X_final = X_raw
                print(f"[VERIFY] After noise injection: class distribution: {dict(zip(*np.unique(Y_final, return_counts=True)))}")
            else:
                X_final = X_raw
                Y_final = Y_raw

        # ============================================================
        # Create plain and augmented datasets (same as before)
        # ============================================================
        plain_transform = val_transform   # ToTensor + Normalize
        plain_dataset = CustomImageDataset(X_final, Y_final, transform=plain_transform)
        print(f"\n3. Plain dataset (for scoring) created with {len(plain_dataset)} samples")

        # Training transform - pass cfg.dataset
        print(f"\n4. Training transform: cfg.augmentation = {cfg.augmentation}")
        if cfg.augmentation == 'weak':
            train_transform, _ = get_weak_augmentation(cfg.dataset)
        elif cfg.augmentation == 'trivial':
            train_transform, _ = get_trivial_augmentation(cfg.dataset)
        elif cfg.augmentation == 'none':
            if cfg.dataset == 'cifar10':
                normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            elif cfg.dataset == 'cifar100':
                normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            else:
                raise NotImplementedError
            train_transform = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            raise NotImplementedError

        aug_dataset = CustomImageDataset(X_final, Y_final, transform=train_transform)
        original_cls_num_list = aug_dataset.get_cls_num_list()
        cfg.original_cls_num_list = original_cls_num_list
        print(f"\n5. Augmented dataset (for training) created with {len(aug_dataset)} samples")

        # ============================================================
        # Apply SAVA selection (if ratio < 1.0) – unchanged
        # ============================================================
        print(f"\n6. Selection: method={cfg.selection_method}, ratio={cfg.selection_ratio}")
        if cfg.selection_ratio < 1.0:
            if cfg.selection_method == 'sava':
                print("   Computing SAVA scores...")
                is_noisy = noise_ratio > 0
                key_gen = SavaCacheKey(
                    config=cfg,
                    is_deepsmote=True,
                    is_noisy=is_noisy,
                    is_oversampled=False,
                    is_noise_first=noise_first,
                    is_selection_first=False
                )
                file_key = key_gen.generate()
                print(f"[DEBUG] SAVA cache key: {file_key}")
                indices = get_sava_selection_indices(
                    train_dataset=plain_dataset,
                    val_dataset=val_ds,
                    keep_ratio=cfg.selection_ratio,
                    device=cfg.device,
                    file_key=file_key,
                    batch_size=getattr(cfg, 'sava_batch_size', 1024),
                    num_classes=cfg.num_classes,
                    resize=32,
                    cache_label_distances=getattr(cfg, 'sava_cache_label_distances', True),
                    corrupt_por=0.0
                )
                print(f"   SAVA selection completed. Kept {len(indices)} indices.")
            elif cfg.selection_method == 'random':
                total = len(plain_dataset)
                n_keep = int(total * cfg.selection_ratio)
                indices = random.sample(range(total), n_keep)
            else:
                raise ValueError(f"Unknown selection_method: {cfg.selection_method}")
            final_train = Subset(aug_dataset, indices)
            print(f"\n7. Final training set: {len(final_train)} samples (selected subset)")
        else:
            final_train = aug_dataset
            print(f"\n7. Final training set: all {len(final_train)} samples (no selection)")

        # Wrapper and inner trainer delegation – same as before
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
        print("DeepSMOTESavaTrainer initialization complete.\n")
        print("="*60)

    # Delegation methods
    def get_criterion(self):
        return self.inner_trainer.get_criterion()

    def train_one_epoch(self):
        self.inner_trainer.train_one_epoch()

    def do_train_val(self):
        self.inner_trainer.do_train_val()

    def validate(self):
        return self.inner_trainer.validate()