import numpy as np
import random
import torchvision.transforms as transforms
from torch.utils.data import Subset
from .trainer import Trainer
import inspect

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
        print("RandomOversamplingSelectionTrainer Initialization (Clean & Streamlined)")
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

        # 2. Load original clean imbalanced dataset
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

        # Default pipeline order to False since noise_first is removed from config
        noise_first = False

        # ------------------------------------------------------------
        # Pipeline 1: Oversample → Noise 
        # ------------------------------------------------------------
        original_counts = np.bincount(Y, minlength=cfg.num_classes)
        majority_count = max(original_counts)
        print(f"Original class distribution: {dict(enumerate(original_counts))}")
        print(f"Majority class size: {majority_count}")

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

        # 7. Create plain dataset (no augmentation, only normalization)
        plain_transform = val_transform   # ToTensor + Normalize
        plain_dataset = CustomImageDataset(X_bal, Y_bal, transform=plain_transform)
        print(f"\n3. Plain dataset (for scoring) created with {len(plain_dataset)} samples")
        print(f"   Transform: ToTensor + Normalize (no augmentation)")

        # 8. Determine training transform
        print(f"\n4. Training transform: cfg.augmentation = {cfg.augmentation}")
        if cfg.augmentation == 'weak':
            train_transform, _ = get_weak_augmentation()
            print("   Using weak augmentation (RandomCrop + RandomHorizontalFlip)")
        elif cfg.augmentation == 'trivial':
            train_transform, _ = get_trivial_augmentation()
            print("   Using trivial augmentation (only ToTensor + Normalize)")
        else:
            raise NotImplementedError(f"Augmentation {cfg.augmentation} not supported")

        # Create augmented dataset (with augmentation)
        aug_dataset = CustomImageDataset(X_bal, Y_bal, transform=train_transform)
        cfg.original_cls_num_list = aug_dataset.get_cls_num_list()

        # 9. Apply selection (SAVA or random) on the plain dataset
        print(f"\n5. Selection: method={cfg.selection_method}, ratio={cfg.selection_ratio}")
        if cfg.selection_ratio < 1.0:
            
            if cfg.selection_method == 'sava':
                print("   Computing SAVA scores...")
                key_gen = SavaCacheKey(config=cfg, is_deepsmote=False, is_noisy=False,
                                       is_oversampled=True, is_noise_first=False)
                file_key = key_gen.generate()
                print(f"[DEBUG] SAVA file_key = {file_key}")
                
                # Fetch explicit parameters from config safely
                sava_batch_size = getattr(cfg, 'sava_batch_size', 1024)
                sava_cache_label_distances = getattr(cfg, 'sava_cache_label_distances', True)
                
                # Dynamic Signature Alignment
                sig = inspect.signature(get_sava_selection_indices)
                
                # Cleaned parameter mapping with fallback defaults injected directly here
                all_possible_args = {
                    'train_dataset': plain_dataset,
                    'val_dataset': val_ds,
                    'keep_ratio': cfg.selection_ratio,
                    'device': cfg.device,
                    'file_key': file_key,
                    'batch_size': sava_batch_size,
                    'num_classes': cfg.num_classes,
                    'parallel': False,                       # Injected default
                    'cuda_num': getattr(cfg, 'gpu', 0),       # Map to standard 'gpu' config key
                    'n_gpu': 1,                              # Injected default
                    'resize': 32,                            # Injected default
                    'cache_label_distances': sava_cache_label_distances,
                    'model_path': None,
                    'corrupt_por': 0.0
                }
                
                # Align raw pixels representation variation cleanly
                if 'feat_repr' in sig.parameters:
                    all_possible_args['feat_repr'] = False
                elif 'feature_repr' in sig.parameters:
                    all_possible_args['feature_repr'] = False

                # Filter arguments against backend definition
                filtered_args = {k: v for k, v in all_possible_args.items() if k in sig.parameters}
                
                indices = get_sava_selection_indices(**filtered_args)
                print(f"[DEBUG] Selected {len(indices)} indices via SAVA")

            elif cfg.selection_method == 'random':
                print("   Randomly selecting samples...")
                total = len(plain_dataset)
                n_keep = int(total * cfg.selection_ratio)
                indices = random.sample(range(total), n_keep)
            else:
                raise ValueError(f"Unknown selection_method: {cfg.selection_method}")

            final_train = Subset(aug_dataset, indices)
            print(f"\n6. Final training set: {len(final_train)} samples (selected subset)")
        else:
            final_train = aug_dataset
            print(f"\n6. Final training set: all {len(final_train)} samples (no selection)")

        # 10. Wrap for inner trainer
        class SimpleWrapper:
            def __init__(self, train, val, cfg):
                self.train_val_sets = (train, val)
                self.cfg = cfg
                targets = train.dataset.Y if hasattr(train, 'dataset') else train.Y
                self.cls_num_list = np.bincount(targets, minlength=cfg.num_classes).tolist()
                print(f"   Wrapper class counts: {self.cls_num_list}")
        wrapper = SimpleWrapper(final_train, val_ds, cfg)
        cfg.cls_num_list = wrapper.cls_num_list

        # 11. Inner trainer
        base_strategy = getattr(cfg, 'base_strategy', 'ERM')
        print(f"\n7. Building inner trainer with base_strategy={base_strategy}")
        self.inner_trainer = build_trainer(cfg, wrapper, model, base_strategy)

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

    def get_criterion(self): return self.inner_trainer.get_criterion()
    def train_one_epoch(self): self.inner_trainer.train_one_epoch()
    def do_train_val(self): self.inner_trainer.do_train_val()
    def validate(self): return self.inner_trainer.validate()