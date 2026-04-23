import numpy as np
import random
import torchvision.transforms as transforms
from torch.utils.data import Subset
from .trainer import Trainer
from imbalanceddl.strategy.selection_method.lava_selection import get_lava_selection_indices
from imbalanceddl.utils.deep_smote_data_loader import CustomImageDataset, load_and_cap_deepsmote
from imbalanceddl.utils._augmentation import get_weak_augmentation, get_trivial_augmentation
from imbalanceddl.strategy.build_trainer import build_trainer
from torchvision import datasets
from imbalanceddl.utils.key_generation import LavaCacheKey
import pdb 
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

        # Load and cap DeepSMOTE data once
        print(f"\n2. Loading DeepSMOTE data for {cfg.dataset}, imb_type={cfg.imb_type}, imb_factor={cfg.imb_factor}")
        X_capped, Y_capped = load_and_cap_deepsmote(
            dataset=cfg.dataset,
            imb_type=cfg.imb_type,
            imb_factor=cfg.imb_factor,
            class_caps=None  # Uses default [5000, 4000, ..., 4000]
        )
        print(f"   Capped data shape: X={X_capped.shape}, Y={Y_capped.shape}")
        unique, counts = np.unique(Y_capped, return_counts=True)
        print(f"   Class distribution after capping: {dict(zip(unique, counts))}")

        # Create plain dataset (no augmentation, only normalization)
        plain_transform = val_transform   # ToTensor + Normalize
        plain_dataset = CustomImageDataset(X_capped, Y_capped, transform=plain_transform)
        print(f"\n3. Plain dataset (for scoring) created with {len(plain_dataset)} samples")
        print(f"   Transform: ToTensor + Normalize (no augmentation)")

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
        
        # Create augmented dataset (with augmentation)
        aug_dataset = CustomImageDataset(X_capped, Y_capped, transform=train_transform)
        original_cls_num_list = aug_dataset.get_cls_num_list()   
        cfg.original_cls_num_list = original_cls_num_list

        print(f"\n5. Augmented dataset (for training) created with {len(aug_dataset)} samples")

        # Apply selection (LAVA or random) on the plain dataset
        print(f"\n6. Selection: method={cfg.selection_method}, ratio={cfg.selection_ratio}")
        if cfg.selection_ratio < 1.0:
            if cfg.selection_method == 'lava':
                print("   Computing LAVA scores...")
                key_gen = LavaCacheKey(config=cfg, is_deepsmote=True)
                file_key = key_gen.generate()
                indices = get_lava_selection_indices(
                    plain_dataset,
                    val_ds,
                    keep_ratio=cfg.selection_ratio,
                    device=cfg.device,
                    file_key=file_key
                )
                # NOTE: TEST FOR SMALL DATASET HALF TRAINING SIZE TO SPEED UP LAVA SCORING, CAN BE INCREASED IF MORE COMPUTE AVAILABLE
                print(f"   LAVA selection completed. Kept {len(indices)} indices.")
            elif cfg.selection_method == 'random':
                print("   Randomly selecting samples...")
                total = len(plain_dataset)
                n_keep = int(total * cfg.selection_ratio)
                indices = random.sample(range(total), n_keep)
                print(f"   Random selection kept {len(indices)} samples out of {total}")
            else:
                raise ValueError(f"Unknown selection_method: {cfg.selection_method}")
            final_train = Subset(aug_dataset, indices)
            print(f"\n7. Final training set: {len(final_train)} samples (selected subset)")
        else:
            final_train = aug_dataset
            print(f"\n7. Final training set: all {len(final_train)} samples (no selection)")

        # Wrap for inner trainer
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

        # Inner trainer
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