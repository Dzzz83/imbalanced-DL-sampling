import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Subset
from .trainer import Trainer
from imbalanceddl.strategy.selection_method.lava_selection import get_lava_selection_indices
from imbalanceddl.utils.deep_smote_data_loader import load_deepsmote_dataset
from imbalanceddl.utils._augmentation import get_weak_augmentation, get_trivial_augmentation
from imbalanceddl.strategy.build_trainer import build_trainer
from torchvision import datasets

class DeepSMOTELAVATrainer(Trainer):
    def __init__(self, cfg, dataset, model, strategy="DeepSMOTELAVA"):
        # 1. Validation dataset
        _, val_transform = get_weak_augmentation()
        if cfg.dataset == 'cifar10':
            val_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
        elif cfg.dataset == 'cifar100':
            val_ds = datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)
        else:
            raise NotImplementedError

        # 2. Load DeepSMOTE datasets: plain (for scoring) and augmented (for training)
        print("Loading pre-generated DeepSMOTE balanced dataset (plain, no augmentation)...")
        deepsmote_plain = load_deepsmote_dataset(
            dataset=cfg.dataset,
            imb_type=cfg.imb_type,
            imb_factor=cfg.imb_factor,
            transform=None
        )

        # Determine training transform
        if cfg.augmentation == 'weak':
            train_transform, _ = get_weak_augmentation()
        elif cfg.augmentation == 'trivial':
            train_transform, _ = get_trivial_augmentation()
        elif cfg.augmentation == 'none':
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            raise NotImplementedError(f"Augmentation {cfg.augmentation} not supported")

        print("Loading same dataset with training augmentation...")
        deepsmote_aug = load_deepsmote_dataset(
            dataset=cfg.dataset,
            imb_type=cfg.imb_type,
            imb_factor=cfg.imb_factor,
            transform=train_transform
        )

        # 3. Apply LAVA selection on the plain dataset
        if cfg.selection_ratio < 1.0 and cfg.selection_method == 'lava':
            print(f"Computing LAVA scores and selecting top {cfg.selection_ratio*100}%...")
            indices = get_lava_selection_indices(
                deepsmote_plain,          # training set (plain)
                val_ds,                   # validation set
                keep_ratio=cfg.selection_ratio,
                device=cfg.device,
                file_key=f"{cfg.dataset}_deepsmote_{cfg.imb_type}_{cfg.imb_factor}_{cfg.rand_number}"
            )
            final_train = Subset(deepsmote_aug, indices)
            print(f"Selected {len(final_train)} samples out of {len(deepsmote_aug)}")
        else:
            final_train = deepsmote_aug

        # 4. Wrap final training set and validation set for the inner trainer
        class SimpleWrapper:
            def __init__(self, train, val, cfg):
                self.train_val_sets = (train, val)
                self.cfg = cfg
                # Compute class counts for compatibility
                if hasattr(train, 'dataset'):  # if it's a Subset
                    targets = train.dataset.Y
                else:
                    targets = train.Y
                self.cls_num_list = np.bincount(targets, minlength=cfg.num_classes).tolist()
        wrapper = SimpleWrapper(final_train, val_ds, cfg)

        # 5. Inner trainer
        base_strategy = getattr(cfg, 'base_strategy', 'ERM')
        print(f"Base strategy: {base_strategy}")
        self.inner_trainer = build_trainer(cfg, wrapper, model, base_strategy)

        # 6. Delegate attributes
        self.cfg = cfg
        self.model = model
        self.epoch = 0
        self.best_acc1 = 0
        self.train_loader = self.inner_trainer.train_loader
        self.val_loader = self.inner_trainer.val_loader
        self.optimizer = self.inner_trainer.optimizer
        self.criterion = self.inner_trainer.criterion
        self.logger = self.inner_trainer.logger
        self.log_training = self.inner_trainer.log_training
        self.log_testing = self.inner_trainer.log_testing
        self.tf_writer = self.inner_trainer.tf_writer

    def get_criterion(self):
        return self.inner_trainer.get_criterion()

    def train_one_epoch(self):
        self.inner_trainer.train_one_epoch()

    def do_train_val(self):
        self.inner_trainer.do_train_val()

    def validate(self):
        return self.inner_trainer.validate()