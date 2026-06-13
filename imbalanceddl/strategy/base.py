import abc
import os
import torch
from sklearn.metrics import confusion_matrix
from imbalanceddl.utils.metrics import shot_acc
import numpy as np
from imbalanceddl.utils.backup_sampler import StratifiedSampler
from imbalanceddl.utils.bsampler import SamplerFactory
from imbalanceddl.utils.logging import setup_logger, create_distribution_table
from collections import Counter
import datetime
from torch.utils.data import Subset
from imbalanceddl.utils.debug_logger import get_debug_logger


class BaseTrainer(metaclass=abc.ABCMeta):
    def __init__(self, cfg, dataset, **kwargs):
        self.cfg = cfg
        self.debug = getattr(cfg, 'debug', False)
        self.debug_logger = get_debug_logger(debug=self.debug)
        self._dataset = dataset
        self._parse_train_val(dataset)
        self.custom_base_name = getattr(cfg, 'store_name', f"{cfg.dataset}_{cfg.strategy}")
        self._prepare_logger()
        self.epoch = 0

    @property
    def dataset(self):
        return self._dataset

    @abc.abstractmethod
    def get_criterion(self):
        return NotImplemented

    @abc.abstractmethod
    def train_one_epoch(self):
        return NotImplemented

    def _parse_train_val(self, dataset):
        self.train_dataset, self.val_dataset = dataset.train_val_sets

        if self.debug:
            self.debug_logger.debug(f"train_dataset type: {type(self.train_dataset)}")
            self.debug_logger.debug(f"val_dataset type: {type(self.val_dataset)}")
            if hasattr(self.train_dataset, 'targets'):
                self.debug_logger.debug(f"train_dataset.targets length: {len(self.train_dataset.targets)}")
            elif hasattr(self.train_dataset, 'dataset') and hasattr(self.train_dataset.dataset, 'targets'):
                self.debug_logger.debug(f"train_dataset.dataset.targets length: {len(self.train_dataset.dataset.targets)}")

        if self.cfg.sampling == "WeightedRandomBatchSampler":
            print("Using WeightedRandomBatchSampler.")
            class_idxs = self.train_dataset.get_class_idxs2()
            sampler_factory = SamplerFactory()
            sampler = sampler_factory.get(class_idxs, self.cfg.batch_size, self.cfg.n_batches, self.cfg.alpha, "random")
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_sampler=sampler)

        elif self.cfg.sampling == "WeightedFixedBatchSampler":
            print("Using WeightedFixedBatchSampler.")
            class_idxs = self.train_dataset.get_class_idxs2()
            sampler_factory = SamplerFactory()
            sampler = sampler_factory.get(class_idxs, self.cfg.batch_size, self.cfg.n_batches, self.cfg.alpha, "fixed")
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_sampler=sampler)

        elif self.cfg.sampling == "Random":
            print("Using Random Sampler.")
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.workers,
                pin_memory=True
            )

        elif self.cfg.sampling == "StratifiedSampler":
            print("Using StratifiedSampler.")
            sampler = StratifiedSampler(
                labels=self.train_dataset.targets,
                num_samples=len(self.train_dataset),
                batch_size=self.cfg.batch_size
            )
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                num_workers=self.cfg.workers,
                pin_memory=True
            )
        else:
            raise ValueError(f"Unsupported sampling method: {self.cfg.sampling}")

        if self.debug:
            self.debug_logger.debug(f"Number of train loader batches: {len(self.train_loader)}")

        class_counts = Counter()
        for batch_idx, (_, batch_labels) in enumerate(self.train_loader):
            class_counts.update(batch_labels.tolist())
            if batch_idx == 0:
                if self.debug:
                    self.debug_logger.debug(f"First batch class counts: {dict(sorted(class_counts.items()))}")
                break

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=100,
            shuffle=False,
            num_workers=self.cfg.workers,
            pin_memory=True
        )
        if self.debug:
            self.debug_logger.debug(f"Validation loader created. Length: {len(self.val_loader)}")

    def _prepare_logger(self):
        log_base = self.cfg.root_log
        os.makedirs(log_base, exist_ok=True)

        log_name = (
            f"{self.cfg.dataset}_"
            f"{self.cfg.selection_method}{self.cfg.selection_ratio}_"
            f"{self.cfg.strategy.lower()}_"
            f"exp{self.cfg.imb_factor}_"
            f"seed{self.cfg.seed}"
        )
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        full_log_path = os.path.join(log_base, f"{log_name}_{timestamp}.log")

        self.logger, self.log_filename = setup_logger(full_log_path)

        header = (
            f"Log file: {os.path.basename(full_log_path)}\n"
            f"Run started: {datetime.datetime.now()}\n"
            f"Dataset: {self.cfg.dataset}\n"
            f"Imbalance type: {self.cfg.imb_type}, factor: {self.cfg.imb_factor}\n"
            f"Selection method: {self.cfg.selection_method}, ratio: {self.cfg.selection_ratio}\n"
            f"Strategy: {self.cfg.strategy}, epochs: {self.cfg.epochs}\n"
            f"Seed: {self.cfg.seed}, rand_number: {self.cfg.rand_number}\n"
            f"Augmentation: {self.cfg.augmentation}\n"
        )
        if hasattr(self.cfg, 'noise_ratio') and self.cfg.noise_ratio > 0:
            header += f"Noise ratio: {self.cfg.noise_ratio}\n"
        mixup_strategies = ['Mixup_DRW', 'Mixup', 'Remix_DRW', 'MAMix_DRW']
        if hasattr(self.cfg, 'mixup_alpha') and self.cfg.strategy in mixup_strategies:
            header += f"mixup_alpha: {self.cfg.mixup_alpha}\n"
        if hasattr(self.cfg, 'mamix_ratio') and self.cfg.mamix_ratio is not None:
            header += f"mamix_ratio: {self.cfg.mamix_ratio}\n"
        header += "=" * 60 + "\n"

        self.logger.info(header)
        self.logger.info("=> No CSV or TensorBoard logging – only console and debug log.")

        self.log_training = None
        self.log_testing = None
        self.tf_writer = None

        def get_cls_num_list(dataset):
            if hasattr(dataset, 'get_cls_num_list'):
                return dataset.get_cls_num_list()
            elif isinstance(dataset, Subset):
                targets = [dataset.dataset[i][1] for i in dataset.indices]
                return np.bincount(targets, minlength=self.cfg.num_classes).tolist()
            elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'get_cls_num_list'):
                return dataset.dataset.get_cls_num_list()
            elif hasattr(dataset, 'targets'):
                targets = dataset.targets
            else:
                targets = [dataset[i][1] for i in range(len(dataset))]
            return np.bincount(targets, minlength=self.cfg.num_classes).tolist()

        selected_counts = get_cls_num_list(self.train_dataset)
        selected_dict = {i: count for i, count in enumerate(selected_counts)}
        if hasattr(self.cfg, 'original_cls_num_list') and self.cfg.original_cls_num_list:
            original_counts = self.cfg.original_cls_num_list
        else:
            original_counts = selected_counts
        orig_dict = {i: count for i, count in enumerate(original_counts)}
        create_distribution_table(self.logger, orig_dict, selected_dict)

    def compute_metrics_and_record(self, all_preds, all_targets, losses, top1, top5, flag='Training'):
        if self.debug:
            self.debug_logger.debug(f"compute_metrics_and_record started for epoch {self.epoch}")
            self.debug_logger.debug(f"all_preds length: {len(all_preds)}, all_targets length: {len(all_targets)}")
            if len(all_preds) > 0:
                self.debug_logger.debug(f"First 10 preds: {all_preds[:10]}")
                self.debug_logger.debug(f"First 10 targets: {all_targets[:10]}")
            unique_preds = np.unique(all_preds)
            unique_targets = np.unique(all_targets)
            self.debug_logger.debug(f"Unique predicted classes: {len(unique_preds)} (first 10: {unique_preds[:10]})")
            self.debug_logger.debug(f"Unique target classes: {len(unique_targets)} (first 10: {unique_targets[:10]})")

        if self.cfg.dataset in ['cifar100', 'tiny200']:
            if self.debug:
                self.debug_logger.debug("About to call shot_acc...")
            many_acc, median_acc, low_acc = shot_acc(
                self.cfg, np.array(all_preds), np.array(all_targets),
                self.train_dataset, acc_per_cls=False
            )
            if self.debug:
                self.debug_logger.debug(f"shot_acc returned: many={many_acc:.4f}, median={median_acc:.4f}, low={low_acc:.4f}")
            group_acc = np.array([many_acc, median_acc, low_acc])
            group_acc_string = f'{flag} Group Acc: {np.array2string(group_acc, separator=",", formatter={"float_kind": lambda x: f"{x:.3f}"})}'
            self.logger.info(group_acc_string)
            print(group_acc_string)
        else:
            group_acc = None
            group_acc_string = None

        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt

        epoch_output = (
            f'Epoch [{self.epoch}] {flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {losses.avg:.5f}'
        )
        cls_acc_string = f'Epoch [{self.epoch}] {flag} Class Recall: {np.array2string(cls_acc, separator=",", formatter={"float_kind": lambda x: f"{x:.3f}"})}'

        print(epoch_output)
        print(cls_acc_string)
        self.logger.info(epoch_output)
        self.logger.info(cls_acc_string)

        if self.cfg.best_model is not None:
            return cls_acc_string

        for handler in self.logger.handlers:
            handler.flush()
