import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from imbalanceddl.strategy.trainer import Trainer
from imbalanceddl.utils.utils import AverageMeter, save_checkpoint
from imbalanceddl.utils.metrics import accuracy


class BalancedDataset(Dataset):
    """Wraps a dataset to return (image, label, weight) for loss weighting."""
    def __init__(self, base_dataset, weights):
        self.base = base_dataset
        self.weights = torch.tensor(weights, dtype=torch.float32)
        if hasattr(base_dataset, 'targets'):
            self.targets = base_dataset.targets
        if hasattr(base_dataset, 'get_cls_num_list'):
            self.get_cls_num_list = base_dataset.get_cls_num_list

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img, label, self.weights[idx]


class ClassBalancedERMTrainer(Trainer):
    """
    Trainer that applies sqrt inverse class frequency weights from epoch 0.
    No SAVA scores are used – this is a pure class‑balancing baseline.
    """
    def __init__(self, cfg, dataset, model, strategy="ClassBalanced_ERM"):
        super().__init__(cfg, dataset, model=model, strategy=strategy)
        self.clip_min = getattr(cfg, 'sava_weights_clip', 0.1)
        self.clip_max = getattr(cfg, 'sava_max_weight', 10.0)
        self._prepare_weights()
        self._override_loader()

    def _prepare_weights(self):
        """Compute sqrt inverse class frequency weights."""
        if hasattr(self.train_dataset, 'targets'):
            targets = np.array(self.train_dataset.targets)
        else:
            targets = np.array([self.train_dataset[i][1] for i in range(len(self.train_dataset))])

        class_counts = np.bincount(targets, minlength=self.cfg.num_classes).astype(np.float64)
        class_counts = np.maximum(class_counts, 1.0)

        max_count = np.max(class_counts)
        class_boost = np.sqrt(max_count / class_counts)
        weights = class_boost[targets]

        clipped = np.clip(weights, self.clip_min, self.clip_max)
        weights = clipped / np.mean(clipped)

        self.sample_weights = weights.astype(np.float32)
        print(f"[ClassBalancedERM] Weight stats: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")

    def _override_loader(self):
        """Wrap dataset with weights and create DataLoader."""
        self.train_dataset = BalancedDataset(self.train_dataset, self.sample_weights)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.workers,
            pin_memory=True
        )
        print("Using loss weighting with sqrt class balance (no SAVA).")

    def get_criterion(self):
        self.criterion = nn.CrossEntropyLoss(reduction='none').cuda(self.cfg.gpu)
        return self.criterion

    def train_one_epoch(self):
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        all_preds, all_targets = [], []

        self.model.train()
        for i, (images, labels, weights) in enumerate(self.train_loader):
            if self.cfg.gpu is not None:
                images = images.cuda(self.cfg.gpu, non_blocking=True)
                labels = labels.cuda(self.cfg.gpu, non_blocking=True)
                weights = weights.cuda(self.cfg.gpu, non_blocking=True)

            outputs, _ = self.model(images)
            loss_per_sample = self.criterion(outputs, labels)
            loss = (loss_per_sample * weights).mean()

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.cfg.print_freq == 0:
                output = (
                    f'Epoch: [{self.epoch}][{i}/{len(self.train_loader)}], '
                    f'lr: {self.optimizer.param_groups[-1]["lr"]:.5f}\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                )
                print(output)
                if self.log_training is not None:
                    self.log_training.write(output + '\n')
                    self.log_training.flush()

        self.compute_metrics_and_record(
            all_preds, all_targets, losses, top1, top5, flag='Training'
        )

    # Override do_train_val to prevent parent from re-creating the loader
    def do_train_val(self):
        for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
            self.epoch = epoch
            self.adjust_learning_rate()
            self.get_criterion()
            assert self.criterion is not None, "No criterion !"
            self.train_one_epoch()
            acc1 = self.validate()
            is_best = acc1 > self.best_acc1
            self.best_acc1 = max(acc1, self.best_acc1)

            output_best = f'Best Prec@1: {self.best_acc1:.3f}\n'
            print(output_best)
            if self.log_testing is not None:
                self.log_testing.write(output_best)
                self.log_testing.flush()

            save_checkpoint(
                self.cfg, {
                    'epoch': self.epoch + 1,
                    'backbone': self.cfg.backbone,
                    'classifier': self.cfg.classifier,
                    'state_dict': self.model.state_dict(),
                    'best_acc1': self.best_acc1,
                    'optimizer': self.optimizer.state_dict()
                }, is_best, self.epoch
            )