# imbalanceddl/strategy/_sava_reweight.py

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn as nn

from imbalanceddl.strategy.trainer import Trainer
from imbalanceddl.utils.utils import AverageMeter, save_checkpoint
from imbalanceddl.utils.metrics import accuracy


class WeightedDataset(Dataset):
    """Wraps a dataset to return (image, label, weight) for loss‑weighting mode."""
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


class SAVAReweightTrainer(Trainer):
    """
    Trainer that uses SAVA scores as sample weights.
    Two modes:
        - 'loss'   : multiply per‑sample loss by weight (reduction='none')
        - 'sampler': use WeightedRandomSampler to oversample valuable samples
    """
    def __init__(self, cfg, dataset, model, strategy="SAVA_Reweight"):
        self.reweight_mode = getattr(cfg, 'reweight_mode', 'loss')
        self.temp = getattr(cfg, 'sava_reweight_temp', None)
        self.clip_min = getattr(cfg, 'sava_weights_clip', 1e-3)
        self.scores_file = getattr(cfg, 'sava_scores_file', None)
        self.method = getattr(cfg, 'sava_reweight_method', 'exp')  # 'exp' or 'inv'
        self.max_class_weight_ratio = getattr(cfg, 'sava_max_class_ratio', 100.0)

        super().__init__(cfg, dataset, model=model, strategy=strategy)
        self.debug = getattr(cfg, 'debug', False)

        self._prepare_weights()
        self._override_loader()

    def _prepare_weights(self):
        """Obtain scores and convert to positive weights, with per‑class balancing."""
        if self.scores_file is not None:
            print(f"Loading SAVA scores from {self.scores_file}")
            scores = np.load(self.scores_file)
        elif hasattr(self.train_dataset, 'scores') and self.train_dataset.scores is not None:
            scores = self.train_dataset.scores
        else:
            raise RuntimeError(
                "No SAVA scores found. Provide --sava_scores_file or ensure the dataset "
                "has a 'scores' attribute (e.g., from SavaDataset with method='sava')."
            )

        scores = np.asarray(scores, dtype=np.float64)
        if len(scores) != len(self.train_dataset):
            raise ValueError(
                f"Scores length {len(scores)} != dataset size {len(self.train_dataset)}"
            )

        # ---- Auto‑scale temperature if needed ----
        if self.temp is None or self.temp <= 0:
            auto_temp = max(2.0 * scores.std(), 1.0)
            print(f"Auto‑set SAVA reweight temperature = {auto_temp:.2f} (std={scores.std():.2f})")
            self.temp = auto_temp
        else:
            score_range = scores.max() - scores.min()
            if score_range > 0 and self.temp < score_range * 0.01:
                print(f"Warning: temperature {self.temp} is very small relative to score range {score_range:.2f}.")

        # ---- Convert to weights ----
        scores_shifted = scores - np.min(scores)   # non‑negative
        if self.method == 'exp':
            weights = np.exp(-scores_shifted / self.temp)
        elif self.method == 'inv':
            # inverse weighting: 1 / (shift + eps)
            weights = 1.0 / (scores_shifted + 1e-6)
        else:
            raise ValueError(f"Unknown reweight method: {self.method}")

        weights = np.clip(weights, self.clip_min, None)

        # ---- Per‑class weight adjustment to prevent extreme imbalance ----
        # Get targets (labels) from the training dataset
        if hasattr(self.train_dataset, 'targets'):
            targets = np.array(self.train_dataset.targets)
        else:
            # fallback: iterate once to collect labels (slow but safe)
            targets = np.array([self.train_dataset[i][1] for i in range(len(self.train_dataset))])

        class_avg_weights = []
        for c in range(self.cfg.num_classes):
            mask = (targets == c)
            if np.any(mask):
                class_avg_weights.append(np.mean(weights[mask]))
            else:
                class_avg_weights.append(0.0)
        class_avg_weights = np.array(class_avg_weights)

        # Print per‑class average weights
        print("SAVA reweighting: per‑class average weights:")
        for c, avg_w in enumerate(class_avg_weights):
            print(f"  Class {c}: {avg_w:.6f}")

        # If any class has zero weight (or very small), boost it
        min_avg = max(class_avg_weights[class_avg_weights > 0].min(), 1e-6)
        max_avg = class_avg_weights.max()
        ratio = max_avg / min_avg
        if ratio > self.max_class_weight_ratio:
            print(f"Warning: class weight ratio {ratio:.2f} exceeds limit {self.max_class_weight_ratio}. "
                  "Clipping per‑class weights to prevent collapse.")
            # Scale down the weights of over‑weighted classes
            for c in range(self.cfg.num_classes):
                if class_avg_weights[c] > min_avg * self.max_class_weight_ratio:
                    # Reduce weights for this class so that its average becomes min_avg * limit
                    scale = (min_avg * self.max_class_weight_ratio) / class_avg_weights[c]
                    mask = (targets == c)
                    weights[mask] *= scale
            # Re‑compute averages after clipping
            for c in range(self.cfg.num_classes):
                mask = (targets == c)
                if np.any(mask):
                    class_avg_weights[c] = np.mean(weights[mask])
            print("After clipping, per‑class averages:")
            for c, avg_w in enumerate(class_avg_weights):
                print(f"  Class {c}: {avg_w:.6f}")

        # Normalise so mean = 1
        weights = weights / np.mean(weights)
        self.sample_weights = weights.astype(np.float32)

        print(f"SAVA reweighting: final mean weight={weights.mean():.4f}, "
              f"min={weights.min():.4f}, max={weights.max():.4f}")

        if self.debug:
            # Log to main logger (if set to DEBUG level) – but we already printed to console.
            pass

    def _override_loader(self):
        """Replace self.train_loader based on reweight_mode."""
        if self.reweight_mode == 'sampler':
            sampler = WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.train_dataset),
                replacement=True
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.cfg.batch_size,
                sampler=sampler,
                num_workers=self.cfg.workers,
                pin_memory=True
            )
            print("Using WeightedRandomSampler for SAVA reweighting.")
        elif self.reweight_mode == 'loss':
            self.train_dataset = WeightedDataset(self.train_dataset, self.sample_weights)
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.workers,
                pin_memory=True
            )
            print("Using loss weighting for SAVA reweighting.")
        else:
            raise ValueError(f"Unknown reweight_mode: {self.reweight_mode}")

        if self.debug:
            # Optionally log loader info to main logger
            pass

    def get_criterion(self):
        if self.reweight_mode == 'loss':
            self.criterion = nn.CrossEntropyLoss(reduction='none').cuda(self.cfg.gpu)
            print("Created CrossEntropyLoss with reduction='none' for loss weighting.")
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='mean').cuda(self.cfg.gpu)
            print("Created CrossEntropyLoss with reduction='mean' for sampler weighting.")
        return self.criterion

    def train_one_epoch(self):
        if self.reweight_mode == 'sampler':
            super().train_one_epoch()
        else:
            self._train_one_epoch_weighted_loss()

    def _train_one_epoch_weighted_loss(self):
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
                    f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                )
                print(output)
                if self.log_training is not None:
                    self.log_training.write(output + '\n')
                    self.log_training.flush()

        self.compute_metrics_and_record(
            all_preds, all_targets, losses, top1, top5, flag='Training'
        )

    def do_train_val(self):
        """Override to preserve custom loader."""
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