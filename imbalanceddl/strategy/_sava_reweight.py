# imbalanceddl/strategy/_sava_reweight.py

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn as nn

from imbalanceddl.strategy.trainer import Trainer
from imbalanceddl.utils.utils import AverageMeter
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
        self.temp = getattr(cfg, 'sava_reweight_temp', 1.0)
        self.clip_min = getattr(cfg, 'sava_weights_clip', 1e-3)
        self.scores_file = getattr(cfg, 'sava_scores_file', None)

        # Parent init sets up train/val loaders, model, optimizer, etc.
        super().__init__(cfg, dataset, model=model, strategy=strategy)

        # Prepare weights from scores
        self._prepare_weights()
        # Override the train loader according to mode
        self._override_loader()

        # Criterion will be created in get_criterion() during training loop.
        # Do not call self.get_criterion() here.

    def _prepare_weights(self):
        """Obtain scores and convert to positive weights."""
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

        # Convert scores to weights: lower score → higher weight.
        # Exponential: w = exp(-score / temp). Shift scores so max = 0 to avoid overflow.
        scores_shifted = scores - np.max(scores)
        weights = np.exp(scores_shifted / self.temp)
        weights = np.clip(weights, self.clip_min, None)
        # Normalise so mean = 1 (helps keep loss scale)
        weights = weights / np.mean(weights)
        self.sample_weights = weights.astype(np.float32)

        print(f"SAVA reweighting: mean weight={weights.mean():.4f}, "
              f"min={weights.min():.4f}, max={weights.max():.4f}")

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
            # Wrap dataset to return weights
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

    def get_criterion(self):
        """Create the loss function. For loss mode, use reduction='none'; for sampler, reduction='mean'."""
        if self.reweight_mode == 'loss':
            self.criterion = nn.CrossEntropyLoss(reduction='none').cuda(self.cfg.gpu)
            print("Created CrossEntropyLoss with reduction='none' for loss weighting.")
        else:
            # Sampler mode uses standard mean reduction.
            self.criterion = nn.CrossEntropyLoss(reduction='mean').cuda(self.cfg.gpu)
            print("Created CrossEntropyLoss with reduction='mean' for sampler weighting.")
        return self.criterion

    def train_one_epoch(self):
        if self.reweight_mode == 'sampler':
            # Standard training (loss mean over batch)
            super().train_one_epoch()
        else:
            self._train_one_epoch_weighted_loss()

    def _train_one_epoch_weighted_loss(self):
        """Training loop with per‑sample loss weighting."""
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
            loss_per_sample = self.criterion(outputs, labels)  # reduction already 'none'
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