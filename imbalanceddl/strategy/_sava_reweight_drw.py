import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn as nn

from imbalanceddl.strategy.trainer import Trainer
from imbalanceddl.utils.utils import AverageMeter, save_checkpoint
from imbalanceddl.utils.metrics import accuracy

class WeightedDataset(Dataset):
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

class SAVAReweightDRWTrainer(Trainer):
    """Trainer that uses SAVA scores with Deferred Re-Weighting (DRW)."""
    def __init__(self, cfg, dataset, model, strategy="SAVA_Reweight_DRW"):
        self.reweight_mode = getattr(cfg, 'reweight_mode', 'loss')
        self.temp = getattr(cfg, 'sava_reweight_temp', 1.0)
        self.clip_min = getattr(cfg, 'sava_weights_clip', 0.1)
        self.clip_max = getattr(cfg, 'sava_max_weight', 10.0)
        self.scores_file = getattr(cfg, 'sava_scores_file', None)
        self.warm_epochs = getattr(cfg, 'warm', 160)

        super().__init__(cfg, dataset, model=model, strategy=strategy)
        self._prepare_weights()
        self._override_loader()

    def _prepare_weights(self):
        """Obtain scores and convert to weights using Effective Number scaling."""
        if self.scores_file is not None:
            print(f"Loading SAVA scores from {self.scores_file}")
            scores = np.load(self.scores_file)
        elif hasattr(self.train_dataset, 'scores') and self.train_dataset.scores is not None:
            scores = self.train_dataset.scores
        else:
            raise RuntimeError("No SAVA scores found.")

        scores = np.asarray(scores, dtype=np.float64)
        
        # 1. Global Min-Max Normalization to [0, 1]
        min_s, max_s = np.min(scores), np.max(scores)
        norm_scores = (scores - min_s) / (max_s - min_s + 1e-6)

        # 2. Convert scores to weights (Lower score = Higher weight)
        raw_weights = np.exp(-norm_scores / self.temp)
        
        # 3. Class-aware Boost using EFFECTIVE NUMBER FORMULA
        if hasattr(self.train_dataset, 'targets'):
            targets = np.array(self.train_dataset.targets)
        else:
            targets = np.array([self.train_dataset[i][1] for i in range(len(self.train_dataset))])

        class_counts = np.bincount(targets, minlength=self.cfg.num_classes).astype(np.float64)
        class_counts = np.maximum(class_counts, 1.0)
        
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        class_boost = (1.0 - beta) / np.array(effective_num)
        class_boost = class_boost / np.sum(class_boost) * self.cfg.num_classes
        
        # Apply boost
        boosted_weights = raw_weights * class_boost[targets]

        # 4. Clipping and Normalization
        clipped_weights = np.clip(boosted_weights, self.clip_min, self.clip_max)
        weights = clipped_weights / np.mean(clipped_weights)
        
        self.sample_weights = weights.astype(np.float32)

        print(f"SAVA DRW Reweight Stats: Min={weights.min():.4f}, Max={weights.max():.4f}, Mean={weights.mean():.4f}")

    def _override_loader(self):
        if self.reweight_mode == 'sampler':
            sampler = WeightedRandomSampler(weights=self.sample_weights, num_samples=len(self.train_dataset), replacement=True)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, sampler=sampler, num_workers=self.cfg.workers, pin_memory=True)
        elif self.reweight_mode == 'loss':
            self.train_dataset = WeightedDataset(self.train_dataset, self.sample_weights)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.workers, pin_memory=True)

    def get_criterion(self):
        self.criterion = nn.CrossEntropyLoss(reduction='none').cuda(self.cfg.gpu)
        return self.criterion

    def train_one_epoch(self):
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        all_preds, all_targets = [], []

        self.model.train()
        for i, data in enumerate(self.train_loader):
            if self.reweight_mode == 'loss':
                images, labels, weights = data
                weights = weights.cuda(self.cfg.gpu, non_blocking=True)
            else:
                images, labels = data
                weights = None

            images = images.cuda(self.cfg.gpu, non_blocking=True)
            labels = labels.cuda(self.cfg.gpu, non_blocking=True)

            outputs, _ = self.model(images)
            loss_per_sample = self.criterion(outputs, labels)

            # DRW LOGIC: ERM for first 160 epochs, then apply SAVA weights
            if self.epoch >= self.warm_epochs and self.reweight_mode == 'loss':
                loss = (loss_per_sample * weights).mean()
            else:
                loss = loss_per_sample.mean()

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
                output = (f'Epoch: [{self.epoch}][{i}/{len(self.train_loader)}], '
                          f'lr: {self.optimizer.param_groups[-1]["lr"]:.5f}\t'
                          f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                          f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')
                print(output)
                if self.log_training is not None:
                    self.log_training.write(output + '\n')
                    self.log_training.flush()

        self.compute_metrics_and_record(all_preds, all_targets, losses, top1, top5, flag='Training')

    def do_train_val(self):
        for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
            self.epoch = epoch
            self.adjust_learning_rate()
            self.get_criterion()
            if epoch == self.warm_epochs:
                print(f"--- Epoch {epoch}: Activating SAVA Weights (DRW) ---")
            self.train_one_epoch()
            acc1 = self.validate()
            is_best = acc1 > self.best_acc1
            self.best_acc1 = max(acc1, self.best_acc1)
            output_best = f'Best Prec@1: {self.best_acc1:.3f}\n'
            print(output_best)
            if self.log_testing is not None:
                self.log_testing.write(output_best)
                self.log_testing.flush()
            save_checkpoint(self.cfg, {
                'epoch': self.epoch + 1, 'backbone': self.cfg.backbone,
                'classifier': self.cfg.classifier, 'state_dict': self.model.state_dict(),
                'best_acc1': self.best_acc1, 'optimizer': self.optimizer.state_dict()
            }, is_best, self.epoch)