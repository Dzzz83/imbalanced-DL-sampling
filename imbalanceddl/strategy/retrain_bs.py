import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from .base import BaseTrainer
from ..loss import BalancedSoftmaxLoss
from ..net.network import build_model
from ..utils.utils import AverageMeter

class BSRetrainer(BaseTrainer):
    def __init__(self, cfg, dataset, **kwargs):
        super(BSRetrainer, self).__init__(cfg, dataset, **kwargs)

        # FIX: Set device explicitly based on cfg.gpu
        if cfg.gpu is not None:
            self.device = torch.device(f'cuda:{cfg.gpu}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cfg = cfg
        self.cls_num_list = cfg.cls_num_list

        # Use only BS loss
        self.criterion_bs = BalancedSoftmaxLoss(self.cls_num_list).to(self.device)

        # Split the training data into 90/10 (same as before)
        self._split_dataset()

        self.logger.info("[INFO] BSRetrainer initialized. Only BS expert will be retrained.")

    def _split_dataset(self):
        targets = np.array(self.train_dataset.targets)
        indices = np.arange(len(targets))
        train_idx, gate_idx = train_test_split(
            indices, test_size=1 - 0.9,  # hardcoded 90% train, 10% gate
            stratify=targets, random_state=self.cfg.seed
        )
        expert_dataset = Subset(self.train_dataset, train_idx)
        self.train_loader = DataLoader(
            expert_dataset, batch_size=self.cfg.batch_size,
            shuffle=True, num_workers=self.cfg.workers, pin_memory=True
        )
        self.gate_dataset = Subset(self.train_dataset, gate_idx)

    def adjust_learning_rate(self, optimizer, epoch):
        if epoch < 15:
            lr = self.cfg.learning_rate * (epoch + 1) / 15.0
        else:
            if epoch < 96:
                lr = self.cfg.learning_rate
            elif epoch < 192:
                lr = self.cfg.learning_rate * 0.1
            elif epoch < 224:
                lr = self.cfg.learning_rate * 0.01
            else:
                lr = self.cfg.learning_rate * 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train_one_epoch(self, model, optimizer, criterion, epoch):
        model.train()
        losses = AverageMeter('Loss', ':.4f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        for images, targets in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), images.size(0))
            _, predicted = logits.max(1)
            acc = predicted.eq(targets).sum().item() / targets.size(0)
            top1.update(acc, targets.size(0))

        self.logger.info(f"[Train] Epoch {epoch}: loss={losses.avg:.4f}, acc={top1.avg*100:.2f}%")
        return top1.avg

    def validate(self, model, epoch):
        model.eval()
        top1 = AverageMeter('Acc@1', ':6.2f')
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                logits, _ = model(images)
                _, pred = logits.max(1)
                acc = pred.eq(targets).sum().item() / targets.size(0)
                top1.update(acc, targets.size(0))
        self.logger.info(f"[Val] Epoch {epoch}: acc={top1.avg*100:.2f}%")
        return top1.avg

    def do_train_val(self):
        self.logger.info("="*50)
        self.logger.info("RETRAINING BS EXPERT ONLY")
        self.logger.info("="*50)

        model = build_model(self.cfg)
        # Use bias=True to allow logit centering
        model.classifier = nn.Linear(model.feature_len, model.num_classes, bias=True).to(self.device)
        model = model.to(self.device)

        optimizer = optim.SGD(
            model.parameters(),
            lr=self.cfg.learning_rate,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay
        )

        best_acc = 0.0
        for epoch in range(self.cfg.epochs):
            self.adjust_learning_rate(optimizer, epoch)
            train_acc = self.train_one_epoch(model, optimizer, self.criterion_bs, epoch)
            if (epoch + 1) % 10 == 0:
                val_acc = self.validate(model, epoch)
                if val_acc > best_acc:
                    best_acc = val_acc

        # Save the retrained BS expert to the original expert directory
        save_path = os.path.join(self.cfg.root_model, "expert_2.pth")
        os.makedirs(self.cfg.root_model, exist_ok=True)
        torch.save({'state_dict': model.state_dict()}, save_path)
        self.logger.info(f"[INFO] Retrained BS expert saved to {save_path} (best val acc: {best_acc*100:.2f}%)")

    def eval_best_model(self):
        pass

    def get_criterion(self):
        return self.criterion_bs