import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from .base import BaseTrainer
from ..loss import LogitAdjustedLoss, BalancedSoftmaxLoss
from ..net.network import build_model
from ..utils.debug_logger import get_debug_logger

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ExpertsTrainer(BaseTrainer):
    def __init__(self, cfg, dataset, **kwargs):
        super(ExpertsTrainer, self).__init__(cfg, dataset, **kwargs)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
        self.debug = getattr(cfg, 'debug', False)
        self.debug_logger = get_debug_logger(debug=self.debug)

        self.cls_num_list = cfg.cls_num_list
        self.criterion_ce = torch.nn.CrossEntropyLoss().to(self.device)
        self.criterion_la = LogitAdjustedLoss(self.cls_num_list, tau=1.0).to(self.device)
        self.criterion_bs = BalancedSoftmaxLoss(self.cls_num_list).to(self.device)
        self.losses = [self.criterion_ce, self.criterion_la, self.criterion_bs]
        self.loss_names = ['CE', 'LA', 'BS']
        
        self.gate_split_ratio = getattr(cfg, 'gate_split_ratio', 0.9)
        self._split_dataset()
        print(f"[INFO] ExpertsTrainer initialized to train 3 independent models.")

    def _split_dataset(self):
        targets = np.array(self.train_dataset.targets)
        indices = np.arange(len(targets))
        train_idx, gate_idx = train_test_split(
            indices, test_size=1 - self.gate_split_ratio,
            stratify=targets, random_state=self.cfg.seed
        )
        expert_dataset = Subset(self.train_dataset, train_idx)
        self.train_loader = DataLoader(
            expert_dataset, batch_size=self.cfg.batch_size,
            shuffle=True, num_workers=self.cfg.workers, pin_memory=True
        )

    def get_criterion(self):
        return self.criterion_ce

    def adjust_learning_rate(self, optimizer, epoch):
        if epoch < 15:
            lr = self.cfg.learning_rate * (epoch + 1) / 15.0
        else:
            if epoch < 96: lr = self.cfg.learning_rate
            elif epoch < 192: lr = self.cfg.learning_rate * 0.1
            elif epoch < 224: lr = self.cfg.learning_rate * 0.01
            else: lr = self.cfg.learning_rate * 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train_one_epoch(self, model, optimizer, criterion, epoch):
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
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

        print(f"[INFO] Train epoch {epoch}: loss={losses.avg:.4f}, acc={top1.avg*100:.2f}%")
        return top1.avg

    def do_train_val(self):
        for i in range(3):
            print(f"\n{'='*50}\n[INFO] Training Independent Expert {i} ({self.loss_names[i]})\n{'='*50}")
            model = build_model(self.cfg)
            
            # FIX: Revert to bias=True for all experts to restore representation learning
            model.classifier = nn.Linear(model.feature_len, model.num_classes, bias=True).to(self.device)
                
            optimizer = optim.SGD(model.parameters(), lr=self.cfg.learning_rate, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
            
            for epoch in range(self.cfg.epochs):
                self.adjust_learning_rate(optimizer, epoch)
                self.train_one_epoch(model, optimizer, self.losses[i], epoch)
                
            save_path = os.path.join(self.cfg.root_model, f"expert_{i}.pth")
            os.makedirs(self.cfg.root_model, exist_ok=True)
            torch.save({'state_dict': model.state_dict()}, save_path)
            print(f"[INFO] Expert {i} saved to {save_path}")
            
        print("[INFO] All experts trained.")

    def eval_best_model(self):
        pass