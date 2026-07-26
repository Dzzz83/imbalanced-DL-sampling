import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from .base import BaseTrainer
from ..loss import LogitAdjustedLoss, BalancedSoftmaxLoss
from ..net.network import build_model
from ..utils.debug_logger import get_debug_logger
from ..net.network import build_model, MultiHeadClassifier

class AverageMeter(object):
    """Computes and stores the average and current value"""
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
        
        # FIX: Label Smoothing on CE to prevent logit explosion
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_la = LogitAdjustedLoss(self.cls_num_list, tau=1.0).to(self.device)
        self.criterion_bs = BalancedSoftmaxLoss(self.cls_num_list).to(self.device)
        
        self.gate_split_ratio = getattr(cfg, 'gate_split_ratio', 0.9)
        self._split_dataset()
        self._init_file_logger()
        print("[INFO] Trained to train a shared‑backbone model with 3 heads")

    def _init_file_logger(self):
        os.makedirs(self.cfg.root_log, exist_ok=True)
        log_path = os.path.join(self.cfg.root_log, 'expert_debug.log')
        
        self.file_logger = logging.getLogger(f"ExpertDebug_{self.cfg.store_name}")
        self.file_logger.setLevel(logging.INFO)
        
        if self.file_logger.handlers:
            self.file_logger.handlers.clear()
            
        handler = logging.FileHandler(log_path, mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.file_logger.addHandler(handler)
        
        self.file_logger.info(f"Starting expert training. Store: {self.cfg.store_name}")
        self.file_logger.info(f"Config - LR: {self.cfg.learning_rate}, WD: {self.cfg.weight_decay}, Bias: False, Label Smoothing: 0.1")

    def _split_dataset(self):
        targets = np.array(self.train_dataset.targets)
        indices = np.arange(len(targets))
        train_idx, _ = train_test_split(
            indices, 
            test_size=1 - self.gate_split_ratio,
            stratify=targets, 
            random_state=self.cfg.seed
        )
        expert_dataset = Subset(self.train_dataset, train_idx)
        self.train_loader = DataLoader(
            expert_dataset, 
            batch_size=self.cfg.batch_size,
            shuffle=True, 
            num_workers=self.cfg.workers, 
            pin_memory=True
        )

    def get_criterion(self):
        return self.criterion_ce

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

    def train_one_epoch(self, model, optimizer, criterion, epoch, expert_name):
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        
        last_logits = None
        last_hidden = None
        
        for images, targets in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            logits, hidden = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), images.size(0))
            _, predicted = logits.max(1)
            acc = predicted.eq(targets).sum().item() / targets.size(0)
            top1.update(acc, targets.size(0))
            
            last_logits = logits.detach()
            last_hidden = hidden.detach()

        print(f"[INFO] Train epoch {epoch}: loss={losses.avg:.4f}, acc={top1.avg*100:.2f}%")
        
        if epoch % 5 == 0 or epoch == self.cfg.epochs - 1:
            w = model.classifier.weight.detach()
            log_msg = (
                f"[{expert_name}] Epoch {epoch} | Loss: {losses.avg:.4f} | "
                f"Weight Norm: {w.norm().item():.4f} | Weight Max: {w.max().item():.4f} | "
            )
            
            if model.classifier.bias is not None:
                b = model.classifier.bias.detach()
                log_msg += f"Bias Norm: {b.norm().item():.4f} | Bias Max: {b.max().item():.4f} | "
            
            if last_hidden is not None:
                feat_norms = last_hidden.norm(dim=1)
                log_msg += f"Feat Norm Mean: {feat_norms.mean().item():.4f} | Feat Norm Max: {feat_norms.max().item():.4f} | "
                
            if last_logits is not None:
                log_msg += f"Max Logit: {last_logits.max().item():.4f} | Min Logit: {last_logits.min().item():.4f}"
                
            self.file_logger.info(log_msg)
            
        return top1.avg

    def do_train_val(self):
        print("[INFO] Training a single model with 3 expert heads (CE, LA, BS) sharing a backbone.")
        
        model = build_model(self.cfg)
        # Replace single classifier with ModuleList of 3 heads
        model.classifier = MultiHeadClassifier(
            in_features=model.feature_len,
            out_features=model.num_classes,
            num_heads=3,
            bias=False
        ).to(self.device)
        
        # Define losses (no label smoothing)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_la = LogitAdjustedLoss(self.cls_num_list, tau=1.0).to(self.device)
        criterion_bs = BalancedSoftmaxLoss(self.cls_num_list).to(self.device)
        criteria = [criterion_ce, criterion_la, criterion_bs]
        
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.cfg.learning_rate,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay
        )
        
        for epoch in range(self.cfg.epochs):
            self.adjust_learning_rate(optimizer, epoch)
            model.train()
            for images, targets in self.train_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                logits_list, _ = model(images)  # list of 3 logits
                loss = sum(criterion(logits, targets) for criterion, logits in zip(criteria, logits_list)) / 3
                loss.backward()
                optimizer.step()
            
            # Optionally print loss/accuracy every few epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch} completed")
        
        save_path = os.path.join(self.cfg.root_model, 'expert_shared.pth')
        torch.save({'state_dict': model.state_dict()}, save_path)
        print(f"[INFO] Shared expert model saved to {save_path}")

    def eval_best_model(self):
        pass