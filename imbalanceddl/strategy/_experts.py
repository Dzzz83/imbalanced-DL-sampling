import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from .base import BaseTrainer
from ..loss import LogitAdjustedLoss, BalancedSoftmaxLoss
from ..net.network import build_model
from ..utils.utils import AverageMeter
from ..utils.metrics import shot_acc

class ExpertsTrainer(BaseTrainer):
    def __init__(self, cfg, dataset, **kwargs):
        super(ExpertsTrainer, self).__init__(cfg, dataset, **kwargs)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.cls_num_list = cfg.cls_num_list
        
        self.gate_split_ratio = getattr(cfg, 'gate_split_ratio', 0.9)
        self._split_dataset()
        self.logger.info(f"[INFO] ExpertsTrainer initialized. Expert train size: {len(self.train_loader.dataset)}, Gate size: {len(self.gate_dataset)}")

        self.debug = getattr(cfg, 'debug', False)
        self.grad_clip_value = getattr(cfg, 'grad_clip_value', 5.0)

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
        self.gate_dataset = Subset(self.train_dataset, gate_idx)
        self.train_loader = DataLoader(
            expert_dataset, batch_size=self.cfg.batch_size,
            shuffle=True, num_workers=self.cfg.workers, pin_memory=True
        )

    def adjust_learning_rate(self, optimizer, epoch, base_lr):
        if epoch < 15:
            lr = base_lr * (epoch + 1) / 15.0
        else:
            if epoch < 96:
                lr = base_lr
            elif epoch < 192:
                lr = base_lr * 0.1
            elif epoch < 224:
                lr = base_lr * 0.01
            else:
                lr = base_lr * 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train_one_epoch(self, model, optimizer, criterion, epoch, expert_name):
        model.train()
        losses = AverageMeter('Loss', ':.4f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            logits, _ = model(images)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                raise RuntimeError(f"Logit explosion detected at epoch {epoch}, step {batch_idx}.")

            loss = criterion(logits, targets)
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"Loss is NaN/Inf at epoch {epoch}, step {batch_idx}")

            loss.backward()
            
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_value)

            optimizer.step()

            losses.update(loss.item(), images.size(0))
            _, predicted = logits.max(1)
            acc = predicted.eq(targets).sum().item() / targets.size(0)
            top1.update(acc, targets.size(0))
            
            last_logits = logits.detach()
            last_hidden = hidden.detach()

        self.logger.info(f"[Train] Epoch {epoch}: loss={losses.avg:.4f}, acc={top1.avg*100:.2f}%")
        return top1.avg

    def validate(self, model):
        model.eval()
        top1 = AverageMeter('Acc@1', ':6.2f')
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                logits, _ = model(images)
                probs = F.softmax(logits, dim=1)
                
                _, pred = logits.max(1)
                acc = pred.eq(targets).sum().item() / targets.size(0)
                top1.update(acc, targets.size(0))
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.concatenate(all_probs, axis=0)

        many_acc, median_acc, low_acc = shot_acc(
            self.cfg, all_preds, all_targets, self.train_dataset, acc_per_cls=False
        )
        
        # Compute NLL for calibration-aware expert selection
        nll = -np.mean(np.log(all_probs[np.arange(len(all_targets)), all_targets] + 1e-8))

        return {
            'acc': top1.avg * 100,
            'many': many_acc * 100,
            'med': median_acc * 100,
            'low': low_acc * 100,
            'nll': nll
        }

    def do_train_val(self):
        os.makedirs(self.cfg.root_model, exist_ok=True)
        
        sweep_taus = [1.0, 1.5, 2.0]
        sweep_biases = [False, True]
        sweep_ls = [0.0, 0.1]
        
        sweep_results = []

        for bias in sweep_biases:
            for ls in sweep_ls:
                # 1. Train CE Expert
                run_name = f"CE_bias{bias}_ls{ls}"
                self.logger.info(f"\n{'='*50}\n[INFO] Training Independent Expert: {run_name}\n{'='*50}")
                model = build_model(self.cfg)
                actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                actual_model.classifier = nn.Linear(actual_model.feature_len, actual_model.num_classes, bias=bias).to(self.device)
                model = model.to(self.device)
                
                optimizer = optim.SGD(model.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
                criterion_ce = torch.nn.CrossEntropyLoss(label_smoothing=ls).to(self.device)
                
                best_metric = 1e9
                best_epoch = 0
                best_state_dict = None
                
                for epoch in range(self.cfg.epochs):
                    self.adjust_learning_rate(optimizer, epoch, base_lr=self.cfg.lr)
                    self.train_one_epoch(model, optimizer, criterion_ce, epoch)
                    metrics = self.validate(model)
                    
                    if metrics['nll'] < best_metric:
                        best_metric = metrics['nll']
                        best_epoch = epoch
                        best_state_dict = copy.deepcopy(model.state_dict())
                
                best_save_path = os.path.join(self.cfg.root_model, f"expert_CE_bias{bias}_ls{ls}_best.pth")
                torch.save({'state_dict': best_state_dict, 'bias': bias, 'tau': None, 'label_smoothing': ls}, best_save_path)
                
                metrics['name'] = run_name
                metrics['best_metric'] = best_metric
                sweep_results.append(metrics)
                self.logger.info(f"[INFO] Expert {run_name} saved to {best_save_path}")
                del model, optimizer, best_state_dict
                torch.cuda.empty_cache()

                # 2. Train BS Expert
                run_name = f"BS_bias{bias}_ls{ls}"
                self.logger.info(f"\n{'='*50}\n[INFO] Training Independent Expert: {run_name}\n{'='*50}")
                model = build_model(self.cfg)
                actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                actual_model.classifier = nn.Linear(actual_model.feature_len, actual_model.num_classes, bias=bias).to(self.device)
                model = model.to(self.device)
                
                optimizer = optim.SGD(model.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
                criterion_bs = BalancedSoftmaxLoss(self.cls_num_list, label_smoothing=ls).to(self.device)
                
                best_metric = 1e9
                best_epoch = 0
                best_state_dict = None
                
                for epoch in range(self.cfg.epochs):
                    self.adjust_learning_rate(optimizer, epoch, base_lr=self.cfg.lr)
                    self.train_one_epoch(model, optimizer, criterion_bs, epoch)
                    metrics = self.validate(model)
                    
                    if metrics['nll'] < best_metric:
                        best_metric = metrics['nll']
                        best_epoch = epoch
                        best_state_dict = copy.deepcopy(model.state_dict())
                
                best_save_path = os.path.join(self.cfg.root_model, f"expert_BS_bias{bias}_ls{ls}_best.pth")
                torch.save({'state_dict': best_state_dict, 'bias': bias, 'tau': None, 'label_smoothing': ls}, best_save_path)
                
                metrics['name'] = run_name
                metrics['best_metric'] = best_metric
                sweep_results.append(metrics)
                self.logger.info(f"[INFO] Expert {run_name} saved to {best_save_path}")
                del model, optimizer, best_state_dict
                torch.cuda.empty_cache()

                # 3. Train LA Expert for all taus
                for tau in sweep_taus:
                    run_name = f"LA_bias{bias}_ls{ls}_t{tau}"
                    self.logger.info(f"\n{'='*50}\n[INFO] Training Independent Expert: {run_name}\n{'='*50}")
                    model = build_model(self.cfg)
                    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                    actual_model.classifier = nn.Linear(actual_model.feature_len, actual_model.num_classes, bias=bias).to(self.device)
                    model = model.to(self.device)
                    
                    criterion_la = LogitAdjustedLoss(self.cls_num_list, tau=tau, label_smoothing=ls).to(self.device)
                    optimizer = optim.SGD(model.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
                    
                    best_metric = 1e9
                    best_epoch = 0
                    best_state_dict = None
                    
                    for epoch in range(self.cfg.epochs):
                        self.adjust_learning_rate(optimizer, epoch, base_lr=self.cfg.lr)
                        self.train_one_epoch(model, optimizer, criterion_la, epoch)
                        metrics = self.validate(model)
                        
                        if metrics['nll'] < best_metric:
                            best_metric = metrics['nll']
                            best_epoch = epoch
                            best_state_dict = copy.deepcopy(model.state_dict())
                    
                    best_save_path = os.path.join(self.cfg.root_model, f"expert_LA_bias{bias}_ls{ls}_t{tau}_best.pth")
                    torch.save({'state_dict': best_state_dict, 'bias': bias, 'tau': tau, 'label_smoothing': ls}, best_save_path)
                    
                    metrics['name'] = run_name
                    metrics['best_metric'] = best_metric
                    sweep_results.append(metrics)
                    self.logger.info(f"[INFO] Expert {run_name} saved to {best_save_path}")
                    del model, optimizer, best_state_dict
                    torch.cuda.empty_cache()

        self.logger.info("\n" + "="*100)
        self.logger.info("STAGE 1 SWEEP SUMMARY TABLE")
        self.logger.info("="*100)
        header = f"{'Run Name':<30} | {'Bal Acc':<8} | {'Many':<6} | {'Med':<6} | {'Low':<6} | {'NLL':<6}"
        self.logger.info(header)
        self.logger.info("-"*100)
        for r in sweep_results:
            row = f"{r['name']:<30} | {r['acc']:<8.2f} | {r['many']:<6.2f} | {r['med']:<6.2f} | {r['low']:<6.2f} | {r['nll']:<6.3f}"
            self.logger.info(row)
        self.logger.info("="*100)

    def eval_best_model(self):
        pass

    def get_criterion(self):
        return torch.nn.CrossEntropyLoss()