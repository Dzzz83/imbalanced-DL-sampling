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
from ..utils.utils import AverageMeter

class ExpertsTrainer(BaseTrainer):
    def __init__(self, cfg, dataset, **kwargs):
        super(ExpertsTrainer, self).__init__(cfg, dataset, **kwargs)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.cls_num_list = cfg.cls_num_list

        # Losses
        self.criterion_ce = torch.nn.CrossEntropyLoss().to(self.device)
        self.criterion_la = LogitAdjustedLoss(self.cls_num_list, tau=1.0).to(self.device)
        self.criterion_bs = BalancedSoftmaxLoss(self.cls_num_list).to(self.device)
        self.losses = [self.criterion_ce, self.criterion_la, self.criterion_bs]
        self.loss_names = ['CE', 'LA', 'BS']

        # Which experts to train (default: all)
        self.experts_to_train = getattr(cfg, 'experts_to_train', [0, 1, 2])
        if not isinstance(self.experts_to_train, (list, tuple)):
            self.experts_to_train = [self.experts_to_train]
        self.experts_to_train = [int(i) for i in self.experts_to_train]

        # Per‑expert hyperparameters – with type conversion
        self.expert_bias = self._parse_list_typed(
            getattr(cfg, 'expert_bias', [True, True, True]), bool
        )
        self.expert_lr = self._parse_list_typed(
            getattr(cfg, 'expert_lr', [cfg.learning_rate] * 3), float
        )
        self.expert_weight_decay = self._parse_list_typed(
            getattr(cfg, 'expert_weight_decay', [cfg.weight_decay] * 3), float
        )

        self.gate_split_ratio = getattr(cfg, 'gate_split_ratio', 0.9)
        self._split_dataset()
        self.logger.info(f"[INFO] ExpertsTrainer initialized. Expert train size: {len(self.train_loader.dataset)}, Gate size: {len(self.gate_dataset)}")

        # Debug/check settings
        self.debug = getattr(cfg, 'debug', False)
        self.grad_clip_value = getattr(cfg, 'grad_clip_value', None)
        self.check_freq = getattr(cfg, 'check_freq', 10)

        # Log which experts will be trained and their exact settings
        self.logger.info(f"[INFO] Training experts: {self.experts_to_train} ({[self.loss_names[i] for i in self.experts_to_train]})")
        for i in self.experts_to_train:
            self.logger.info(
                f"[EXPERT {i} ({self.loss_names[i]})] bias={self.expert_bias[i]}, "
                f"lr={self.expert_lr[i]}, weight_decay={self.expert_weight_decay[i]}"
            )

        # DEBUG: Write a separate debug file if debug flag is True
        if self.debug:
            debug_file = os.path.join(self.cfg.root_log, 'debug_experts.txt')
            with open(debug_file, 'w') as f:
                f.write("=== EXPERTS TRAINER DEBUG ===\n")
                f.write(f"Training experts: {self.experts_to_train}\n")
                for i in self.experts_to_train:
                    f.write(f"Expert {i} ({self.loss_names[i]}): bias={self.expert_bias[i]}, lr={self.expert_lr[i]}, wd={self.expert_weight_decay[i]}\n")
            self.logger.info(f"[DEBUG] Wrote debug info to {debug_file}")

    def _parse_list_typed(self, value, dtype):
        if not isinstance(value, (list, tuple)):
            value = [value] * 3
        # Ensure length is at least 3 (pad if necessary)
        while len(value) < 3:
            value.append(value[-1] if len(value) > 0 else dtype(0))
        return [dtype(v) for v in value]

    def _split_dataset(self):
        targets = np.array(self.train_dataset.targets)
        indices = np.arange(len(targets))
        train_idx, gate_idx = train_test_split(
            indices, test_size=1 - self.gate_split_ratio,
            stratify=targets, random_state=self.cfg.seed
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

    def _check_logits(self, logits, epoch, step):
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            self.logger.error(f"[Epoch {epoch}, Step {step}] NaN/Inf detected in logits!")
            return True
        if self.debug and (epoch % self.check_freq == 0 and step == 0):
            with torch.no_grad():
                mean_val = logits.mean().item()
                std_val = logits.std().item()
                max_val = logits.max().item()
                min_val = logits.min().item()
                linf = logits.abs().max().item()
                self.logger.debug(
                    f"Logits stats: mean={mean_val:.4f}, std={std_val:.4f}, "
                    f"min={min_val:.4f}, max={max_val:.4f}, L∞={linf:.4f}"
                )
        return False

    def _check_gradients(self, model, epoch):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    self.logger.error(f"[Epoch {epoch}] NaN/Inf gradient in parameter {p.shape}!")
                    return True
        total_norm = total_norm ** 0.5
        if self.debug and (epoch % self.check_freq == 0):
            self.logger.debug(f"Total gradient norm: {total_norm:.6f}")
        if self.grad_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_value)
            if self.debug:
                self.logger.debug(f"Gradients clipped to {self.grad_clip_value}")
        return False

    def _log_weight_norms(self, model, epoch):
        if self.debug and (epoch % self.check_freq == 0):
            for name, param in model.named_parameters():
                if 'classifier' in name and param.dim() >= 2:
                    w_norm = param.data.norm(2).item()
                    self.logger.debug(f"Weight norm ({name}): {w_norm:.6f}")

    def train_one_epoch(self, model, optimizer, criterion, epoch):
        model.train()
        losses = AverageMeter('Loss', ':.4f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            logits, _ = model(images)

            if self._check_logits(logits, epoch, batch_idx):
                raise RuntimeError("Logit explosion detected. Aborting training.")

            loss = criterion(logits, targets)
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.error(f"Loss is NaN/Inf at epoch {epoch}, step {batch_idx}: {loss.item()}")
                raise RuntimeError("Loss became NaN/Inf.")

            loss.backward()

            if self._check_gradients(model, epoch):
                raise RuntimeError("Gradient explosion detected. Aborting training.")

            optimizer.step()

            losses.update(loss.item(), images.size(0))
            _, predicted = logits.max(1)
            acc = predicted.eq(targets).sum().item() / targets.size(0)
            top1.update(acc, targets.size(0))

            if self.debug and batch_idx % 100 == 0:
                self.logger.debug(f"Step {batch_idx}: loss={loss.item():.4f}, acc={acc*100:.2f}%")

        self.logger.info(f"[Train] Epoch {epoch}: loss={losses.avg:.4f}, acc={top1.avg*100:.2f}%")
        self._log_weight_norms(model, epoch)
        return top1.avg

    def validate(self, model, epoch):
        model.eval()
        top1 = AverageMeter('Acc@1', ':6.2f')
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                logits, _ = model(images)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    self.logger.error(f"[Val Epoch {epoch}] NaN/Inf in logits during validation!")
                    return 0.0
                _, pred = logits.max(1)
                acc = pred.eq(targets).sum().item() / targets.size(0)
                top1.update(acc, targets.size(0))
        self.logger.info(f"[Val] Epoch {epoch}: acc={top1.avg*100:.2f}%")
        return top1.avg

    def do_train_val(self):
        for i in self.experts_to_train:
            self.logger.info(f"\n{'='*50}\n[INFO] Training Independent Expert {i} ({self.loss_names[i]})\n{'='*50}")

            bias = self.expert_bias[i]
            lr = self.expert_lr[i]
            wd = self.expert_weight_decay[i]

            # DEBUG: Log the exact settings being used for this expert
            self.logger.info(f"[DEBUG] Expert {i} ({self.loss_names[i]}) -> bias={bias}, lr={lr}, wd={wd}")

            model = build_model(self.cfg)
            model.classifier = nn.Linear(model.feature_len, model.num_classes, bias=bias).to(self.device)
            model = model.to(self.device)

            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=self.cfg.momentum, weight_decay=wd)

            best_acc = 0.0
            for epoch in range(self.cfg.epochs):
                self.adjust_learning_rate(optimizer, epoch, base_lr=lr)
                try:
                    train_acc = self.train_one_epoch(model, optimizer, self.losses[i], epoch)
                except RuntimeError as e:
                    self.logger.error(f"Training failed: {e}")
                    break

                if (epoch + 1) % 10 == 0:
                    val_acc = self.validate(model, epoch)
                    if val_acc > best_acc:
                        best_acc = val_acc

            if best_acc == 0.0:
                self.logger.error(f"Expert {i} training failed or no validation improvement.")
            else:
                save_path = os.path.join(self.cfg.root_model, f"expert_{i}.pth")
                os.makedirs(self.cfg.root_model, exist_ok=True)
                torch.save({'state_dict': model.state_dict()}, save_path)
                self.logger.info(f"[INFO] Expert {i} saved to {save_path} (best val acc: {best_acc*100:.2f}%)")

        self.logger.info("[INFO] All requested experts trained.")

    def eval_best_model(self):
        pass

    def get_criterion(self):
        return self.criterion_ce