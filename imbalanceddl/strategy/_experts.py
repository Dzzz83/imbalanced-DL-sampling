import os
import torch
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from .base import BaseTrainer
from ..loss import LogitAdjustedLoss, BalancedSoftmaxLoss
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
        self.model = kwargs.get('model')
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"[INFO] ExpertsTrainer: model on {self.device}")

        self.debug = getattr(cfg, 'debug', False)
        self.debug_logger = get_debug_logger(debug=self.debug)

        self.cls_num_list = cfg.cls_num_list
        self.criterion_ce = torch.nn.CrossEntropyLoss().to(self.device)
        self.criterion_la = LogitAdjustedLoss(self.cls_num_list, tau=1.0).to(self.device)
        self.criterion_bs = BalancedSoftmaxLoss(self.cls_num_list).to(self.device)
        self.losses = [self.criterion_ce, self.criterion_la, self.criterion_bs]
        
        # FIX: Define log_prior for proper posterior adjustment during validation
        cls_num_list = torch.FloatTensor(self.cls_num_list)
        probs = cls_num_list / cls_num_list.sum()
        self.log_prior = probs.log().to(self.device)

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )
        self.best_acc = 0.0
        
        self.gate_split_ratio = getattr(cfg, 'gate_split_ratio', 0.9)
        self._split_dataset()
        
        print(f"[INFO] ExpertsTrainer initialized with CE, LA (tau=1.0), BS losses.")

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
        print(f"[INFO] Expert training split size: {len(expert_dataset)}")

    def get_criterion(self):
        return self.criterion_ce

    def adjust_learning_rate(self, epoch):
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
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_one_epoch(self):
        self.model.train()
        losses = AverageMeter()
        top1 = AverageMeter()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            out, _ = self.model(images)
            experts_logits = out

            loss = 0.0
            for i, logits in enumerate(experts_logits):
                loss += self.losses[i](logits, targets)
            
            loss = loss / 3.0
            
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

            if self.debug and batch_idx == 0 and self.epoch % 10 == 0:
                self.debug_logger.debug("="*50)
                self.debug_logger.debug(f"[TRAIN EPOCH {self.epoch}] Logit Analysis")
                for i, logits in enumerate(experts_logits):
                    true_logit = logits[0, targets[0].item()].item()
                    max_logit, pred_class = torch.max(logits[0], dim=0)
                    gap = max_logit.item() - true_logit
                    self.debug_logger.debug(f"  Expert {i}: True Logit={true_logit:.4f}, Max Logit={max_logit.item():.4f} (Class {pred_class.item()}), Gap={gap:.4f}")

            # FIX: Use adjusted posteriors for accurate validation metrics
            probs = [
                torch.softmax(experts_logits[0], dim=1),
                torch.softmax(experts_logits[1] - self.log_prior, dim=1),
                torch.softmax(experts_logits[2] + self.log_prior, dim=1)
            ]
            avg_probs = torch.stack(probs, dim=0).mean(dim=0)
            _, predicted = avg_probs.max(1)
            acc = predicted.eq(targets).sum().item() / targets.size(0)
            top1.update(acc, targets.size(0))

        print(f"[INFO] Train epoch {self.epoch}: loss={losses.avg:.4f}, acc={top1.avg*100:.2f}%")
        return losses, top1

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        top1 = AverageMeter()

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            out, _ = self.model(images)
            experts_logits = out

            if self.debug and self.epoch % 10 == 0 and not hasattr(self, '_val_logged'):
                self.debug_logger.debug("="*50)
                self.debug_logger.debug(f"[VAL EPOCH {self.epoch}] Logit Analysis")
                for i, logits in enumerate(experts_logits):
                    true_logit = logits[0, targets[0].item()].item()
                    max_logit, pred_class = torch.max(logits[0], dim=0)
                    gap = max_logit.item() - true_logit
                    self.debug_logger.debug(f"  Expert {i}: True Logit={true_logit:.4f}, Max Logit={max_logit.item():.4f} (Class {pred_class.item()}), Gap={gap:.4f}")
                self._val_logged = True

            # FIX: Use adjusted posteriors for accurate validation metrics
            probs = [
                torch.softmax(experts_logits[0], dim=1),
                torch.softmax(experts_logits[1] - self.log_prior, dim=1),
                torch.softmax(experts_logits[2] + self.log_prior, dim=1)
            ]
            avg_probs = torch.stack(probs, dim=0).mean(dim=0)

            _, predicted = avg_probs.max(1)
            acc = predicted.eq(targets).sum().item() / targets.size(0)
            top1.update(acc, targets.size(0))

        print(f"[INFO] Val epoch {self.epoch}: acc={top1.avg*100:.2f}%")
        return top1

    @torch.no_grad()
    def validate_individual(self):
        self.model.eval()
        top1_avg = AverageMeter()
        top1_heads = [AverageMeter() for _ in range(3)]

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            out, _ = self.model(images)
            experts_logits = out

            if self.debug and not hasattr(self, '_logged_individual'):
                self.debug_logger.debug("="*50)
                self.debug_logger.debug("[EVAL BEST MODEL] Logit Analysis")
                for i, logits in enumerate(experts_logits):
                    true_logit = logits[0, targets[0].item()].item()
                    max_logit, pred_class = torch.max(logits[0], dim=0)
                    gap = max_logit.item() - true_logit
                    self.debug_logger.debug(f"  Expert {i}: True Logit={true_logit:.4f}, Max Logit={max_logit.item():.4f} (Class {pred_class.item()}), Gap={gap:.4f}")
                self._logged_individual = True

            # FIX: Use adjusted posteriors for accurate validation metrics
            probs = [
                torch.softmax(experts_logits[0], dim=1),
                torch.softmax(experts_logits[1] - self.log_prior, dim=1),
                torch.softmax(experts_logits[2] + self.log_prior, dim=1)
            ]
            avg_probs = torch.stack(probs, dim=0).mean(dim=0)
            _, pred_avg = avg_probs.max(1)
            acc_avg = pred_avg.eq(targets).sum().item() / targets.size(0)
            top1_avg.update(acc_avg, targets.size(0))

            for i, prob in enumerate(probs):
                _, pred_i = prob.max(1)
                acc_i = pred_i.eq(targets).sum().item() / targets.size(0)
                top1_heads[i].update(acc_i, targets.size(0))

        per_head_acc = [meter.avg * 100 for meter in top1_heads]
        avg_acc = top1_avg.avg * 100
        print(f"[INFO] Individual validation: avg_acc={avg_acc:.2f}%, CE={per_head_acc[0]:.2f}%, LA={per_head_acc[1]:.2f}%, BS={per_head_acc[2]:.2f}%")
        return avg_acc, per_head_acc

    def do_train_val(self):
        print("[INFO] Starting expert training.")
        for epoch in range(self.cfg.epochs):
            self.epoch = epoch
            self.adjust_learning_rate(epoch)
            train_losses, train_top1 = self.train_one_epoch()
            val_top1 = self.validate()

            log_msg = f"Epoch {epoch}: Train Loss {train_losses.avg:.4f} | Train Acc {train_top1.avg*100:.2f}% | Val Acc {val_top1.avg*100:.2f}%"
            self.logger.info(log_msg)
            print(log_msg)

        self.save_checkpoint(epoch, val_top1.avg)
        print("[INFO] Expert training complete.")

    def save_checkpoint(self, epoch, acc):
        os.makedirs(self.cfg.root_model, exist_ok=True)
        path = os.path.join(self.cfg.root_model, f"checkpoint_experts_epoch{epoch}.pth")
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
            'val_acc': acc,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, path)
        print(f"[INFO] Checkpoint saved: {path}")

    def eval_best_model(self):
        self.logger.info(f"=> Loading best model from {self.cfg.best_model}")
        print(f"[INFO] Loading best model from {self.cfg.best_model}")
        checkpoint = torch.load(self.cfg.best_model)
        self.model.load_state_dict(checkpoint['state_dict'])
        avg_acc, per_head_acc = self.validate_individual()
        eval_msg = (f"=> Best Model: Avg Acc = {avg_acc:.2f}% | "
                    f"CE Head = {per_head_acc[0]:.2f}% | "
                    f"LA Head = {per_head_acc[1]:.2f}% | "
                    f"BS Head = {per_head_acc[2]:.2f}%")
        self.logger.info(eval_msg)
        print(eval_msg)