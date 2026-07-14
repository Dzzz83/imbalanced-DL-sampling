import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from .base import BaseTrainer
from ..utils.gate_features import compute_gate_features
from ..utils.metrics import accuracy
from ..utils.debug_logger import get_debug_logger
from ..utils.utils import AverageMeter
from ..net.network import build_model

class GateMLP(nn.Module):
    def __init__(self, input_dim=24, hidden1=256, hidden2=128, num_experts=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_experts)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GateTrainer(BaseTrainer):
    def __init__(self, cfg, dataset, **kwargs):
        self.debug = getattr(cfg, 'debug', False)
        self.debug_logger = get_debug_logger(debug=self.debug)
        print("[INFO] GateTrainer initialization started.")
        if self.debug:
            self.debug_logger.debug("GateTrainer initialization started.")

        super(GateTrainer, self).__init__(cfg, dataset, **kwargs)
        self.expert_checkpoint = cfg.expert_checkpoint
        if self.expert_checkpoint is None:
            raise ValueError("GateTrainer requires a pre-trained expert model checkpoint via --best_model")
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
        if self.debug:
            self.debug_logger.debug(f"Device: {self.device}, expert checkpoint: {self.expert_checkpoint}")

        orig_strategy = self.cfg.strategy
        self.cfg.strategy = 'Experts'
        self.model = build_model(self.cfg)
        self.cfg.strategy = orig_strategy
        if self.debug:
            self.debug_logger.debug("Built expert model with 3 independent backbones and heads.")

        if self.debug:
            self.debug_logger.debug("Loading expert checkpoint...")
        checkpoint = torch.load(self.expert_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        print("[INFO] Expert model loaded and frozen.")
        if self.debug:
            self.debug_logger.debug("Expert model loaded and frozen.")

        self.gate_split_ratio = getattr(cfg, 'gate_split_ratio', 0.9)
        if self.debug:
            self.debug_logger.debug(f"Gate split ratio: {self.gate_split_ratio}")
        self._split_dataset()

        self.gate = GateMLP(
            input_dim=24,
            hidden1=cfg.gate_hidden_size,
            hidden2=cfg.gate_hidden_size2,
            num_experts=3
        ).to(self.device)
        if self.debug:
            self.debug_logger.debug(f"Gate MLP: {self.gate}")

        self.optimizer = optim.AdamW(
            self.gate.parameters(),
            lr=cfg.gate_lr,
            weight_decay=cfg.gate_weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: (epoch + 1) / 5.0 if epoch < 5 
                                    else 0.5 * (1 + math.cos(math.pi * (epoch - 5) / (cfg.gate_epochs - 5)))
        )
        
        self.lambda_ent = cfg.lambda_ent
        self.lambda_bal = cfg.lambda_bal
        self.gate_epochs = cfg.gate_epochs
        self.best_gate_acc = 0.0
        if self.debug:
            self.debug_logger.debug(f"Gate hyperparams: lambda_ent={self.lambda_ent}, lambda_bal={self.lambda_bal}, epochs={self.gate_epochs}")
        print("[INFO] GateTrainer initialization complete.")

    def _split_dataset(self):
        targets = np.array(self.train_dataset.targets)
        indices = np.arange(len(targets))
        if self.debug:
            self.debug_logger.debug(f"Splitting dataset of size {len(targets)} with ratio {self.gate_split_ratio}")
        train_idx, gate_idx = train_test_split(
            indices, test_size=1 - self.gate_split_ratio,
            stratify=targets, random_state=self.cfg.seed
        )
        self.gate_dataset = Subset(self.train_dataset, gate_idx)
        self.gate_loader = DataLoader(
            self.gate_dataset, batch_size=self.cfg.batch_size,
            shuffle=True, num_workers=self.cfg.workers, pin_memory=True
        )
        print(f"[INFO] Gating split size: {len(self.gate_dataset)}")

    def get_criterion(self):
        return None

    def train_one_epoch(self, epoch):
        self.gate.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(self.gate_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.no_grad():
                logits_list, _ = self.model(images)

            phi = compute_gate_features(logits_list)
            gate_logits = self.gate(phi)
            weights = F.softmax(gate_logits, dim=1) # B, 3

            probs = [F.softmax(logits, dim=1) for logits in logits_list]
            B = labels.size(0)

            # L_gate: Mixture NLL
            prob_true = torch.stack([p[torch.arange(B), labels] for p in probs], dim=1) # B, 3
            mix_prob = (weights * prob_true).sum(dim=1) # B
            mix_nll = -torch.log(mix_prob + 1e-8).mean()

            # R_ent: sum(w * log(w)) 
            ent_reg = (weights * torch.log(weights + 1e-8)).sum(dim=1).mean()

            # R_bal: MSE between batch averaged weights and uniform target
            avg_weights = weights.mean(dim=0)
            bal_reg = ((avg_weights - 1.0 / 3.0) ** 2).sum()

            loss = mix_nll + self.lambda_ent * ent_reg + self.lambda_bal * bal_reg

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            
            mix_prob_full = torch.zeros_like(probs[0])
            for i in range(3):
                mix_prob_full += weights[:, i:i+1] * probs[i]
            _, pred = mix_prob_full.max(dim=1)
            
            total_correct += pred.eq(labels).sum().item()
            total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples * 100
        print(f"[INFO] Gate Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%")
        return avg_loss, avg_acc

    def do_train_val(self):
        print("[INFO] Starting gate training...")
        for epoch in range(self.gate_epochs):
            loss, acc = self.train_one_epoch(epoch)
            val_acc = self.validate()
            self.scheduler.step()
            
        # Save the final epoch gate. No test-set early stopping.
        self.save_gate_checkpoint(epoch, val_acc)
        print("[INFO] Gate training complete.")

    def validate(self):
        self.gate.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                B = images.size(0)
                
                logits_list, _ = self.model(images)
                phi = compute_gate_features(logits_list)
                gate_logits = self.gate(phi)
                weights = F.softmax(gate_logits, dim=1)
                
                probs = [F.softmax(logits, dim=1) for logits in logits_list]
                
                k = getattr(self.cfg, 'routing_sparsity', 2)
                topk_weights, topk_indices = torch.topk(weights, k, dim=1)
                topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
                
                stacked_probs = torch.stack(probs, dim=1) # B, 3, C
                mix_prob = torch.zeros_like(stacked_probs[:, 0, :]) # B, C
                
                for i in range(k):
                    idx = topk_indices[:, i]
                    w = topk_weights[:, i].unsqueeze(1)
                    expert_probs = stacked_probs[torch.arange(B), idx, :]
                    mix_prob += w * expert_probs

                _, pred = mix_prob.max(dim=1)
                total_correct += pred.eq(labels).sum().item()
                total_samples += B

        acc = total_correct / total_samples * 100
        print(f"[INFO] Gate Validation Acc: {acc:.2f}%")
        return acc

    def save_gate_checkpoint(self, epoch, acc):
        os.makedirs(self.cfg.root_model, exist_ok=True)
        path = os.path.join(self.cfg.root_model, f"gate_checkpoint_epoch{epoch}.pth")
        state = {
            'epoch': epoch,
            'gate_state_dict': self.gate.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_acc': self.best_gate_acc,
            'val_acc': acc,
        }
        torch.save(state, path)
        print(f"[INFO] Gate checkpoint saved: {path}")

    def evaluate_gate(self, gate_checkpoint_path=None):
        if gate_checkpoint_path is not None:
            checkpoint = torch.load(gate_checkpoint_path, map_location=self.device)
            self.gate.load_state_dict(checkpoint['gate_state_dict'])
            print(f"[INFO] Loaded gate checkpoint from {gate_checkpoint_path}")

        self.gate.eval()
        self.model.eval()

        top1_avg = AverageMeter('Acc')
        top1_gated = AverageMeter('Acc')
        top1_heads = [AverageMeter('Acc') for _ in range(3)]
        weight_sum = torch.zeros(3, device=self.device)
        weight_entropy_sum = 0.0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                B = images.size(0)

                logits_list, _ = self.model(images)
                probs = [F.softmax(logits, dim=1) for logits in logits_list]

                avg_prob = torch.stack(probs, dim=0).mean(dim=0)
                _, pred_avg = avg_prob.max(dim=1)
                top1_avg.update(pred_avg.eq(labels).sum().item() / B, B)

                for i, p in enumerate(probs):
                    _, pred_i = p.max(dim=1)
                    top1_heads[i].update(pred_i.eq(labels).sum().item() / B, B)

                phi = compute_gate_features(logits_list)
                gate_logits = self.gate(phi)
                weights = F.softmax(gate_logits, dim=1)

                k = getattr(self.cfg, 'routing_sparsity', 2)
                topk_weights, topk_indices = torch.topk(weights, k, dim=1)
                topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)

                stacked_probs = torch.stack(probs, dim=1)
                mix_prob = torch.zeros_like(stacked_probs[:, 0, :])
                for i in range(k):
                    idx = topk_indices[:, i]
                    w = topk_weights[:, i].unsqueeze(1)
                    expert_probs = stacked_probs[torch.arange(B), idx, :]
                    mix_prob += w * expert_probs

                _, pred_gated = mix_prob.max(dim=1)
                top1_gated.update(pred_gated.eq(labels).sum().item() / B, B)

                weight_sum += weights.sum(dim=0)
                weight_entropy_sum += -(weights * torch.log(weights + 1e-8)).sum(dim=1).sum().item()
                total_samples += B

        avg_weight = weight_sum / total_samples
        avg_weight_entropy = weight_entropy_sum / total_samples

        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Average Ensemble Accuracy: {top1_avg.avg * 100:.2f}%")
        print(f"Gated Ensemble Accuracy   : {top1_gated.avg * 100:.2f}%")
        print(f"Improvement               : {(top1_gated.avg - top1_avg.avg) * 100:+.2f}%")
        print(f"\nPer-Expert Accuracy:")
        print(f"  CE Head : {top1_heads[0].avg * 100:.2f}%")
        print(f"  LA Head : {top1_heads[1].avg * 100:.2f}%")
        print(f"  BS Head : {top1_heads[2].avg * 100:.2f}%")
        print(f"\nAverage Routing Weights (over validation set):")
        print(f"  CE: {avg_weight[0]:.3f}  LA: {avg_weight[1]:.3f}  BS: {avg_weight[2]:.3f}")
        print(f"Average Entropy of Routing Weights: {avg_weight_entropy:.4f}")
        print("="*60)

    def eval_best_model(self):
        if self.cfg.best_model is not None:
            self.evaluate_gate(gate_checkpoint_path=self.cfg.best_model)
        else:
            print("[INFO] No best_model specified; evaluating current gate.")
            self.evaluate_gate()