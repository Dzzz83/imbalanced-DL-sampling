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
from ..utils.plugin_rule import define_groups, tune_plugin_bal, tune_plugin_worst, compute_plugin_metrics
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
        
        orig_strategy = self.cfg.strategy
        self.cfg.strategy = 'Experts'
        self.model = build_model(self.cfg)
        self.cfg.strategy = orig_strategy

        checkpoint = torch.load(self.expert_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        print("[INFO] Expert model loaded and frozen.")

        self.gate_split_ratio = getattr(cfg, 'gate_split_ratio', 0.9)
        self._split_dataset()

        self.gate = GateMLP(
            input_dim=24,
            hidden1=cfg.gate_hidden_size,
            hidden2=cfg.gate_hidden_size2,
            num_experts=3
        ).to(self.device)

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
        print("[INFO] GateTrainer initialization complete.")

    def _split_dataset(self):
        targets = np.array(self.train_dataset.targets)
        indices = np.arange(len(targets))
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
            weights = F.softmax(gate_logits, dim=1)

            probs = [F.softmax(logits, dim=1) for logits in logits_list]
            B = labels.size(0)

            prob_true = torch.stack([p[torch.arange(B), labels] for p in probs], dim=1)
            mix_prob = (weights * prob_true).sum(dim=1)
            mix_nll = -torch.log(mix_prob + 1e-8).mean()

            ent_reg = (weights * torch.log(weights + 1e-8)).sum(dim=1).mean()

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
            
        self.save_gate_checkpoint(epoch, val_acc)
        print("[INFO] Gate training complete.")

    @torch.no_grad()
    def extract_posteriors(self, loader):
        self.gate.eval()
        self.model.eval()
        
        all_p_mix = []
        all_labels = []
        
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            
            logits_list, _ = self.model(images)
            phi = compute_gate_features(logits_list)
            gate_logits = self.gate(phi)
            weights = F.softmax(gate_logits, dim=1)
            
            probs = [F.softmax(logits, dim=1) for logits in logits_list]
            
            # Top-k routing on probabilities
            k = getattr(self.cfg, 'routing_sparsity', 2)
            topk_weights, topk_indices = torch.topk(weights, k, dim=1)
            topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
            
            stacked_probs = torch.stack(probs, dim=1)
            mix_prob = torch.zeros_like(stacked_probs[:, 0, :])
            
            for i in range(k):
                idx = topk_indices[:, i]
                w = topk_weights[:, i].unsqueeze(1)
                expert_probs = stacked_probs[torch.arange(images.size(0)), idx, :]
                mix_prob += w * expert_probs

            all_p_mix.append(mix_prob.cpu().numpy())
            all_labels.append(labels.numpy())
            
        return np.concatenate(all_p_mix, axis=0), np.concatenate(all_labels, axis=0)

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
                
                stacked_probs = torch.stack(probs, dim=1)
                mix_prob = torch.zeros_like(stacked_probs[:, 0, :])
                
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

        print("[INFO] Extracting posteriors for Gate Split (Tuning)...")
        p_mix_tune, labels_tune = self.extract_posteriors(self.gate_loader)
        
        print("[INFO] Extracting posteriors for Test Split (Evaluation)...")
        p_mix_test, labels_test = self.extract_posteriors(self.val_loader)
        
        group_ids = define_groups(self.cfg.cls_num_list)
        target_rejections = np.arange(0.0, 1.1, 0.1)
        
        algo = getattr(self.cfg, 'plugin_algo', 'Bal')
        print(f"\n[INFO] Tuning Plug-in [{algo}] parameters on Gate Split...")
        
        if algo == 'Bal':
            tuned_params = tune_plugin_bal(p_mix_tune, labels_tune, group_ids, target_rejections)
        else:
            tuned_params = tune_plugin_worst(p_mix_tune, labels_tune, group_ids, target_rejections)
            
        print("\n" + "="*70)
        print(f"STAGE 3: PLUG-IN [{algo}] EVALUATION RESULTS (TEST SET)")
        print("="*70)
        print(f"{'Target Rej':<12} | {'Actual Cov':<12} | {'Bal Risk':<10} | {'Bal Acc':<10}")
        print("-"*70)
        
        for rho in target_rejections:
            if rho not in tuned_params:
                continue
                
            p = tuned_params[rho]
            preds, reject, cov, risk = compute_plugin_metrics(
                p_mix_test, labels_test, group_ids, p['alpha'], p['mu'], p['c']
            )
            
            # Balanced Accuracy = 1 - Risk (evaluated only on non-rejected samples)
            # If coverage is 0, accuracy is undefined
            bal_acc = (1.0 - risk) * 100 if cov > 0 else 0.0
            
            print(f"{rho*100:>10.1f}% | {cov*100:>10.2f}% | {risk:>10.4f} | {bal_acc:>9.2f}%")
            
        print("="*70)

    def eval_best_model(self):
        if self.cfg.best_model is not None:
            self.evaluate_gate(gate_checkpoint_path=self.cfg.best_model)
        else:
            print("[INFO] No best_model specified; evaluating current gate.")
            self.evaluate_gate()