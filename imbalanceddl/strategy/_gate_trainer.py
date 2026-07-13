import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from .base import BaseTrainer
from ..utils.gate_features import compute_gate_features
from ..utils.metrics import accuracy

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
        x = self.fc3(x)  # logits
        return x

class GateTrainer(BaseTrainer):
    def __init__(self, cfg, dataset, **kwargs):
        super(GateTrainer, self).__init__(cfg, dataset, **kwargs)
        # Expect a pre-trained expert model checkpoint
        self.expert_checkpoint = cfg.best_model  # or a separate arg
        if self.expert_checkpoint is None:
            raise ValueError("GateTrainer requires a pre-trained expert model checkpoint via --best_model")
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

        # Load expert model
        self.model = kwargs.get('model')  # should be the expert model architecture
        if self.model is None:
            raise ValueError("GateTrainer requires a model instance")
        checkpoint = torch.load(self.expert_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        # Freeze experts
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        print("[INFO] Expert model loaded and frozen.")

        # Split dataset into expert-training and gate splits
        self.gate_split_ratio = getattr(cfg, 'gate_split_ratio', 0.9)
        self._split_dataset()

        # Build gate MLP
        self.gate = GateMLP(
            input_dim=24,
            hidden1=cfg.gate_hidden_size,
            hidden2=cfg.gate_hidden_size2,
            num_experts=3
        ).to(self.device)

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.gate.parameters(),
            lr=cfg.gate_lr,
            weight_decay=cfg.gate_weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.gate_epochs, eta_min=0
        )
        # Warmup: we'll implement a simple linear warmup in train loop

        self.lambda_ent = cfg.lambda_ent
        self.lambda_bal = cfg.lambda_bal
        self.gate_epochs = cfg.gate_epochs
        self.best_gate_acc = 0.0

    def _split_dataset(self):
        # self.train_dataset is the full training set
        targets = np.array(self.train_dataset.targets)
        indices = np.arange(len(targets))
        # Stratified split
        train_idx, gate_idx = train_test_split(
            indices, test_size=1 - self.gate_split_ratio,
            stratify=targets, random_state=self.cfg.seed
        )
        self.gate_dataset = Subset(self.train_dataset, gate_idx)
        # Keep the train_dataset as expert-training (but we won't use it here)
        # We'll create dataloader for gate only
        self.gate_loader = DataLoader(
            self.gate_dataset, batch_size=self.cfg.batch_size,
            shuffle=True, num_workers=self.cfg.workers, pin_memory=True
        )
        # Also create a validation loader from the original validation set
        # (we'll use the same val_loader from base)
        print(f"[INFO] Gating split size: {len(self.gate_dataset)}")

    def get_criterion(self):
        return None  # not used

    def train_one_epoch(self, epoch):
        self.gate.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(self.gate_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Forward through experts (frozen)
            with torch.no_grad():
                logits_list, _ = self.model(images)  # returns list of 3 logits

            # Compute gate features
            phi = compute_gate_features(logits_list)  # (B, 24)
            gate_logits = self.gate(phi)  # (B, 3)
            weights = F.softmax(gate_logits, dim=1)  # (B, 3)

            # Compute mixture NLL
            # Get probability of true class from each expert
            probs = [F.softmax(logits, dim=1) for logits in logits_list]
            B = labels.size(0)
            # Gather the probability of the true class for each expert
            prob_true = []
            for p in probs:
                prob_true.append(p[torch.arange(B), labels])  # (B,)
            prob_true = torch.stack(prob_true, dim=1)  # (B, 3)
            # Mixture probability: sum_e w_e * p_e(y|x)
            mix_prob = (weights * prob_true).sum(dim=1)  # (B,)
            mix_nll = -torch.log(mix_prob + 1e-8).mean()

            # Entropy regularizer: negative entropy to encourage diversity
            # R_ent = -sum w log w, but we add it as positive term: -sum w log w
            ent_reg = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
            # Balance regularizer: mean weight per expert over batch
            avg_weights = weights.mean(dim=0)  # (3,)
            bal_reg = ((avg_weights - 1/3) ** 2).sum()

            loss = mix_nll + self.lambda_ent * ent_reg + self.lambda_bal * bal_reg

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            # Accuracy of mixture
            mix_pred = (weights * probs[0] + weights[:, 1:2] * probs[1] + weights[:, 2:3] * probs[2]).sum(dim=1)
            _, pred = mix_pred.max(dim=1)
            correct = pred.eq(labels).sum().item()
            total_correct += correct
            total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples * 100
        print(f"[INFO] Gate Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%")
        return avg_loss, avg_acc

    def do_train_val(self):
        # Warmup steps: we can implement simple linear warmup for first 5 epochs
        warmup_epochs = 5
        print("[INFO] Starting gate training...")
        for epoch in range(self.gate_epochs):
            # Adjust learning rate with warmup
            if epoch < warmup_epochs:
                lr = self.cfg.gate_lr * (epoch + 1) / warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                self.scheduler.step()

            loss, acc = self.train_one_epoch(epoch)
            # Validate on validation set
            val_acc = self.validate()
            if val_acc > self.best_gate_acc:
                self.best_gate_acc = val_acc
                self.save_gate_checkpoint(epoch, val_acc)
        print("[INFO] Gate training complete.")

    def validate(self):
        # Use the same validation loader from base
        self.gate.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits_list, _ = self.model(images)
                phi = compute_gate_features(logits_list)
                gate_logits = self.gate(phi)
                weights = F.softmax(gate_logits, dim=1)
                probs = [F.softmax(logits, dim=1) for logits in logits_list]
                # Apply routing sparsity (top-k)
                k = getattr(self.cfg, 'routing_sparsity', 2)
                topk_weights, topk_indices = torch.topk(weights, k, dim=1)
                # Renormalize
                topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
                # Form mixture from top-k only
                mix_prob = torch.zeros_like(probs[0])
                for i in range(k):
                    idx = topk_indices[:, i].unsqueeze(1)  # (B,1)
                    w = topk_weights[:, i].unsqueeze(1)    # (B,1)
                    # Gather the corresponding expert probabilities
                    # We need to select probs for each sample based on idx
                    # We'll stack probs along dim=1: (B, 3, C)
                    stacked_probs = torch.stack(probs, dim=1)  # (B, 3, C)
                    # Gather for each sample
                    expert_probs = stacked_probs[torch.arange(images.size(0)), idx.squeeze(1), :]  # (B, C)
                    mix_prob += w * expert_probs
                _, pred = mix_prob.max(dim=1)
                total_correct += pred.eq(labels).sum().item()
                total_samples += images.size(0)
        acc = total_correct / total_samples * 100
        print(f"[INFO] Gate Validation Acc: {acc:.2f}%")
        return acc

    def save_gate_checkpoint(self, epoch, acc):
        os.makedirs(self.cfg.root_model, exist_ok=True)
        path = os.path.join(self.cfg.root_model, f"gate_checkpoint_epoch{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'gate_state_dict': self.gate.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_acc': self.best_gate_acc,
            'val_acc': acc,
        }, path)
        print(f"[INFO] Gate checkpoint saved: {path}")

    # Override eval_best_model if needed
    def eval_best_model(self):
        # Load best gate and evaluate
        pass