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

        # ------------------------------------------------------------
        # Build an expert model with 3 heads (Experts architecture)
        # ------------------------------------------------------------
        orig_strategy = self.cfg.strategy
        self.cfg.strategy = 'Experts'
        # build_model uses cfg.gpu etc.; it returns a model already on the correct device
        self.model = build_model(self.cfg)
        self.cfg.strategy = orig_strategy
        if self.debug:
            self.debug_logger.debug("Built expert model with 3 heads.")

        # Load checkpoint
        if self.debug:
            self.debug_logger.debug("Loading expert checkpoint...")
        checkpoint = torch.load(self.expert_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        # Freeze experts
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        print("[INFO] Expert model loaded and frozen.")
        if self.debug:
            self.debug_logger.debug("Expert model loaded and frozen.")

        # Split dataset
        self.gate_split_ratio = getattr(cfg, 'gate_split_ratio', 0.9)
        if self.debug:
            self.debug_logger.debug(f"Gate split ratio: {self.gate_split_ratio}")
        self._split_dataset()

        # Build gate MLP
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
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.gate_epochs, eta_min=0
        )
        self.lambda_ent = cfg.lambda_ent
        self.lambda_bal = cfg.lambda_bal
        self.gate_epochs = cfg.gate_epochs
        self.best_gate_acc = 0.0
        if self.debug:
            self.debug_logger.debug(f"Gate hyperparams: lambda_ent={self.lambda_ent}, lambda_bal={self.lambda_bal}, epochs={self.gate_epochs}")
        print("[INFO] GateTrainer initialization complete.")
        if self.debug:
            self.debug_logger.debug("GateTrainer initialization complete.")

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
        if self.debug:
            self.debug_logger.debug(f"Gating split size: {len(self.gate_dataset)}")
            self.debug_logger.debug(f"First 5 gate indices: {gate_idx[:5]}")

    def get_criterion(self):
        return None

    def train_one_epoch(self, epoch):
        self.gate.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        if self.debug:
            self.debug_logger.debug(f"Training gate epoch {epoch}")

        for batch_idx, (images, labels) in enumerate(self.gate_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if self.debug and batch_idx == 0:
                self.debug_logger.debug(f"Batch 0: images shape {images.shape}, labels shape {labels.shape}")

            with torch.no_grad():
                logits_list, _ = self.model(images)
                if self.debug and batch_idx == 0:
                    self.debug_logger.debug(f"Expert logits shapes: {[logits.shape for logits in logits_list]}")

            phi = compute_gate_features(logits_list)
            if self.debug and batch_idx == 0:
                self.debug_logger.debug(f"Gate features shape: {phi.shape}")

            gate_logits = self.gate(phi)
            weights = F.softmax(gate_logits, dim=1)
            if self.debug and batch_idx == 0:
                self.debug_logger.debug(f"Gate weights sample: {weights[0].detach().cpu().numpy()}")

            probs = [F.softmax(logits, dim=1) for logits in logits_list]
            B = labels.size(0)

            # Loss: mixture NLL using probability of true class
            prob_true = []
            for p in probs:
                prob_true.append(p[torch.arange(B), labels])
            prob_true = torch.stack(prob_true, dim=1)  # (B, 3)
            mix_prob = (weights * prob_true).sum(dim=1)
            mix_nll = -torch.log(mix_prob + 1e-8).mean()

            ent_reg = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
            avg_weights = weights.mean(dim=0)
            bal_reg = ((avg_weights - 1/3) ** 2).sum()

            loss = mix_nll + self.lambda_ent * ent_reg + self.lambda_bal * bal_reg
            if self.debug and batch_idx == 0:
                self.debug_logger.debug(f"Loss components: mix_nll={mix_nll.item():.4f}, ent_reg={ent_reg.item():.4f}, bal_reg={bal_reg.item():.4f}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)

            # Accuracy: full mixture distribution
            mix_prob_full = weights[:, 0:1] * probs[0] + weights[:, 1:2] * probs[1] + weights[:, 2:3] * probs[2]
            _, pred = mix_prob_full.max(dim=1)
            correct = pred.eq(labels).sum().item()
            total_correct += correct
            total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples * 100
        print(f"[INFO] Gate Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%")
        if self.debug:
            self.debug_logger.debug(f"Gate Epoch {epoch} completed: avg_loss={avg_loss:.4f}, avg_acc={avg_acc:.2f}%")
        return avg_loss, avg_acc

    def do_train_val(self):
        warmup_epochs = 5
        print("[INFO] Starting gate training...")
        if self.debug:
            self.debug_logger.debug("Starting gate training loop.")
        for epoch in range(self.gate_epochs):
            if epoch < warmup_epochs:
                lr = self.cfg.gate_lr * (epoch + 1) / warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                if self.debug and epoch == 0:
                    self.debug_logger.debug(f"Warmup epoch {epoch}: lr={lr}")
            else:
                self.scheduler.step()
                if self.debug and epoch == warmup_epochs:
                    self.debug_logger.debug("End of warmup, scheduler step applied.")

            loss, acc = self.train_one_epoch(epoch)
            val_acc = self.validate()
            if self.debug:
                self.debug_logger.debug(f"Epoch {epoch}: validation acc={val_acc:.2f}%")
            if val_acc > self.best_gate_acc:
                self.best_gate_acc = val_acc
                self.save_gate_checkpoint(epoch, val_acc)
        print("[INFO] Gate training complete.")
        if self.debug:
            self.debug_logger.debug("Gate training complete.")

    def validate(self):
        self.gate.eval()
        total_correct = 0
        total_samples = 0
        if self.debug:
            self.debug_logger.debug("Starting validation.")
        with torch.no_grad():
            for idx, (images, labels) in enumerate(self.val_loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits_list, _ = self.model(images)
                phi = compute_gate_features(logits_list)
                gate_logits = self.gate(phi)
                weights = F.softmax(gate_logits, dim=1)
                probs = [F.softmax(logits, dim=1) for logits in logits_list]
                k = getattr(self.cfg, 'routing_sparsity', 2)
                topk_weights, topk_indices = torch.topk(weights, k, dim=1)
                topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
                mix_prob = torch.zeros_like(probs[0])
                for i in range(k):
                    idx_exp = topk_indices[:, i].unsqueeze(1)
                    w = topk_weights[:, i].unsqueeze(1)
                    stacked_probs = torch.stack(probs, dim=1)
                    expert_probs = stacked_probs[torch.arange(images.size(0)), idx_exp.squeeze(1), :]
                    mix_prob += w * expert_probs
                _, pred = mix_prob.max(dim=1)
                total_correct += pred.eq(labels).sum().item()
                total_samples += images.size(0)
                if self.debug and idx == 0:
                    self.debug_logger.debug(f"Validation batch 0: weights sample {weights[0].detach().cpu().numpy()}")
        acc = total_correct / total_samples * 100
        print(f"[INFO] Gate Validation Acc: {acc:.2f}%")
        if self.debug:
            self.debug_logger.debug(f"Validation accuracy: {acc:.2f}%")
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
        if self.debug:
            self.debug_logger.debug(f"Gate checkpoint saved at {path}")

    def evaluate_gate(self, gate_checkpoint_path=None):
        """
        Evaluate the gated ensemble on the validation set.
        If gate_checkpoint_path is provided, load that gate; otherwise use the current gate.
        """
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

                # Average ensemble
                avg_prob = torch.stack(probs, dim=0).mean(dim=0)
                _, pred_avg = avg_prob.max(dim=1)
                acc_avg_frac = pred_avg.eq(labels).sum().item() / B
                top1_avg.update(acc_avg_frac, B)

                # Individual heads
                for i, p in enumerate(probs):
                    _, pred_i = p.max(dim=1)
                    acc_i_frac = pred_i.eq(labels).sum().item() / B
                    top1_heads[i].update(acc_i_frac, B)

                # Gated ensemble
                phi = compute_gate_features(logits_list)
                gate_logits = self.gate(phi)
                weights = F.softmax(gate_logits, dim=1)

                # Apply sparsity (top-k)
                k = getattr(self.cfg, 'routing_sparsity', 2)
                topk_weights, topk_indices = torch.topk(weights, k, dim=1)
                topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)

                # Build mixture from top-k
                mix_prob = torch.zeros_like(probs[0])
                for i in range(k):
                    idx = topk_indices[:, i].unsqueeze(1)
                    w = topk_weights[:, i].unsqueeze(1)
                    stacked_probs = torch.stack(probs, dim=1)
                    expert_probs = stacked_probs[torch.arange(B), idx.squeeze(1), :]
                    mix_prob += w * expert_probs

                _, pred_gated = mix_prob.max(dim=1)
                acc_gated_frac = pred_gated.eq(labels).sum().item() / B
                top1_gated.update(acc_gated_frac, B)

                # Accumulate weight statistics
                weight_sum += weights.sum(dim=0)
                weight_entropy_sum += -(weights * torch.log(weights + 1e-8)).sum(dim=1).sum().item()
                total_samples += B

        avg_weight = weight_sum / total_samples
        avg_weight_entropy = weight_entropy_sum / total_samples

        avg_acc = top1_avg.avg * 100
        gated_acc = top1_gated.avg * 100
        head_accs = [m.avg * 100 for m in top1_heads]

        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Average Ensemble Accuracy: {avg_acc:.2f}%")
        print(f"Gated Ensemble Accuracy   : {gated_acc:.2f}%")
        print(f"Improvement               : {gated_acc - avg_acc:+.2f}%")
        print(f"\nPer-Expert Accuracy:")
        print(f"  CE Head : {head_accs[0]:.2f}%")
        print(f"  LA Head : {head_accs[1]:.2f}%")
        print(f"  BS Head : {head_accs[2]:.2f}%")
        print(f"\nAverage Routing Weights (over validation set):")
        print(f"  CE: {avg_weight[0]:.3f}  LA: {avg_weight[1]:.3f}  BS: {avg_weight[2]:.3f}")
        print(f"Average Entropy of Routing Weights: {avg_weight_entropy:.4f}")
        print("="*60)

        return {
            'avg_ensemble_acc': avg_acc,
            'gated_ensemble_acc': gated_acc,
            'head_accs': head_accs,
            'avg_weights': avg_weight.cpu().numpy(),
            'avg_entropy': avg_weight_entropy
        }

    def eval_best_model(self):
        # Load the best gate checkpoint (you need to specify it in config)
        if self.cfg.best_model is not None:
            self.evaluate_gate(gate_checkpoint_path=self.cfg.best_model)
        else:
            print("[INFO] No best_model specified; evaluating current gate.")
            self.evaluate_gate()