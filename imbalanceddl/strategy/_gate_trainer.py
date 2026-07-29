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
from ..utils.debug_logger import get_debug_logger
from ..utils.plugin_rule import define_groups, define_groups_2, compute_aurc_metrics
from ..net.network import build_model
import glob

class ExpertEnsemble(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.experts = nn.ModuleList()
        expert_dir = getattr(cfg, 'expert_ckpt_dir', cfg.root_model)

        ce_bias = getattr(cfg, 'ce_bias', False)
        ce_ls = getattr(cfg, 'ce_ls', 0.0)
        la_bias = getattr(cfg, 'la_bias', False)
        la_ls = getattr(cfg, 'la_ls', 0.0)
        la_tau = getattr(cfg, 'la_tau', 1.5)
        bs_bias = getattr(cfg, 'bs_bias', False)
        bs_ls = getattr(cfg, 'bs_ls', 0.0)

        ckpt_patterns = [
            f"expert_CE_bias{ce_bias}_ls{ce_ls}_epoch*.pth",
            f"expert_LA_bias{la_bias}_ls{la_ls}_t{la_tau}_epoch*.pth",
            f"expert_BS_bias{bs_bias}_ls{bs_ls}_epoch*.pth",
        ]

        for i, pattern in enumerate(ckpt_patterns):
            files = glob.glob(os.path.join(expert_dir, pattern))
            if not files:
                fallback_name = pattern.replace("_epoch*", "_best")
                fallback_path = os.path.join(expert_dir, fallback_name)
                if os.path.isfile(fallback_path):
                    ckpt_path = fallback_path
                else:
                    raise FileNotFoundError(f"[ERROR] Expert checkpoint not found for pattern: {pattern}")
            else:
                ckpt_path = sorted(files)[-1]
            
            print(f"[INFO] Loading expert {i} from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

            has_bias = ckpt.get('bias', False)
            model = build_model(cfg)
            
            actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            actual_model.classifier = nn.Linear(actual_model.feature_len, actual_model.num_classes, bias=has_bias).to(device)
            
            state_dict = ckpt['state_dict']
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            actual_model.load_state_dict(new_state_dict)
            
            for param in actual_model.parameters():
                param.requires_grad = False
            actual_model.eval()
            self.experts.append(actual_model.to(device))

    @torch.no_grad()
    def forward(self, x):
        logits_list = []
        for expert in self.experts:
            logits, _ = expert(x)
            logits_list.append(logits)
        return logits_list, None

class GateMLP(nn.Module):
    def __init__(self, input_dim=24, hidden1=256, hidden2=128, num_experts=3, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_experts)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class GateTrainer(BaseTrainer):
    def __init__(self, cfg, dataset, **kwargs):
        self.debug = getattr(cfg, 'debug', False)
        self.debug_logger = get_debug_logger(debug=self.debug)
        print("[INFO] GateTrainer initialization started.")

        super(GateTrainer, self).__init__(cfg, dataset, **kwargs)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

        self.model = ExpertEnsemble(cfg, self.device).to(self.device)
        self.model.eval()
        self.logger.info("[INFO] Expert ensemble loaded and frozen.")

        self.gate_split_ratio = getattr(cfg, 'gate_split_ratio', 0.9)
        self._split_dataset()

        dropout = getattr(cfg, 'gate_dropout', 0.0)
        self.gate = GateMLP(
            input_dim=24,
            hidden1=cfg.gate_hidden_size,
            hidden2=cfg.gate_hidden_size2,
            num_experts=3,
            dropout=dropout
        ).to(self.device)

        self.lambda_ent = getattr(cfg, 'lambda_ent', 0.01)
        self.lambda_bal = getattr(cfg, 'lambda_bal', 0.05)
        self.gate_epochs = cfg.gate_epochs
        self.eval_interval = getattr(cfg, 'eval_interval', 10)
        self.best_gate_acc = 0.0
        
        self.logger.info("[INFO] GateTrainer initialization complete (Standard NLL Enabled).")

    def _split_dataset(self):
        if isinstance(self.train_dataset, Subset):
            all_targets = np.array(self.train_dataset.dataset.targets)
            targets = all_targets[self.train_dataset.indices]
        else:
            targets = np.array(self.train_dataset.targets)
            
        indices = np.arange(len(targets))
        train_idx, gate_idx = train_test_split(
            indices, test_size=1 - self.gate_split_ratio,
            stratify=targets, random_state=self.cfg.seed
        )
        self.gate_dataset = Subset(self.train_dataset, gate_idx)
        
        gate_bs = getattr(self.cfg, 'gating_batch_size', 128)
        self.gate_loader = DataLoader(
            self.gate_dataset, 
            batch_size=gate_bs,
            shuffle=True,
            num_workers=self.cfg.workers, 
            pin_memory=True
        )
        
        if isinstance(self.val_dataset, Subset):
            all_val_targets = np.array(self.val_dataset.dataset.targets)
            val_targets = all_val_targets[self.val_dataset.indices]
        else:
            val_targets = np.array(self.val_dataset.targets)
            
        val_indices = np.arange(len(val_targets))
        tune_idx, test_idx = train_test_split(
            val_indices, test_size=0.8, 
            stratify=val_targets, random_state=self.cfg.seed
        )
        
        self.tune_dataset = Subset(self.val_dataset, tune_idx)
        self.test_dataset = Subset(self.val_dataset, test_idx)
        
        self.tune_loader = DataLoader(self.tune_dataset, batch_size=128, shuffle=False, num_workers=self.cfg.workers, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=self.cfg.workers, pin_memory=True)
        
        self.logger.info(f"[INFO] Gating split size: {len(self.gate_dataset)} (Standard Shuffle Enabled)")
        self.logger.info(f"[INFO] Plugin Tune size: {len(self.tune_dataset)} | Final Test size: {len(self.test_dataset)}")

    def get_criterion(self):
        return None

    def get_probs(self, logits_list, T):
        p_ce = F.softmax(logits_list[0] / T, dim=1)
        
        cls_num_list = torch.tensor(self.cfg.cls_num_list, device=self.device, dtype=torch.float32)
        log_prior = torch.log(cls_num_list / cls_num_list.sum() + 1e-12)
        p_la = F.softmax((logits_list[1] + self.cfg.la_tau * log_prior) / T, dim=1)
        
        log_spc = torch.log(cls_num_list + 1e-12)
        p_bs = F.softmax((logits_list[2] + log_spc) / T, dim=1)
        
        return [p_ce, p_la, p_bs]

    @torch.no_grad()
    def log_feature_statistics(self):
        self.gate.eval()
        self.model.eval()
        all_phis = []
        
        for i, (images, _) in enumerate(self.gate_loader):
            if i >= 5: 
                break
            images = images.to(self.device)
            logits_list, _ = self.model(images)
            probs = self.get_probs(logits_list, T=1.0) 
            phi = compute_gate_features(logits_list, probs)
            all_phis.append(phi.cpu())
            
        all_phis = torch.cat(all_phis, dim=0)
        mean_phi = all_phis.mean(dim=0)
        std_phi = all_phis.std(dim=0)
        
        feat_names = [
            "CE_Ent", "CE_Max", "CE_Marg", "CE_Top5", "CE_Tail", "CE_Cos", "CE_KL",
            "LA_Ent", "LA_Max", "LA_Marg", "LA_Top5", "LA_Tail", "LA_Cos", "LA_KL",
            "BS_Ent", "BS_Max", "BS_Marg", "BS_Top5", "BS_Tail", "BS_Cos", "BS_KL",
            "Glb_MeanEnt", "Glb_ClassVar", "Glb_ConfDisp"
        ]
        
        self.logger.info("\n" + "="*80)
        self.logger.info("GATE INPUT FEATURE STATISTICS (T=1.0)")
        self.logger.info("="*80)
        self.logger.info(f"{'Feature':<15} | {'Mean':<10} | {'Std':<10}")
        self.logger.info("-"*80)
        for name, m, s in zip(feat_names, mean_phi, std_phi):
            self.logger.info(f"{name:<15} | {m:<10.4f} | {s:<10.4f}")
        self.logger.info("="*80 + "\n")

    def train_one_epoch(self, epoch, T, gate_loader, optimizer, scheduler):
        self.gate.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        total_weights_sum = torch.zeros(3, device=self.device)

        for batch_idx, (images, labels) in enumerate(gate_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.no_grad():
                logits_list, _ = self.model(images)

            probs = self.get_probs(logits_list, T)
            phi = compute_gate_features(logits_list, probs)

            gate_logits = self.gate(phi)
            weights = F.softmax(gate_logits, dim=1)
            B = labels.size(0)

            prob_true = torch.stack([p[torch.arange(B), labels] for p in probs], dim=1)
            mix_prob = (weights * prob_true).sum(dim=1)
            
            mix_nll = -torch.log(mix_prob + 1e-8).mean()

            ent_reg = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
            batch_avg_weights = weights.mean(dim=0)
            bal_reg = ((batch_avg_weights - 1.0 / 3.0) ** 2).sum()

            loss = mix_nll - self.lambda_ent * ent_reg + self.lambda_bal * bal_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_weights_sum += weights.sum(dim=0)

            mix_prob_full = torch.zeros_like(probs[0])
            for i in range(3):
                mix_prob_full += weights[:, i:i+1] * probs[i]
            _, pred = mix_prob_full.max(dim=1)
            total_correct += pred.eq(labels).sum().item()
            total_samples += images.size(0)

        scheduler.step()
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples * 100
        
        epoch_avg_weights = total_weights_sum / total_samples
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            w_ce, w_la, w_bs = epoch_avg_weights[0].item(), epoch_avg_weights[1].item(), epoch_avg_weights[2].item()
            log_line = f"  Epoch {epoch+1} Routing -> Avg Weights: CE={w_ce:.3f}, LA={w_la:.3f}, BS={w_bs:.3f}"
            print(log_line)
            self.logger.info(log_line)
            
        return avg_loss, avg_acc

    def do_train_val(self):
        self.log_feature_statistics()
        
        batch_sizes = getattr(self.cfg, 'gate_batch_sizes', [128])
        temperatures = getattr(self.cfg, 'gate_temperatures', [1.0])
        
        self.logger.info(f"\n[INFO] Starting Gate Sweep. Batch Sizes: {batch_sizes}, Temperatures: {temperatures}")
        
        sweep_results = []

        for bs in batch_sizes:
            gate_loader = DataLoader(
                self.gate_dataset, batch_size=bs,
                shuffle=True, num_workers=self.cfg.workers, pin_memory=True
            )

            for T in temperatures:
                print("\n" + "#"*80)
                print(f"# SWEEPING GATE: Batch Size = {bs}, Temperature = {T}")
                print("#"*80)
                self.logger.info(f"SWEEP: Batch Size={bs}, Temp={T}")

                self._reset_gate_and_optimizer()
                
                best_gate_nll = 1e9
                best_epoch = -1

                for epoch in range(self.gate_epochs):
                    loss, acc = self.train_one_epoch(epoch, T, gate_loader, self.optimizer, self.scheduler)
                    
                    p_mix_gate, labels_gate = self.extract_posteriors(self.gate_loader, T)
                    
                    true_probs = p_mix_gate[np.arange(len(labels_gate)), labels_gate]
                    current_nll = -np.mean(np.log(true_probs + 1e-8))

                    if (epoch + 1) % 10 == 0 or epoch == 0:
                        print(f"  Epoch {epoch+1}/{self.gate_epochs}: train_loss={loss:.4f}, gate_acc={acc:.2f}%, Gate NLL: {current_nll:.4f}")

                    if current_nll < best_gate_nll:
                        best_gate_nll = current_nll
                        best_epoch = epoch
                        self.save_gate_checkpoint(epoch, bs, T, is_best=True)
                        
                sweep_results.append({
                    'batch_size': bs, 'temp': T, 'best_epoch': best_epoch + 1, 'best_nll': best_gate_nll
                })
                print(f"[INFO] Finished BS={bs}, T={T}. Best Epoch: {best_epoch+1} with NLL: {best_gate_nll:.4f}")

        print("\n" + "="*100)
        print("GATE SWEEP FINAL SUMMARY")
        print("="*100)
        print(f"{'BS':<5} | {'T':<5} | {'Best Epoch':<10} | {'Best NLL':<20}")
        print("-"*50)
        for r in sweep_results:
            print(f"{r['batch_size']:<5} | {r['temp']:<5} | {r['best_epoch']:<10} | {r['best_nll']:<20.4f}")
        print("="*100)
        
        self.eval_best_model()

    def save_gate_checkpoint(self, epoch, bs, T, is_best=False):
        os.makedirs(self.cfg.root_model, exist_ok=True)
        path = os.path.join(self.cfg.root_model, f"gate_checkpoint_bs{bs}_T{T}_epoch{epoch}.pth")
        state = {
            'epoch': epoch,
            'batch_size': bs,
            'temperature': T,
            'gate_state_dict': self.gate.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if is_best:
            torch.save(state, path)
            self.logger.info(f"New Best Gate NLL found! Saved checkpoint: {path}")

    def _reset_gate_and_optimizer(self):
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.gate.apply(weight_init)
        self.optimizer = optim.AdamW(
            self.gate.parameters(),
            lr=self.cfg.gate_lr,
            weight_decay=self.cfg.gate_weight_decay
        )
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: (epoch + 1) / 5.0 if epoch < 5
                                    else 0.5 * (1 + math.cos(math.pi * (epoch - 5) / (self.cfg.gate_epochs - 5)))
        )

    @torch.no_grad()
    def extract_posteriors(self, loader, T):
        self.gate.eval()
        self.model.eval()
        all_p_mix = []
        all_labels = []
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            logits_list, _ = self.model(images)
            probs = self.get_probs(logits_list, T)
            phi = compute_gate_features(logits_list, probs)

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
                expert_probs = stacked_probs[torch.arange(images.size(0)), idx, :]
                mix_prob += w * expert_probs

            all_p_mix.append(mix_prob.cpu().numpy())
            all_labels.append(labels.numpy())
        return np.concatenate(all_p_mix, axis=0), np.concatenate(all_labels, axis=0)

    def eval_best_model(self):
        self.logger.info("\n" + "="*80)
        self.logger.info("STAGE 3: PLUG-IN EVALUATION")
        self.logger.info("="*80)
        
        files = glob.glob(os.path.join(self.cfg.root_model, "gate_checkpoint_*.pth"))
        if not files:
            self.logger.error("Best gate checkpoint not found! Run training first.")
            return
            
        best_gate_path = max(files, key=os.path.getmtime)
            
        ckpt = torch.load(best_gate_path, map_location='cpu')
        self.gate.load_state_dict(ckpt['gate_state_dict'])
        T = ckpt['temperature']
        self.logger.info(f"Loaded best gate from {best_gate_path} (Epoch {ckpt['epoch']}) with T={T}")
        
        self.logger.info("Extracting posteriors for tune (val) and test sets...")
        p_mix_val, labels_val = self.extract_posteriors(self.tune_loader, T)
        p_mix_test, labels_test = self.extract_posteriors(self.test_loader, T)
        
        group_ids = define_groups_2(self.cfg.cls_num_list)
        
        mode = self.cfg.plugin_algo
        self.logger.info(f"Running Plug-in [{mode}] evaluation...")
        
        metrics = compute_aurc_metrics(
            p_mix_val, labels_val, 
            p_mix_test, labels_test, 
            group_ids, 
            cls_num_list=self.cfg.cls_num_list, 
            mode=mode
        )
        
        self.logger.info("\n" + "-"*40)
        self.logger.info(f"AURC: {metrics['AURC']:.4f}")
        self.logger.info(f"NLL: {metrics['NLL']:.4f}")
        self.logger.info(f"Brier: {metrics['Brier']:.4f}")
        self.logger.info(f"tail-ECE: {metrics['tail-ECE']:.4f}")
        self.logger.info("-"*40 + "\n")
        print(f"Plug-in [{mode}] Results -> AURC: {metrics['AURC']:.4f} | NLL: {metrics['NLL']:.4f} | Brier: {metrics['Brier']:.4f} | tail-ECE: {metrics['tail-ECE']:.4f}")