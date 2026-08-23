import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
from .base import BaseTrainer
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
        # Logit-level routing: gate sees the concatenated raw 100-dim logits (300-dim)
        embeddings = torch.cat(logits_list, dim=1)
        return logits_list, embeddings

class GateMLP(nn.Module):
    """Non-linear logit router with batch-normalized input standardization.

    Replaces the naive ``LayerNorm -> Linear`` peak-detector. The hidden
    ReLU layer lets the gate suppress overconfident-but-wrong experts
    conditionally, and BatchNorm1d standardizes the 300-dim logit scales
    across the batch (CE/LA/BS can live on very different magnitudes).

    Architecture: BatchNorm1d(300) -> Linear(300, 64) -> ReLU -> Linear(64, 3).

    ``fc`` (hidden projection) keeps the legacy attribute name so
    ``GateTrainer.train_one_epoch`` can still log ``gate.fc.weight.grad``.
    """

    def __init__(self, input_dim=300, num_experts=3, hidden_dim=64):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc_out = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        x = self.bn(x)
        x = self.act(self.fc(x))
        x = self.fc_out(x)
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

        self.gate = GateMLP(
            input_dim=300,  # Concatenated raw logits (100-dim x 3 experts)
            num_experts=3
        ).to(self.device)

        self.gate_epochs = cfg.gate_epochs
        self.eval_interval = getattr(cfg, 'eval_interval', 1)
        self.best_gate_acc = 0.0
        
        self.logger.info("[INFO] GateTrainer initialization complete (Supervised Routing Enabled).")

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

        # Inverse-class-frequency sampling: give every class equal expected
        # coverage so Head/Tail classes are seen equally during gate training.
        gate_targets = targets[gate_idx]
        class_counts = np.bincount(gate_targets, minlength=self.cfg.num_classes).astype(np.float64)
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = class_weights[gate_targets]
        self.gate_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        gate_bs = getattr(self.cfg, 'gating_batch_size', 128)
        self.gate_loader = DataLoader(
            self.gate_dataset,
            batch_size=gate_bs,
            sampler=self.gate_sampler,
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
        
        self.logger.info(f"[INFO] Gating split size: {len(self.gate_dataset)} (WeightedRandomSampler Enabled)")
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
    def validate(self, T):
        self.gate.eval()
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_oracle_matches = []
        
        for images, labels in self.tune_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            logits_list, embeddings = self.model(images)
            probs = self.get_probs(logits_list, T)
            
            gate_logits = self.gate(embeddings)
            weights = F.softmax(gate_logits, dim=1)
            
            mix_prob = torch.zeros_like(probs[0])
            for i in range(3):
                mix_prob += weights[:, i:i+1] * probs[i]
                
            _, pred = mix_prob.max(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            B = labels.size(0)
            true_probs_experts = torch.stack([p[torch.arange(B), labels] for p in probs], dim=1)
            target_expert = torch.argmax(true_probs_experts, dim=1)
            gate_choice = torch.argmax(weights, dim=1)
            all_oracle_matches.extend(gate_choice.eq(target_expert).cpu().numpy())
            
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute strict Balanced Accuracy
        bal_acc = np.mean([np.mean(all_preds[all_labels == c] == c) for c in range(self.cfg.num_classes) if np.sum(all_labels == c) > 0]) * 100

        # Oracle Match Accuracy: fraction of samples where the gate's routing
        # choice equals the oracle expert (highest true-class probability).
        oracle_match_acc = np.mean(all_oracle_matches) * 100

        return bal_acc, oracle_match_acc

    @torch.no_grad()
    def log_feature_statistics(self):
        self.gate.eval()
        self.model.eval()
        all_embeddings = []
        
        for i, (images, _) in enumerate(self.gate_loader):
            if i >= 5: 
                break
            images = images.to(self.device)
            _, embeddings = self.model(images)
            all_embeddings.append(embeddings.cpu())
            
        all_embeddings = torch.cat(all_embeddings, dim=0)
        mean_emb = all_embeddings.mean(dim=0)
        std_emb = all_embeddings.std(dim=0)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("GATE INPUT EMBEDDING STATISTICS (300-dim)")
        self.logger.info("="*80)
        self.logger.info(f"Global Mean: {mean_emb.mean().item():.4f} | Global Std: {std_emb.mean().item():.4f}")
        self.logger.info(f"Min Val: {all_embeddings.min().item():.4f} | Max Val: {all_embeddings.max().item():.4f}")
        self.logger.info("="*80 + "\n")

    def train_one_epoch(self, epoch, T, gate_loader, optimizer, scheduler):
        self.gate.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_expert_match = 0
        
        total_weights_sum = torch.zeros(3, device=self.device)

        for batch_idx, (images, labels) in enumerate(gate_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.no_grad():
                logits_list, embeddings = self.model(images)

            probs = self.get_probs(logits_list, T)

            gate_logits = self.gate(embeddings)
            weights = F.softmax(gate_logits, dim=1)
            B = labels.size(0)

            true_probs_experts = torch.stack([p[torch.arange(B), labels] for p in probs], dim=1)
            target_expert = torch.argmax(true_probs_experts, dim=1)

            mix_prob_full = torch.zeros_like(probs[0])
            for i in range(3):
                mix_prob_full += weights[:, i:i+1] * probs[i]

            loss = F.nll_loss(torch.log(mix_prob_full + 1e-8), labels)

            optimizer.zero_grad()
            loss.backward()
            
            if batch_idx == 0 and ((epoch + 1) % 10 == 0 or epoch == 0):
                self.logger.info("\n" + "="*80)
                self.logger.info(f"🔍 DIAGNOSTIC LOG: EPOCH {epoch+1} | BATCH 0")
                self.logger.info("="*80)
                
                self.logger.info(f"Loss Components -> CE Loss: {loss.item():.4f}")
                
                emb_mean = embeddings.mean().item()
                emb_std = embeddings.std().item()
                self.logger.info(f"Input Embeddings -> Mean: {emb_mean:.4f} | Std: {emb_std:.4f}")
                
                logits_std = gate_logits.std(dim=0).mean().item()
                weights_std = weights.std(dim=0).mean().item()
                avg_weights = weights.mean(dim=0).tolist()
                self.logger.info(f"Gate Logits (pre-softmax) -> Avg Std across batch: {logits_std:.6f}")
                self.logger.info(f"Weights (post-softmax) -> Avg Std across batch: {weights_std:.6f} | Mean: {[f'{w:.4f}' for w in avg_weights]}")
                
                target_dist = torch.zeros(3, device=self.device)
                for i in range(3):
                    target_dist[i] = (target_expert == i).float().mean()
                self.logger.info(f"Target Expert Distribution: CE={target_dist[0]:.3f} | LA={target_dist[1]:.3f} | BS={target_dist[2]:.3f}")
                
                grad_norm_fc = self.gate.fc.weight.grad.norm().item() if self.gate.fc.weight.grad is not None else 0.0
                self.logger.info(f"Gradient Norms -> FC (Linear Router): {grad_norm_fc:.6f}")
                self.logger.info("="*80 + "\n")

            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_weights_sum += weights.sum(dim=0)

            gate_preds = torch.argmax(gate_logits, dim=1)
            total_expert_match += gate_preds.eq(target_expert).sum().item()
            
            _, pred = mix_prob_full.max(dim=1)
            total_correct += pred.eq(labels).sum().item()
            total_samples += images.size(0)

        scheduler.step()
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples * 100
        avg_expert_match = total_expert_match / total_samples * 100
        
        epoch_avg_weights = total_weights_sum / total_samples
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            w_ce, w_la, w_bs = epoch_avg_weights[0].item(), epoch_avg_weights[1].item(), epoch_avg_weights[2].item()
            log_line = f"  Epoch {epoch+1} Train Routing -> Avg Weights: CE={w_ce:.3f}, LA={w_la:.3f}, BS={w_bs:.3f} | Gate Acc: {avg_expert_match:.2f}%"
            print(log_line)
            self.logger.info(log_line)
            
        return avg_loss, avg_expert_match

    def do_train_val(self):
        self.log_feature_statistics()
        
        batch_sizes = getattr(self.cfg, 'gate_batch_sizes', [128])
        temperatures = getattr(self.cfg, 'gate_temperatures', [1.0])
        
        self.logger.info(f"\n[INFO] Starting Gate Sweep. Batch Sizes: {batch_sizes}, Temperatures: {temperatures}")
        
        sweep_results = []

        for bs in batch_sizes:
            gate_loader = DataLoader(
                self.gate_dataset, batch_size=bs,
                sampler=self.gate_sampler, num_workers=self.cfg.workers, pin_memory=True
            )

            for T in temperatures:
                print("\n" + "#"*80)
                print(f"# SWEEPING GATE: Batch Size = {bs}, Temperature = {T}")
                print("#"*80)
                self.logger.info(f"SWEEP: Batch Size={bs}, Temp={T}")

                self._reset_gate_and_optimizer()
                
                best_val_acc = 0.0
                best_epoch = -1

                for epoch in range(self.gate_epochs):
                    train_loss, train_gate_acc = self.train_one_epoch(epoch, T, gate_loader, self.optimizer, self.scheduler)
                    
                    # Evaluate validation mixture accuracy every epoch
                    val_mixture_acc, val_oracle_match = self.validate(T)
                    
                    print(f"  Epoch {epoch+1}/{self.gate_epochs}: train_loss={train_loss:.4f}, train_gate_acc={train_gate_acc:.2f}%, val_mixture_acc={val_mixture_acc:.2f}%, oracle_match={val_oracle_match:.2f}%")

                    if val_mixture_acc > best_val_acc:
                        best_val_acc = val_mixture_acc
                        best_epoch = epoch
                        self.save_gate_checkpoint(epoch, bs, T, val_mixture_acc, is_best=True)
                        
                sweep_results.append({
                    'batch_size': bs, 'temp': T, 'best_epoch': best_epoch + 1, 'best_val_acc': best_val_acc
                })
                print(f"[INFO] Finished BS={bs}, T={T}. Best Epoch: {best_epoch+1} with Val Mixture Acc: {best_val_acc:.4f}")

        print("\n" + "="*100)
        print("GATE SWEEP FINAL SUMMARY")
        print("="*100)
        print(f"{'BS':<5} | {'T':<5} | {'Best Epoch':<10} | {'Best Val Mixture Acc':<20}")
        print("-"*50)
        for r in sweep_results:
            print(f"{r['batch_size']:<5} | {r['temp']:<5} | {r['best_epoch']:<10} | {r['best_val_acc']:<20.4f}")
        print("="*100)
        
        self.eval_best_model()

    def save_gate_checkpoint(self, epoch, bs, T, val_acc, is_best=False):
        os.makedirs(self.cfg.root_model, exist_ok=True)
        path = os.path.join(self.cfg.root_model, f"gate_checkpoint_bs{bs}_T{T}_epoch{epoch}.pth")
        state = {
            'epoch': epoch,
            'batch_size': bs,
            'temperature': T,
            'gate_state_dict': self.gate.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'val_acc': val_acc
        }
        if is_best:
            torch.save(state, path)
            self.logger.info(f"New Best Val Mixture Acc found ({val_acc:.2f}%)! Saved checkpoint: {path}")

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
            logits_list, embeddings = self.model(images)
            probs = self.get_probs(logits_list, T)

            gate_logits = self.gate(embeddings)
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
            
        # FIX: Added weights_only=False for PyTorch 2.6+ compatibility
        ckpt = torch.load(best_gate_path, map_location='cpu', weights_only=False)
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