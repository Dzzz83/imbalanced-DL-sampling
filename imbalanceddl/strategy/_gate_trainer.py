import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
from .base import BaseTrainer
from ..utils.debug_logger import get_debug_logger
from ..utils.plugin_rule import define_groups, define_groups_2, compute_aurc_metrics
from ..utils.gate_features import (
    calibrate_expert_probs, calibrate_expert_logits,
    build_gate_input, gate_input_dim, build_mixture, build_oracle_target,
    expert_disagreement, compute_gate_input_dim,
)
from ..net.network import build_model
import glob

# Grids for post-hoc calibration on the tune set (see literature review §7).
# The upper end of GATE_TEMP_GRID extends past 3.0: the 8/25 run fitted
# gate_temp = 3.0 (the old grid edge), i.e. the tune set wanted the gate
# *softer* than the grid allowed — direct evidence of noise-driven routing.
GATE_TEMP_GRID = [0.3, 0.5, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0, 6.0]
MIX_TEMP_GRID = [0.6, 0.8, 1.0, 1.25, 1.5, 2.0]
EXPERT_TEMP_GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


class _IdentityCalibrator:
    """Fallback correctness calibrator when a tune group has too few
    correct samples to fit an isotonic map."""

    def predict(self, conf):
        return np.clip(conf, 0.05, 0.95)

    def __call__(self, conf):
        return self.predict(conf)


class ExpertEnsemble(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.gate_input_mode = getattr(cfg, 'gate_input_mode', 'probability')
        self.experts = nn.ModuleList()
        expert_dir = getattr(cfg, 'expert_ckpt_dir', cfg.root_model)

        ce_bias = getattr(cfg, 'ce_bias', False)
        ce_ls = getattr(cfg, 'ce_ls', 0.0)
        la_bias = getattr(cfg, 'la_bias', False)
        la_ls = getattr(cfg, 'la_ls', 0.0)
        la_tau = getattr(cfg, 'la_tau', 1.5)
        self.la_tau = la_tau
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

        # Gate-input parameters (set after per-expert temperature fitting).
        self.expert_T = [1.0, 1.0, 1.0]
        self.normalize_blocks = True
        self.freq_features = False

    def set_gate_params(self, expert_T=None, normalize_blocks=True,
                        freq_features=None, gate_input_mode=None):
        """Store the calibration/feature params used to build gate inputs."""
        if expert_T is not None:
            self.expert_T = list(expert_T)
        self.normalize_blocks = bool(normalize_blocks)
        if freq_features is not None:
            self.freq_features = bool(freq_features)
        if gate_input_mode is not None:
            self.gate_input_mode = gate_input_mode

    @torch.no_grad()
    def forward(self, x):
        logits_list = []
        hidden_list = []
        for expert in self.experts:
            logits, hidden = expert(x)
            logits_list.append(logits)
            hidden_list.append(hidden)

        if self.gate_input_mode == 'penultimate':
            # Penultimate feature routing (Exp 19): concatenate the three
            # 64-dim per-expert embeddings directly, bypassing the calibrated-
            # probability + statistics pipeline entirely. The ResNet32 backbone
            # features preserve cross-expert diversity (mean pairwise block
            # correlation r ≈ 0.02) that the softmax / L2-normalization step
            # in the probability pipeline destroys (r ≈ 0.68).
            embeddings = torch.cat(hidden_list, dim=1)  # (B, 192)
        else:
            # Default probability-space gate features: bias-adjusted +
            # per-expert temperature-scaled posteriors + confidence/entropy/
            # agreement stats (+ class-frequency features, round-3). T=1.0
            # here: the global sweep temperature only scales the *mixture*;
            # per-expert temperatures (fit on tune) are the calibration knobs.
            probs = calibrate_expert_probs(
                logits_list, self.cfg.cls_num_list, self.la_tau,
                T=1.0, per_expert_T=self.expert_T,
            )
            embeddings = build_gate_input(
                probs, normalize_blocks=self.normalize_blocks,
                cls_num_list=self.cfg.cls_num_list if self.freq_features else None,
            )
        return logits_list, embeddings


class GateMLP(nn.Module):
    """Linear or non-linear router over calibrated probability features.

    The input is ``build_gate_input(probs)``: the three experts' calibrated
    probability distributions + per-expert confidence/margin/entropy +
    pairwise agreement (see ``imbalanceddl.utils.gate_features``).

    When ``linear_router=True`` (recommended for small training sets):
      ``Linear(D, 3)`` — no BatchNorm, no hidden layer.
      Matches the finding from Liu et al. (TMLR 2024) that MLP routers don't
      beat linear routers in vision MoE, and is better matched to the limited
      gate training data (~1,125 samples for CIFAR-100-LT).

    When ``linear_router=False`` (legacy):
      ``BatchNorm1d(D) -> Linear(D, 64) -> ReLU -> Dropout -> Linear(64, 3)``

    ``fc`` keeps the legacy attribute name so
    ``GateTrainer.train_one_epoch`` can still log ``gate.fc.weight.grad``
    when using the non-linear variant (for linear, fc.weight is the same
    as fc_out.weight post-forward — logging is a no-op).
    """

    def __init__(self, input_dim=312, num_experts=3, hidden_dim=64,
                 dropout=0.0, linear_router=False):
        super().__init__()
        self.linear_router = linear_router
        if linear_router:
            # Single linear layer: 316/312-dim features → 3 expert weights.
            # No BN, no ReLU — the softmax is applied outside in the loss.
            self.fc = nn.Linear(input_dim, num_experts)
            self.fc_out = self.fc  # alias for backward compat
        else:
            self.dropout = dropout
            self.bn = nn.BatchNorm1d(input_dim)
            self.fc = nn.Linear(input_dim, hidden_dim)
            self.act = nn.ReLU()
            self.fc_out = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        if self.linear_router:
            return self.fc(x)
        x = self.bn(x)
        x = self.act(self.fc(x))
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc_out(x)


class GateTrainer(BaseTrainer):
    def __init__(self, cfg, dataset, **kwargs):
        self.debug = getattr(cfg, 'debug', False)
        self.debug_logger = get_debug_logger(debug=self.debug)
        print("[INFO] GateTrainer initialization started.")

        super(GateTrainer, self).__init__(cfg, dataset, **kwargs)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

        # --- Stage-2 post-mortem flags (literature_review_moe_routing.md §7) ---
        self.target_mode = getattr(cfg, 'gate_target_mode', 'mix_nll')
        self.mix_space = getattr(cfg, 'mix_space', 'logit')
        self.weight_floor = getattr(cfg, 'gate_weight_floor', 0.0)
        self.norm_blocks = getattr(cfg, 'gate_norm_blocks', True)
        self.fit_expert_temps = getattr(cfg, 'fit_expert_temps', True)
        self.fit_gate_temp = getattr(cfg, 'fit_gate_temp', True)
        self.fit_mix_temp = getattr(cfg, 'fit_mix_temp', True)
        self.tau_oracle = getattr(cfg, 'gate_oracle_tau', 0.2)
        self.k = getattr(cfg, 'routing_sparsity', 2)
        self.la_tau = getattr(cfg, 'la_tau', 1.5)
        # Round-2 fixes: constrain the gate to deviate only with evidence.
        self.kl_uniform = getattr(cfg, 'gate_kl_uniform', 0.0)
        self.disagree_weight = getattr(cfg, 'gate_disagree_weight', False)
        self.gate_dropout = getattr(cfg, 'gate_dropout', 0.0)
        # Round-3 fix: explicit class-frequency features (head/tail signal).
        self.freq_features = getattr(cfg, 'gate_freq_features', False)
        # Exp 19: penultimate feature routing mode (192-dim embeddings).
        self.gate_input_mode = getattr(cfg, 'gate_input_mode', 'probability')
        if self.disagree_weight and self.kl_uniform <= 0.0:
            self.logger.warning(
                "[WARN] gate_disagree_weight=True with gate_kl_uniform=0: on "
                "samples where all experts agree the loss is zero, so the "
                "gate's weights there stay at their random init (accuracy is "
                "unaffected — the prediction is fixed — but NLL/calibration "
                "can suffer). Set gate_kl_uniform > 0 to pin them to uniform."
            )

        self.model = ExpertEnsemble(cfg, self.device).to(self.device)
        self.model.eval()
        self.logger.info("[INFO] Expert ensemble loaded and frozen.")

        self.gate_split_ratio = getattr(cfg, 'gate_split_ratio', 0.9)
        self._split_dataset()

        self.gate = GateMLP(
            input_dim=compute_gate_input_dim(
                self.cfg.num_classes,
                freq_features=self.freq_features,
                gate_input_mode=self.gate_input_mode,
            ),
            num_experts=3,
            dropout=self.gate_dropout,
            # Safety net: penultimate mode MUST use linear router
            # (MLP overparameterizes the 579-param task by 22×).
            linear_router=getattr(cfg, 'gate_linear_router',
                                  self.gate_input_mode == 'penultimate'),
        ).to(self.device)

        self.gate_epochs = cfg.gate_epochs
        self.eval_interval = getattr(cfg, 'eval_interval', 1)
        self.best_gate_acc = 0.0

        # Cache the tune set's raw expert logits ONCE (validation + all
        # calibration fitting reuse them; the gate is re-applied each epoch).
        # IMPORTANT: this caches RAW logits (before temperature scaling), which
        # are independent of expert_T.  The recalibration happens on-the-fly in
        # _tune_calibrated() using the fitted expert_T (set below).  So this
        # MUST run before _fit_expert_temperatures, which needs the cached
        # raw logits to search for optimal per-expert temperatures.
        self._cache_tune_logits()

        # Per-expert temperatures (calibration for mixing, BalPoE-style).
        manual_temps = getattr(cfg, 'expert_temperatures', None)
        if manual_temps is not None:
            self.expert_T = list(manual_temps)
        elif self.fit_expert_temps:
            self.expert_T = self._fit_expert_temperatures()
        else:
            self.expert_T = [1.0, 1.0, 1.0]
        self.model.set_gate_params(self.expert_T, self.norm_blocks,
                                   self.freq_features, self.gate_input_mode)
        self.logger.info(
            f"[INFO] Per-expert temperatures: CE={self.expert_T[0]:.3f} | "
            f"LA={self.expert_T[1]:.3f} | BS={self.expert_T[2]:.3f}"
        )

        # Correctness calibrators (learning-to-defer style targets).
        self.calibrators = None
        if self.target_mode == 'correctness':
            self.calibrators = self._fit_correctness_calibrators()

        self.logger.info("[INFO] GateTrainer initialization complete "
                         f"(target={self.target_mode}, mix_space={self.mix_space}, "
                         f"k={self.k}).")

    # ------------------------------------------------------------------ #
    # Dataset splits (unchanged)                                          #
    # ------------------------------------------------------------------ #
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
            val_indices, test_size=0.5,
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

    # ------------------------------------------------------------------ #
    # Calibration / target construction on the tune set                   #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _cache_tune_logits(self):
        logits = [[], [], []]
        labels = []
        if self.gate_input_mode == 'penultimate':
            tune_emb = []
        for images, lab in self.tune_loader:
            images = images.to(self.device, non_blocking=True)
            logits_list, embeddings = self.model(images)
            for i in range(3):
                logits[i].append(logits_list[i].cpu())
            labels.append(lab.cpu())
            if self.gate_input_mode == 'penultimate':
                tune_emb.append(embeddings.cpu())
        self.tune_logits = [torch.cat(l, dim=0) for l in logits]
        self.tune_labels = torch.cat(labels)
        if self.gate_input_mode == 'penultimate':
            self.tune_embeddings = torch.cat(tune_emb, dim=0)
            self.logger.info(
                f"[INFO] Cached {self.tune_labels.size(0)} tune samples + "
                f"{self.tune_embeddings.size(1)}-dim penultimate embeddings "
                f"for validation."
            )
        self.logger.info(
            f"[INFO] Cached {self.tune_labels.size(0)} tune samples for "
            f"validation and calibration fitting."
        )

    def _tune_calibrated(self, T):
        """Calibrated probs + gate embeddings for the cached tune logits.

        For ``gate_input_mode='penultimate'`` the embeddings are the cached
        penultimate hidden states (no calibration needed); the probability
        distributions are still returned (from calibrated logits) for oracle
        match / target computation.
        """
        probs = calibrate_expert_probs(
            self.tune_logits, self.cfg.cls_num_list, self.la_tau,
            T=T, per_expert_T=self.expert_T,
        )
        if self.gate_input_mode == 'penultimate':
            return probs, self.tune_embeddings
        embeddings = build_gate_input(
            probs, normalize_blocks=self.norm_blocks,
            cls_num_list=self.cfg.cls_num_list if self.freq_features else None,
        )
        return probs, embeddings

    def get_probs(self, logits_list, T):
        return calibrate_expert_probs(
            logits_list, self.cfg.cls_num_list, self.la_tau,
            T, per_expert_T=self.expert_T,
        )

    @torch.no_grad()
    def _fit_expert_temperatures(self, grid=EXPERT_TEMP_GRID):
        """Fit one temperature per expert on the tune set (minimize
        prior-weighted NLL of the calibrated posterior)."""
        labels = self.tune_labels
        N = labels.size(0)
        cls_num_list = np.array(self.cfg.cls_num_list, dtype=np.float64)
        priors = cls_num_list / cls_num_list.sum()
        w = torch.from_numpy(priors[labels.numpy()]).float()
        w = w / w.sum()

        best_temps = []
        for i in range(3):
            best_T, best_nll = 1.0, float('inf')
            for t in grid:
                zcal = calibrate_expert_logits(
                    self.tune_logits, self.cfg.cls_num_list, self.la_tau,
                    T=t, per_expert_T=[1.0, 1.0, 1.0],
                )
                p = F.softmax(zcal[i], dim=1)
                p_y = p[torch.arange(N), labels]
                nll = -float((w * torch.log(p_y.clamp_min(1e-12))).sum())
                if nll < best_nll:
                    best_T, best_nll = t, nll
            best_temps.append(best_T)
        return best_temps

    @torch.no_grad()
    def _fit_correctness_calibrators(self):
        """Fit per-expert maps `max-prob -> P(expert correct)` on the tune set
        (isotonic regression; learning-to-defer style correctness targets)."""
        labels = self.tune_labels
        probs = calibrate_expert_probs(
            self.tune_logits, self.cfg.cls_num_list, self.la_tau,
            T=1.0, per_expert_T=self.expert_T,
        )
        calibrators = []
        for p in probs:
            conf = p.max(dim=1).values.numpy()
            correct = (p.argmax(dim=1) == labels).numpy().astype(np.float64)
            if correct.sum() < 20:
                self.logger.info(
                    f"[WARN] Too few correct tune samples ({correct.sum()}) "
                    f"for isotonic fit; using clipped-confidence fallback."
                )
                calibrators.append(_IdentityCalibrator())
            else:
                iso = IsotonicRegression(out_of_bounds='clip',
                                         y_min=0.02, y_max=0.98)
                iso.fit(conf, correct)
                calibrators.append(iso)
        return calibrators

    def _correctness_target(self, probs):
        """(B, 3) target: normalized per-expert P(correct | max-prob)."""
        confs = torch.stack([p.max(dim=1).values for p in probs], dim=1)
        t = torch.zeros_like(confs)
        for j, cal in enumerate(self.calibrators):
            vals = cal.predict(confs[:, j].cpu().numpy())
            t[:, j] = torch.from_numpy(np.asarray(vals, dtype=np.float32)).to(confs.device)
        return t / t.sum(dim=1, keepdim=True)

    # ------------------------------------------------------------------ #
    # Validation (cached tune set; fits gate & mixture temperatures)      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bal_acc_from_probs(p_mix, labels):
        preds = p_mix.argmax(dim=1).numpy()
        labels = labels.numpy()
        accs = [
            np.mean(preds[labels == c] == c)
            for c in range(p_mix.size(1)) if np.sum(labels == c) > 0
        ]
        return float(np.mean(accs)) * 100 if accs else 0.0

    @torch.no_grad()
    def validate(self, T):
        """Balanced accuracy of the final mixture on the tune set.

        Uses the exact same mixture recipe as test-time evaluation
        (build_mixture with the configured k/space/floor). Optionally fits a
        gate-logit temperature (maximize bal acc) and a final mixture
        temperature (minimize prior-weighted NLL, logit space only).
        """
        self.gate.eval()
        probs, embeddings = self._tune_calibrated(T)
        labels = self.tune_labels

        gate_logits = self.gate(embeddings.to(self.device)).cpu()

        if self.fit_gate_temp:
            best_bal, best_Tg = -1.0, 1.0
            for Tg in GATE_TEMP_GRID:
                weights = F.softmax(gate_logits / Tg, dim=1)
                p_mix = build_mixture(
                    self.tune_logits, weights, self.cfg.cls_num_list,
                    self.la_tau, T=T, per_expert_T=self.expert_T,
                    k=self.k, space=self.mix_space,
                    weight_floor=self.weight_floor,
                )
                bal = self._bal_acc_from_probs(p_mix, labels)
                if bal > best_bal:
                    best_bal, best_Tg = bal, Tg
            gate_temp = best_Tg
            if gate_temp >= GATE_TEMP_GRID[-1]:
                self.logger.warning(
                    "[WARN] Fitted gate_temp at grid edge (softest allowed). "
                    "The tune set prefers the gate as close to uniform as "
                    "possible — routing decisions may be net-negative noise. "
                    "Consider gate_kl_uniform / gate_disagree_weight / k=3."
                )
        else:
            gate_temp = 1.0

        weights = F.softmax(gate_logits / gate_temp, dim=1)
        p_mix = build_mixture(
            self.tune_logits, weights, self.cfg.cls_num_list,
            self.la_tau, T=T, per_expert_T=self.expert_T,
            k=self.k, space=self.mix_space, weight_floor=self.weight_floor,
        )
        bal_acc = self._bal_acc_from_probs(p_mix, labels)

        # Oracle match: does the gate's argmax equal the expert with the
        # highest true-class probability?
        B = labels.size(0)
        true_probs_experts = torch.stack(
            [p[torch.arange(B), labels] for p in probs], dim=1
        )
        target_expert = torch.argmax(true_probs_experts, dim=1)
        gate_choice = torch.argmax(weights, dim=1)
        oracle_match_acc = float((gate_choice == target_expert).float().mean()) * 100

        # Final mixture temperature (calibration only; does not change argmax).
        mix_temp = 1.0
        if self.mix_space == 'logit' and self.fit_mix_temp:
            cls_num_list = np.array(self.cfg.cls_num_list, dtype=np.float64)
            priors = cls_num_list / cls_num_list.sum()
            w = torch.from_numpy(priors[labels.numpy()]).float()
            w = w / w.sum()
            best_nll = float('inf')
            for Tm in MIX_TEMP_GRID:
                p = build_mixture(
                    self.tune_logits, weights, self.cfg.cls_num_list,
                    self.la_tau, T=T, per_expert_T=self.expert_T,
                    k=self.k, space=self.mix_space,
                    weight_floor=self.weight_floor,
                    mix_temperature=Tm,
                )
                p_y = p[torch.arange(B), labels]
                nll = -float((w * torch.log(p_y.clamp_min(1e-12))).sum())
                if nll < best_nll:
                    best_nll, mix_temp = nll, Tm

        return bal_acc, oracle_match_acc, gate_temp, mix_temp

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
        self.logger.info(f"GATE INPUT FEATURE STATISTICS "
                         f"({all_embeddings.size(1)}-dim)")
        self.logger.info("="*80)
        self.logger.info(f"Global Mean: {mean_emb.mean().item():.4f} | Global Std: {std_emb.mean().item():.4f}")
        self.logger.info(f"Min Val: {all_embeddings.min().item():.4f} | Max Val: {all_embeddings.max().item():.4f}")
        self.logger.info("="*80 + "\n")

    # ------------------------------------------------------------------ #
    # Training                                                           #
    # ------------------------------------------------------------------ #
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

            true_probs_experts = torch.stack(
                [p[torch.arange(B), labels] for p in probs], dim=1
            )
            target_expert = torch.argmax(true_probs_experts, dim=1)

            # The mixture is built with the EXACT same recipe used at
            # validation/test time (k, space, weight floor) — the gate is
            # trained on the operation it is scored with (RC2 fix).
            p_mix = build_mixture(
                logits_list, weights, self.cfg.cls_num_list,
                self.la_tau, T=T, per_expert_T=self.expert_T,
                k=self.k, space=self.mix_space,
                weight_floor=self.weight_floor,
            )

            # --- per-sample loss ---
            if self.target_mode == 'mix_nll':
                # Mixture NLL of the final mixture. In logit space the
                # gradient w.r.t. gate logits depends on logit *differences*,
                # so it does not vanish when the mixture is confident or on
                # tail samples (RC4 fix).
                per_sample = F.nll_loss(
                    torch.log(p_mix.clamp_min(1e-12)), labels, reduction='none'
                )
            elif self.target_mode == 'hard_oracle':
                # Direct CE on the argmax-best expert (matches the successful
                # PenultimateRoutingSimulator probe).  Gives a strong,
                # non-zero gradient on EVERY sample, including tail classes
                # where all three experts have tiny probabilities — the
                # argmax still picks one, so the gradient never vanishes.
                loss = F.cross_entropy(gate_logits, target_expert)
                per_sample = None  # not used per-sample below
            elif self.target_mode == 'logprob':
                # Soft-oracle KL with log-space sharpened target (RC1 fix):
                # tail samples get a decisive target even when all p_i(y) are
                # tiny.
                soft_target = build_oracle_target(
                    true_probs_experts, self.tau_oracle, space='logprob'
                )
                log_weights = F.log_softmax(gate_logits, dim=1)
                per_sample = F.kl_div(log_weights, soft_target,
                                      reduction='none').sum(1)
            else:  # correctness
                # L2D-style target: calibrated P(expert correct | max-prob).
                soft_target = self._correctness_target(probs)
                log_weights = F.log_softmax(gate_logits, dim=1)
                per_sample = F.kl_div(log_weights, soft_target,
                                      reduction='none').sum(1)

            # Round-2: route only where routing can matter. When all experts
            # predict the same class, any convex mixture argmaxes to that
            # class — the gate's loss on those samples only shapes noisy
            # weight choices with zero accuracy benefit (verified: the tune
            # set preferred gate_temp=3.0, i.e. as close to uniform as the
            # grid allowed).
            if self.disagree_weight and per_sample is not None:
                disagree = expert_disagreement(probs)
                per_sample = per_sample * disagree.float()
            if per_sample is not None:
                loss = per_sample.mean()

            # Round-2: KL(w || uniform) — deviate from uniform only where the
            # mixture gradient consistently beats the pull. Soft version of
            # RIDE's "default = collective, add experts only when uncertain".
            if self.kl_uniform > 0.0 and self.target_mode != 'hard_oracle':
                kl_u = (
                    weights
                    * (torch.log(weights.clamp_min(1e-12)) + math.log(3))
                ).sum(1).mean()
                loss = loss + self.kl_uniform * kl_u

            optimizer.zero_grad()
            loss.backward()

            if batch_idx == 0 and ((epoch + 1) % 10 == 0 or epoch == 0):
                self.logger.info("\n" + "="*80)
                self.logger.info(f"🔍 DIAGNOSTIC LOG: EPOCH {epoch+1} | BATCH 0")
                self.logger.info("="*80)

                self.logger.info(f"Loss ({self.target_mode}): {loss.item():.4f}")

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

            _, pred = p_mix.max(dim=1)
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

        return avg_loss, avg_expert_match, avg_acc

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
                best_gate_temp = 1.0
                best_mix_temp = 1.0

                for epoch in range(self.gate_epochs):
                    train_loss, train_gate_acc, train_mix_acc = self.train_one_epoch(epoch, T, gate_loader, self.optimizer, self.scheduler)

                    # Evaluate the exact test-time mixture on the tune set.
                    val_mixture_acc, val_oracle_match, gate_temp, mix_temp = self.validate(T)

                    print(f"  Epoch {epoch+1}/{self.gate_epochs}: train_loss={train_loss:.4f}, train_mix_acc={train_mix_acc:.2f}%, val_mixture_acc={val_mixture_acc:.2f}%, oracle_match={val_oracle_match:.2f}%")
                    if (epoch + 1) % 10 == 0 or epoch == 0:
                        gap = train_mix_acc - val_mixture_acc
                        flag = "  <-- overfitting" if gap > 3.0 else ""
                        self.logger.info(f"    train-vs-val mixture acc gap: {gap:+.2f} pp{flag}")

                    if val_mixture_acc > best_val_acc:
                        best_val_acc = val_mixture_acc
                        best_epoch = epoch
                        best_gate_temp = gate_temp
                        best_mix_temp = mix_temp
                        self.save_gate_checkpoint(
                            epoch, bs, T, val_mixture_acc, is_best=True,
                            gate_temp=gate_temp, mix_temp=mix_temp,
                        )

                sweep_results.append({
                    'batch_size': bs, 'temp': T, 'best_epoch': best_epoch + 1,
                    'best_val_acc': best_val_acc,
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

    def save_gate_checkpoint(self, epoch, bs, T, val_acc, is_best=False,
                             gate_temp=1.0, mix_temp=1.0):
        os.makedirs(self.cfg.root_model, exist_ok=True)
        path = os.path.join(self.cfg.root_model, f"gate_checkpoint_bs{bs}_T{T}_epoch{epoch}.pth")
        state = {
            'epoch': epoch,
            'batch_size': bs,
            'temperature': T,
            'gate_state_dict': self.gate.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'val_acc': val_acc,
            # Stage-2 fix metadata: the exact recipe used for this checkpoint.
            'gate_temp': gate_temp,
            'mix_temp': mix_temp,
            'expert_temps': list(self.expert_T),
            'k': self.k,
            'mix_space': self.mix_space,
            'target_mode': self.target_mode,
            'tau': self.tau_oracle,
            'norm_blocks': self.norm_blocks,
            'weight_floor': self.weight_floor,
            # Round-2 fix metadata.
            'kl_uniform': self.kl_uniform,
            'disagree_weight': self.disagree_weight,
            'dropout': self.gate_dropout,
            # Round-3 fix metadata.
            'freq_features': self.freq_features,
            # Exp 19:
            'gate_input_mode': self.gate_input_mode,
            # Linear router flag (architectural — must reconstruct correctly).
            'linear_router': self.gate.linear_router,
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
        # Flat LR: the previous warmup schedule kept LR at 0.2-0.8x for the
        # first 4 epochs, so with early plateau the gate never escaped its
        # (near-uniform) initialization.
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: 1.0
        )

    # ------------------------------------------------------------------ #
    # Final evaluation                                                   #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def extract_posteriors(self, loader, T, gate_temp=1.0, mix_temp=1.0):
        self.gate.eval()
        self.model.eval()
        all_p_mix = []
        all_labels = []
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            logits_list, embeddings = self.model(images)

            gate_logits = self.gate(embeddings) / gate_temp
            weights = F.softmax(gate_logits, dim=1)

            p_mix = build_mixture(
                logits_list, weights, self.cfg.cls_num_list,
                self.la_tau, T=T, per_expert_T=self.expert_T,
                k=self.k, space=self.mix_space,
                weight_floor=self.weight_floor,
                mix_temperature=mix_temp,
            )

            all_p_mix.append(p_mix.cpu().numpy())
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
        ckpt_mode = ckpt.get('gate_input_mode', 'probability')
        ckpt_freq = ckpt.get('freq_features', self.freq_features)
        if ckpt_mode != self.gate_input_mode or ckpt_freq != self.freq_features:
            self.gate_input_mode = ckpt_mode
            self.freq_features = ckpt_freq
            new_dim = compute_gate_input_dim(
                self.cfg.num_classes,
                freq_features=self.freq_features,
                gate_input_mode=self.gate_input_mode,
            )
            self.logger.info(
                f"[INFO] Checkpoint uses gate_input_mode={ckpt_mode} "
                f"(freq_features={ckpt_freq}); rebuilding gate with "
                f"input_dim={new_dim}"
            )
            self.gate = GateMLP(
                input_dim=new_dim,
                num_experts=3, dropout=self.gate_dropout,
                linear_router=ckpt.get('linear_router', False),
            ).to(self.device)
        self.gate.load_state_dict(ckpt['gate_state_dict'])
        T = ckpt.get('temperature', 1.0)
        gate_temp = ckpt.get('gate_temp', 1.0)
        mix_temp = ckpt.get('mix_temp', 1.0)
        self.expert_T = list(ckpt.get('expert_temps', [1.0, 1.0, 1.0]))
        self.k = ckpt.get('k', getattr(self.cfg, 'routing_sparsity', 2))
        self.mix_space = ckpt.get('mix_space', self.mix_space)
        self.weight_floor = ckpt.get('weight_floor', self.weight_floor)
        self.norm_blocks = ckpt.get('norm_blocks', self.norm_blocks)
        self.model.set_gate_params(self.expert_T, self.norm_blocks,
                                   self.freq_features, self.gate_input_mode)
        self.logger.info(
            f"Loaded best gate from {best_gate_path} (Epoch {ckpt['epoch']}) "
            f"with T={T}, gate_temp={gate_temp:.3f}, mix_temp={mix_temp:.3f}, "
            f"expert_temps={[f'{t:.3f}' for t in self.expert_T]}"
        )

        self.logger.info("Extracting posteriors for tune (val) and test sets...")
        p_mix_val, labels_val = self.extract_posteriors(self.tune_loader, T, gate_temp, mix_temp)
        p_mix_test, labels_test = self.extract_posteriors(self.test_loader, T, gate_temp, mix_temp)

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
