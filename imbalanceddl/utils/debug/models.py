import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from imbalanceddl.net.network import build_model
from imbalanceddl.utils.gate_features import (
    calibrate_expert_probs, build_gate_input, gate_input_dim,
)

class ExpertEnsemble(nn.Module):
    def __init__(self, cfg, device, ckpt_paths, expert_T=None,
                 normalize_blocks=True, freq_features=False):
        super().__init__()
        self.cfg = cfg
        self.device = device
        # la_tau: prefer the config value (as the trainer does); fall back to
        # parsing the LA checkpoint filename (as ultra_debug.py does).
        self.la_tau = getattr(cfg, 'la_tau', None)
        if self.la_tau is None:
            tau_match = re.search(r't([\d\.]+)',
                                  os.path.basename(ckpt_paths['LA']))
            self.la_tau = float(tau_match.group(1)) if tau_match else 1.5
        # Per-expert temperatures + block normalization must match the
        # trainer-side ExpertEnsemble (set from gate-checkpoint metadata).
        self.expert_T = list(expert_T) if expert_T is not None else [1.0, 1.0, 1.0]
        self.normalize_blocks = bool(normalize_blocks)
        self.freq_features = bool(freq_features)
        self.experts = nn.ModuleList()
        for name, path in ckpt_paths.items():
            print(f"[INFO] Loading expert {name} from {path}")
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
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
        # Probability-space routing: build the exact same calibrated-
        # probability + confidence/agreement feature vector the trainer-side
        # ExpertEnsemble produces (per-expert temperatures + block
        # normalization + optional class-frequency features), so the gate is
        # evaluated on the representation it was trained on.
        probs = calibrate_expert_probs(
            logits_list, self.cfg.cls_num_list, self.la_tau,
            T=1.0, per_expert_T=self.expert_T,
        )
        embeddings = build_gate_input(
            probs, normalize_blocks=self.normalize_blocks,
            cls_num_list=self.cfg.cls_num_list if self.freq_features else None,
        )
        return logits_list, embeddings

    @torch.no_grad()
    def forward_with_hidden(self, x):
        """Like forward(), but also returns per-expert penultimate embeddings.

        The ResNet32 backbone outputs 64-dim features after avg_pool (the
        ``hidden`` tensor returned by ``Network(x)``). These are the features
        *before* the classifier head — richer and less correlated across
        experts than the L2-normalized probability vectors.

        Returns
        -------
        logits_list : list of 3 tensors, each (B, C)
            Raw expert logits (same as forward).
        embeddings : (B, D) tensor
            Gate-input features (316-dim calibrated probabilities + stats).
        hidden_list : list of 3 tensors, each (B, 64)
            Per-expert penultimate (pre-classifier) embeddings.
        """
        logits_list = []
        hidden_list = []
        for expert in self.experts:
            logits, hidden = expert(x)
            logits_list.append(logits)
            hidden_list.append(hidden)
        probs = calibrate_expert_probs(
            logits_list, self.cfg.cls_num_list, self.la_tau,
            T=1.0, per_expert_T=self.expert_T,
        )
        embeddings = build_gate_input(
            probs, normalize_blocks=self.normalize_blocks,
            cls_num_list=self.cfg.cls_num_list if self.freq_features else None,
        )
        return logits_list, embeddings, hidden_list

class GateMLP(nn.Module):
    """Linear or non-linear router matching the trainer-side architecture.

    When ``linear_router=True`` (recommended):
      ``Linear(D, 3)`` — no BatchNorm, no hidden layer.
    When ``linear_router=False`` (legacy):
      ``BatchNorm1d(D) -> Linear(D, 64) -> ReLU -> [Dropout] -> Linear(64, 3)``
    where D = ``gate_input_dim(num_classes)``.

    Attribute names (bn, fc, act, fc_out) match ``_gate_trainer.GateMLP`` so
    trained state_dicts load unchanged (dropout has no parameters).

    To reduce the risk of dimension mismatch between the BN layer and the
    actual gate-input feature vector, the caller may supply *either*
    ``input_dim`` (legacy) or ``num_classes`` + ``freq_features`` (preferred).
    When ``num_classes`` is given, ``input_dim`` is computed automatically
    via ``gate_input_dim(num_classes, freq_features=freq_features)``.
    """

    def __init__(self, input_dim=None, num_experts=3, hidden_dim=64,
                 dropout=0.0, num_classes=None, freq_features=False,
                 linear_router=False):
        super().__init__()
        self.linear_router = linear_router
        self.dropout = dropout
        # Compute input_dim from num_classes/freq_features when provided,
        # falling back to the explicit (or default 312) input_dim.
        if num_classes is not None:
            input_dim = gate_input_dim(num_classes,
                                       freq_features=bool(freq_features))
        elif input_dim is None:
            input_dim = 312  # safe default for CIFAR-100, no freq_features
        self._input_dim = input_dim
        if linear_router:
            # Single linear layer — no BN, no ReLU.
            self.fc = nn.Linear(input_dim, num_experts)
            self.fc_out = self.fc
        else:
            self.bn = nn.BatchNorm1d(input_dim)
            self.fc = nn.Linear(input_dim, hidden_dim)
            self.act = nn.ReLU()
            self.fc_out = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        # Runtime dimension guard — catches gate-input mismatches early
        # with a descriptive message instead of PyTorch's terse BN error.
        if x.size(-1) != self._input_dim:
            raise RuntimeError(
                f"GateMLP: expected input with {self._input_dim} features, "
                f"but received {x.size(-1)}. This is usually caused by a "
                f"mismatch between the freq_features setting used to build "
                f"the gate input (via build_gate_input(..., "
                f"cls_num_list=...)) and the freq_features used to "
                f"initialize the GateMLP. Ensure both paths agree."
            )
        if self.linear_router:
            return self.fc(x)
        x = self.bn(x)
        x = self.act(self.fc(x))
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc_out(x)