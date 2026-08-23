import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from imbalanceddl.net.network import build_model
from imbalanceddl.utils.gate_features import (
    calibrate_expert_probs, build_gate_input,
)

class ExpertEnsemble(nn.Module):
    def __init__(self, cfg, device, ckpt_paths):
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
        # Probability-space routing (T=1.0): build the exact same
        # calibrated-probability + confidence/agreement feature vector the
        # trainer-side ExpertEnsemble produces, so the gate is evaluated on
        # the representation it was trained on.
        probs = calibrate_expert_probs(
            logits_list, self.cfg.cls_num_list, self.la_tau, T=1.0
        )
        embeddings = build_gate_input(probs)
        return logits_list, embeddings

class GateMLP(nn.Module):
    """Non-linear router matching the trainer-side architecture.

    BatchNorm1d(D) -> Linear(D, 64) -> ReLU -> Linear(64, 3), where
    D = ``gate_input_dim(num_classes)``. Attribute names (bn, fc, act,
    fc_out) match ``_gate_trainer.GateMLP`` so trained state_dicts load
    unchanged.
    """

    def __init__(self, input_dim=312, num_experts=3, hidden_dim=64):
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