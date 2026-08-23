import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from imbalanceddl.net.network import build_model

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
        # Probability-space routing (T=1.0), matching the trainer-side
        # ExpertEnsemble so the gate sees the same input distribution at
        # evaluation time as it saw during training.
        embeddings = self._calibrated_probs(logits_list, T=1.0)
        return logits_list, embeddings

    def _calibrated_probs(self, logits_list, T=1.0):
        """Calibrate expert logits to probs (same math as get_probs)."""
        p_ce = F.softmax(logits_list[0] / T, dim=1)

        cls_num_list = torch.tensor(
            self.cfg.cls_num_list, device=self.device, dtype=torch.float32
        )
        log_prior = torch.log(cls_num_list / cls_num_list.sum() + 1e-12)
        p_la = F.softmax((logits_list[1] + self.la_tau * log_prior) / T, dim=1)

        log_spc = torch.log(cls_num_list + 1e-12)
        p_bs = F.softmax((logits_list[2] + log_spc) / T, dim=1)

        return torch.cat([p_ce, p_la, p_bs], dim=1)

class GateMLP(nn.Module):
    """Non-linear logit router matching the trainer-side architecture.

    BatchNorm1d(300) -> Linear(300, 64) -> ReLU -> Linear(64, 3).
    Attribute names (bn, fc, act, fc_out) match ``_gate_trainer.GateMLP``
    so trained gate state_dicts load unchanged.
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