import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from imbalanceddl.net.network import build_model

class ExpertEnsemble(nn.Module):
    def __init__(self, cfg, device, ckpt_paths):
        super().__init__()
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
        # Logit-level routing: gate sees the concatenated raw 100-dim logits (300-dim)
        embeddings = torch.cat(logits_list, dim=1)
        return logits_list, embeddings

class GateMLP(nn.Module):
    def __init__(self, input_dim=192, num_experts=3):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc(x)
        return x