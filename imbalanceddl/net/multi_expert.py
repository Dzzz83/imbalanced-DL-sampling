import torch
import torch.nn as nn

class MultiExpertNetwork(nn.Module):
    def __init__(self, backbone, num_classes, num_experts=3):
        super(MultiExpertNetwork, self).__init__()
        self.backbone = backbone
        
        # Dynamically infer feature dimension from the backbone
        with torch.no_grad():
            dummy_out = self.backbone(torch.randn(1, 3, 32, 32))
            if isinstance(dummy_out, tuple):
                feat_dim = dummy_out[0].shape[-1]
            else:
                feat_dim = dummy_out.shape[-1]
                
        self.experts = nn.ModuleList([
            nn.Linear(feat_dim, num_classes) for _ in range(num_experts)
        ])

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]
        return [expert(features) for expert in self.experts]