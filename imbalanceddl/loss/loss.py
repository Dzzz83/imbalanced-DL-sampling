import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def focal_loss(input_values, gamma):
    p = torch.exp(-input_values)
    loss = (1 - p)**gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(
            F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)

class LogitAdjustedLoss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0, reduction='mean'):
        super(LogitAdjustedLoss, self).__init__()
        cls_num_list = torch.FloatTensor(cls_num_list)
        probs = cls_num_list / cls_num_list.sum()
        self.register_buffer('log_prior', probs.log())
        self.tau = tau
        self.reduction = reduction

    def forward(self, logits, targets):
        # Move buffer to logits device
        log_prior = self.log_prior.to(logits.device)
        adjusted_logits = logits + self.tau * log_prior
        loss = F.cross_entropy(adjusted_logits, targets, reduction=self.reduction)
        return loss

class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list, reduction='mean'):
        super(BalancedSoftmaxLoss, self).__init__()
        cls_num_list = torch.FloatTensor(cls_num_list)
        probs = cls_num_list / cls_num_list.sum()
        self.register_buffer('log_prior', probs.log())
        self.reduction = reduction
        print(f"[INFO] BalancedSoftmaxLoss: log_prior sample: {self.log_prior[:5]}")

    def forward(self, logits, targets):
        # FIX: Move log_prior to the same device as logits
        log_prior = self.log_prior.to(logits.device)
        adjusted_logits = logits + log_prior
        loss = F.cross_entropy(adjusted_logits, targets, reduction=self.reduction)
        return loss