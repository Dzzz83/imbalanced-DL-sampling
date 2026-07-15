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
        self.register_buffer('log_prior', cls_num_list.log())
        self.tau = tau
        self.reduction = reduction
        print(f"[INFO] LogitAdjustedLoss: tau={tau}, log_prior sample: {self.log_prior[:5]}")

    def forward(self, logits, targets):
        # Paper: z_y - tau * log(pi_y)
        #
        # Example calculation:
        # log_prior[0] (Class 0, 5000 samples)  = log(5000) ≈ 8.51
        # log_prior[99] (Class 99, 5 samples)   = log(5)    ≈ 1.61
        #
        # If logits[0]  = 10.0 and logits[99] = 2.0, and tau = 1.0:
        # adjusted_logits[0]  = 10.0 - 1.0 * 8.51 = 1.49  (Heavily penalized)
        # adjusted_logits[99] = 2.0  - 1.0 * 1.61 = 0.39  (Slightly reduced)
        #
        adjusted_logits = logits - self.tau * self.log_prior
        
        loss = F.cross_entropy(adjusted_logits, targets, reduction=self.reduction)
        return loss

class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list, reduction='mean'):
        super(BalancedSoftmaxLoss, self).__init__()
        cls_num_list = torch.FloatTensor(cls_num_list)
        self.register_buffer('log_prior', cls_num_list.log())
        self.reduction = reduction
        print(f"[INFO] BalancedSoftmaxLoss: log_prior sample: {self.log_prior[:5]}")

    def forward(self, logits, targets):
        # Paper: sum(exp(z_j) * pi_j) = sum(exp(z_j + log(pi_j)))
        #
        # Example calculation:
        # log_prior[0] (Class 0, 5000 samples)  = log(5000) ≈ 8.51
        # log_prior[99] (Class 99, 5 samples)   = log(5)    ≈ 1.61
        #
        # If logits[0]  = 10.0 and logits[99] = 2.0:
        # adjusted_logits[0]  = 10.0 + 8.51 = 18.51  (Massive boost to Head class)
        # adjusted_logits[99] = 2.0  + 1.61 = 3.61  (Small boost to Tail class)
        #
        # By adding log_prior, the Head class logit is heavily inflated in the Softmax 
        # denominator. This artificially increases Head class probability, forcing the 
        # network to output much larger raw logits for Tail classes to compete.
        #
        adjusted_logits = logits + self.log_prior
        
        loss = F.cross_entropy(adjusted_logits, targets, reduction=self.reduction)
        return loss