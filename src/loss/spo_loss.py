from typing import Dict

import torch


class SPOLoss():
    def __init__(self, eps):
        self.eps = eps
        
    def __call__(self, new_log_prob, old_log_prob, advantages):
        log_ratio = new_log_prob - old_log_prob
        ratio = torch.exp(log_ratio)

        return -(advantages * ratio - torch.abs(advantages) * torch.pow(ratio - 1, 2) / (2 * self.eps)).mean()
