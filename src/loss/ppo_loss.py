import torch


class PPOLoss:
    def __init__(self, eps: float):
        self.eps = eps

    def __call__(self, ratio: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        unclipped_objective = ratio * advantages
        clipped_ratio = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps)
        clipped_objective = clipped_ratio * advantages
        return -torch.min(unclipped_objective, clipped_objective).mean()
