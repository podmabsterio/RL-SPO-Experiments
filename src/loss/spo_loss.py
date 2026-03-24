import torch


class SPOLoss:
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, ratio, advantages):
        return -(
            advantages * ratio
            - torch.abs(advantages) * (ratio - 1.0).pow(2) / (2 * self.eps)
        ).mean()
