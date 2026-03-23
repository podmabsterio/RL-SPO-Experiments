import torch

class MSEValueLoss():
    def __call__(values, returns):
        return 0.5 * torch.mean((values - returns) ** 2)
