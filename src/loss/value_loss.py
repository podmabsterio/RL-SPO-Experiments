import torch


class MSEValueLoss:
    def __init__(self, clip_value_loss: bool = False, epsilon: float = 0.2):
        self.clip_value_loss = clip_value_loss
        self.epsilon = epsilon

    def __call__(self, values, returns, old_values=None):
        values = values.view(-1)
        returns = returns.view(-1)

        if not self.clip_value_loss:
            return 0.5 * torch.mean((values - returns) ** 2)

        if old_values is None:
            raise ValueError("old_values must be provided when clip_value_loss=True")

        old_values = old_values.view(-1)

        value_loss_unclipped = (values - returns) ** 2
        value_clipped = old_values + torch.clamp(
            values - old_values,
            -self.epsilon,
            self.epsilon,
        )
        value_loss_clipped = (value_clipped - returns) ** 2
        value_loss = 0.5 * torch.mean(torch.max(value_loss_unclipped, value_loss_clipped))

        return value_loss