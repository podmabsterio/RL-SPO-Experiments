import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def act(self, obs):
        dist = self.model.actor(obs)
        actions = dist.sample()
        log_prob = dist.log_prob(actions)
        value = self.model.critic(obs)

        return {
            "actions": actions,
            "log_prob": log_prob,
            "value": value.squeeze(),
        }

    def evaluate_actions(self, obs, actions):
        dist = self.model.actor(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.model.critic(obs)

        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value.squeeze(),
        }

    def get_value(self, obs):
        return self.model.critic(obs).squeeze()
