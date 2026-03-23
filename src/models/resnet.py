import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchvision.models import resnet18


def init_layer(layer, gain=2**0.5, bias_const=0.0):
    nn.init.orthogonal_(layer.weight, gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class ResNet18(nn.Module):
    def __init__(self, envs):
        super().__init__()

        action_dim = envs.single_action_space.n

        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()

        self.encoder = backbone
        self.actor_head = init_layer(nn.Linear(512, action_dim), gain=0.01)
        self.critic_head = init_layer(nn.Linear(512, 1), gain=1.0)

    def _features(self, obs):
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        obs = obs.float() / 255.0
        return self.encoder(obs)

    def actor(self, obs):
        x = self._features(obs)
        logits = self.actor_head(x)
        return Categorical(logits=logits)

    def critic(self, obs):
        x = self._features(obs)
        return self.critic_head(x).squeeze(-1)