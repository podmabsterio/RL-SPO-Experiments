import torch
import torch.nn as nn
from torch.distributions import Categorical


def init_layer(layer, gain=2**0.5, bias_const=0.0):
    nn.init.orthogonal_(layer.weight, gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super().__init__()
        self.conv1 = init_layer(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
        )
        self.conv2 = init_layer(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )
        self.down_sample = down_sample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out = out + identity
        out = self.relu(out)
        return out


def make_layer(in_channels, out_channels, blocks, stride=1):
    down_sample = None
    if stride != 1 or in_channels != out_channels:
        down_sample = nn.Sequential(
            init_layer(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                gain=1.0,
            )
        )

    layers = [BasicBlock(in_channels, out_channels, stride=stride, down_sample=down_sample)]
    for _ in range(1, blocks):
        layers.append(BasicBlock(out_channels, out_channels))

    return nn.Sequential(*layers)


class ResNet18(nn.Module):
    def __init__(self, envs, base_channels=32):
        super().__init__()

        action_dim = envs.single_action_space.n
        in_channels = envs.single_observation_space.shape[0]

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.conv1 = init_layer(
            nn.Conv2d(
                in_channels,
                c1,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        )
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = make_layer(c1, c1, blocks=2, stride=1)
        self.layer2 = make_layer(c1, c2, blocks=2, stride=2)
        self.layer3 = make_layer(c2, c3, blocks=2, stride=2)
        self.layer4 = make_layer(c3, c4, blocks=2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = init_layer(nn.Linear(c4, c4))

        self.actor_head = init_layer(nn.Linear(c4, action_dim), gain=0.01)
        self.critic_head = init_layer(nn.Linear(c4, 1), gain=1.0)

    def _features(self, obs):
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)

        obs = obs.float() / 255.0

        x = self.conv1(obs)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def actor(self, obs):
        x = self._features(obs)
        logits = self.actor_head(x)
        return Categorical(logits=logits)

    def critic(self, obs):
        x = self._features(obs)
        return self.critic_head(x).squeeze(-1)
