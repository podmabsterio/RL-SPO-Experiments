import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent


def init_layer(layer):
    torch.nn.init.orthogonal_(layer.weight, np.sqrt(2))
    torch.nn.init.constant_(layer.bias, 0.0)
    return layer


def get_n_layer_mlp(in_dim, out_dim, hidden_dims_sequence):
    dims = list(hidden_dims_sequence) + [out_dim]

    module_list = [init_layer(nn.Linear(in_dim, dims[0]))]
    prev_dim = dims[0]
    for dim in dims[1:]:
        module_list.append(nn.Tanh())
        module_list.append(init_layer(nn.Linear(prev_dim, dim)))
        prev_dim = dim

    return nn.Sequential(*module_list)


class NormalActionsMLP(nn.Module):
    def __init__(self, envs, actor_hidden_dims, critic_hidden_dims=None):
        super().__init__()
        if critic_hidden_dims is None:
            critic_hidden_dims = [64, 64]

        state_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.array(envs.single_action_space.shape).prod()
        self.critic = get_n_layer_mlp(state_dim, 1, critic_hidden_dims)

        self.actor_mean = get_n_layer_mlp(state_dim, action_dim, actor_hidden_dims)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

    def _flatten_obs(self, obs):
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return obs.reshape(obs.shape[0], -1)

    def actor(self, obs):
        obs = self._flatten_obs(obs)

        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_log_std).expand_as(mean)

        dist = Independent(Normal(loc=mean, scale=std), 1)
        return dist
