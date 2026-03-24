import gymnasium as gym
import numpy as np
import torch


class RolloutBuffer:
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        observation_space,
        action_space,
        device="cpu",
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = torch.device(device)

        if not isinstance(observation_space, gym.spaces.Box):
            raise NotImplementedError(
                f"Only Box observation spaces are supported for now, got {type(observation_space)}"
            )

        self.obs_shape = observation_space.shape

        if isinstance(action_space, gym.spaces.Discrete):
            self.is_discrete = True
            self.action_shape = ()
            self.action_dtype = torch.long
        elif isinstance(action_space, gym.spaces.Box):
            self.is_discrete = False
            self.action_shape = action_space.shape
            self.action_dtype = torch.float32
        else:
            raise NotImplementedError(
                f"Unsupported action space type: {type(action_space)}"
            )

        self._allocate_storage()
        self.reset()

    def _allocate_storage(self):
        self.obs = torch.zeros(
            (self.num_steps, self.num_envs, *self.obs_shape),
            device=self.device,
        )

        if self.is_discrete:
            self.actions = torch.zeros(
                (self.num_steps, self.num_envs),
                dtype=self.action_dtype,
                device=self.device,
            )
        else:
            self.actions = torch.zeros(
                (self.num_steps, self.num_envs, *self.action_shape),
                dtype=self.action_dtype,
                device=self.device,
            )

        self.rewards = torch.zeros(
            (self.num_steps, self.num_envs),
            device=self.device,
        )
        self.dones = torch.zeros(
            (self.num_steps, self.num_envs),
            device=self.device,
        )
        self.values = torch.zeros(
            (self.num_steps, self.num_envs),
            device=self.device,
        )
        self.log_probs = torch.zeros(
            (self.num_steps, self.num_envs),
            device=self.device,
        )

        self.advantages = torch.zeros(
            (self.num_steps, self.num_envs),
            device=self.device,
        )
        self.returns = torch.zeros(
            (self.num_steps, self.num_envs),
            device=self.device,
        )

    def reset(self):
        self.pos = 0
        self.full = False
        self._last_dones = torch.zeros(
            self.num_envs,
            device=self.device,
        )

    def add(
        self,
        obs,
        actions,
        rewards,
        dones,
        values,
        log_probs,
    ):
        if self.pos >= self.num_steps:
            raise RuntimeError("RolloutBuffer overflow: add() called too many times.")

        self.obs[self.pos].copy_(obs)
        self.actions[self.pos].copy_(actions)
        self.rewards[self.pos].copy_(rewards)
        self.dones[self.pos].copy_(dones)
        self.values[self.pos].copy_(values)
        self.log_probs[self.pos].copy_(log_probs)

        self._last_dones = dones
        self.pos += 1

        if self.pos == self.num_steps:
            self.full = True

    def last_dones(self):
        return self._last_dones

    def compute_advantages(
        self,
        last_value: torch.Tensor,
        last_done: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ):
        if not self.full:
            raise RuntimeError("Cannot compute advantages before buffer is full.")

        last_gae = torch.zeros(self.num_envs, device=self.device)

        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_value = last_value
                next_non_terminal = 1.0 - last_done
            else:
                next_value = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]

            delta = (
                self.rewards[step]
                + gamma * next_value * next_non_terminal
                - self.values[step]
            )

            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae

        self.returns = self.advantages + self.values

    def get_all(self):
        if not self.full:
            raise RuntimeError("Cannot read from buffer before it is full.")

        batch = {
            "obs": self._flatten_time_env(self.obs),
            "actions": self._flatten_time_env(self.actions),
            "old_log_prob": self._flatten_time_env(self.log_probs),
            "old_value": self._flatten_time_env(self.values),
            "advantages": self._flatten_time_env(self.advantages),
            "returns": self._flatten_time_env(self.returns),
        }
        return batch

    def iter_minibatches(self, num_minibatches: int, shuffle=True):
        if not self.full:
            raise RuntimeError("Cannot iterate minibatches before buffer is full.")

        batch = self.get_all()
        batch_size = self.num_steps * self.num_envs

        if batch_size % num_minibatches != 0:
            raise ValueError(
                f"Batch size {batch_size} must be divisible by num_minibatches {num_minibatches}."
            )

        minibatch_size = batch_size // num_minibatches

        if shuffle:
            indices = torch.randperm(batch_size, device=self.device)
        else:
            indices = torch.arange(batch_size, device=self.device)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_idx = indices[start:end]

            yield {
                "obs": batch["obs"][mb_idx],
                "actions": batch["actions"][mb_idx],
                "old_log_prob": batch["old_log_prob"][mb_idx],
                "old_value": batch["old_value"][mb_idx],
                "advantages": batch["advantages"][mb_idx],
                "returns": batch["returns"][mb_idx],
            }

    def _flatten_time_env(self, x):
        return x.reshape(self.num_steps * self.num_envs, *x.shape[2:])
