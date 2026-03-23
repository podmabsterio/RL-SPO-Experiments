from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from torch import nn

from src.models import ActorCritic

class Trainer:
    def __init__(
        self,
        cfg,
        envs: Any,
        num_envs,
        model: ActorCritic,
        optimizer: torch.optim.Optimizer,
        scheduler,
        buffer: Any,
        logger: Any,
        policy_loss_fn: Callable[..., Dict[str, torch.Tensor]],
        value_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        self.cfg = cfg
        self.envs = envs
        self.num_envs = num_envs
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.buffer = buffer
        self.logger = logger
        self.policy_loss_fn = policy_loss_fn
        self.value_loss_fn = value_loss_fn

        self.device = next(model.parameters()).device

        self.global_step = 0
        self.update_step = 0

        self.obs = None
        self._episode_returns = np.zeros(num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(num_envs, dtype=np.int32)

    def train(self):
        self.obs, _ = self.envs.reset()
        num_updates = self.cfg.total_timesteps // (self.cfg.rollout_steps * self.num_envs)

        for update_idx in range(1, num_updates + 1):
            self.update_step = update_idx

            rollout_stats = self.collect_rollout()
            self.buffer.compute_advantages(
                last_value=rollout_stats["last_value"],
                last_done=rollout_stats["last_done"],
                gamma=self.cfg.gamma,
                gae_lambda=self.cfg.gae_lambda,
            )

            update_stats = self.update_model()

            log_data = {
                "update": float(update_idx),
                "global_step": float(self.global_step),
                **rollout_stats["metrics"],
                **update_stats,
            }

            if self.scheduler is not None:
                self.scheduler.step()
                log_data["lr"] = float(self.optimizer.param_groups[0]["lr"])

            self.logger.log_metrics(log_data, step=self.global_step)

    @torch.no_grad()
    def collect_rollout(self):
        self.model.eval()
        self.buffer.reset()

        episodic_returns = []
        episodic_lengths = []

        for _ in range(self.cfg.rollout_steps):
            obs_tensor = self._to_tensor(self.obs)

            act_out = self.model.act(obs_tensor)
            actions = act_out["actions"]
            log_prob = act_out["log_prob"]
            value = act_out["value"]

            env_actions = self._actions_to_env(actions)
            next_obs, rewards, terminated, truncated, infos = self.envs.step(env_actions)

            dones = np.logical_or(terminated, truncated)
            self.buffer.add(
                obs=obs_tensor,
                actions=actions,
                rewards=self._to_tensor(rewards, dtype=torch.float32),
                dones=self._to_tensor(dones, dtype=torch.float32),
                values=value,
                log_probs=log_prob,
            )

            self.global_step += self.num_envs

            self._episode_returns += rewards
            self._episode_lengths += 1

            for env_i, done in enumerate(dones):
                if done:
                    episodic_returns.append(float(self._episode_returns[env_i]))
                    episodic_lengths.append(int(self._episode_lengths[env_i]))

                    payload = {
                        "episode/return": float(self._episode_returns[env_i]),
                        "episode/length": int(self._episode_lengths[env_i]),
                    }
                    
                    self.logger.log_metrics(
                        payload,
                        step=self.global_step,
                    )

                    self._episode_returns[env_i] = 0.0
                    self._episode_lengths[env_i] = 0

            self.obs = next_obs

        last_obs_tensor = self._to_tensor(self.obs)
        last_value = self.model.get_value(last_obs_tensor)
        last_done = self.buffer.last_dones()

        metrics = {}
        if len(episodic_returns) > 0:
            metrics["rollout/episodic_return_mean"] = float(np.mean(episodic_returns))
            metrics["rollout/episodic_return_std"] = float(np.std(episodic_returns))
            metrics["rollout/episodic_length_mean"] = float(np.mean(episodic_lengths))

        return {
            "last_value": last_value,
            "last_done": last_done,
            "metrics": metrics,
        }

    def update_model(self):
        self.model.train()

        policy_losses = []
        value_losses = []
        entropy_values = []
        total_losses = []
        ratio_deviations = []
        clip_fractions = []
        grad_norms = []

        all_pred_values = []
        all_target_returns = []

        for _ in range(self.cfg.num_epochs):
            for batch in self.buffer.iter_minibatches(
                num_minibatches=self.cfg.num_minibatches,
                shuffle=True,
            ):
                stats = self._update_minibatch(batch)

                policy_losses.append(stats["policy_loss"])
                value_losses.append(stats["value_loss"])
                entropy_values.append(stats["entropy"])
                total_losses.append(stats["total_loss"])
                ratio_deviations.append(stats["ratio_deviation"])

                if "clip_fraction" in stats:
                    clip_fractions.append(stats["clip_fraction"])

                all_pred_values.append(stats["pred_values"])
                all_target_returns.append(stats["target_returns"])
                grad_norms.append(stats["grad_norm"])

        result = {
            "train/policy_loss": float(np.mean(policy_losses)),
            "train/value_loss": float(np.mean(value_losses)),
            "train/entropy": float(np.mean(entropy_values)),
            "train/total_loss": float(np.mean(total_losses)),
            "train/ratio_deviation": float(np.mean(ratio_deviations)),
            "train/grad_norm": float(np.mean(grad_norms)),
        }

        if len(clip_fractions) > 0:
            result["train/clip_fraction"] = float(np.mean(clip_fractions))

        return result

    def _update_minibatch(self, batch: Dict[str, torch.Tensor]):
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_prob = batch["old_log_prob"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)

        if self.cfg.normalize_advantages:
            advantages = self._normalize_advantages(advantages)

        eval_out = self.model.evaluate_actions(obs, actions)
        new_log_prob = eval_out["log_prob"]
        entropy = eval_out["entropy"]
        values = eval_out["value"]

        log_ratio = new_log_prob - old_log_prob
        ratio = torch.exp(log_ratio)
        
        ratio_deviation = (torch.abs(ratio - 1)).mean()
                
        p_loss = self.policy_loss_fn(
            ratio=ratio,
            advantages=advantages,
        )


        v_loss = self.value_loss_fn(values, returns)
        entropy_mean = entropy.mean()

        total_loss = (
            p_loss
            + self.cfg.value_coef * v_loss
            - self.cfg.entropy_coef * entropy_mean
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        
        grad_norm = self._get_grad_norm()

        stats = {
            "policy_loss": float(p_loss.detach().cpu().item()),
            "value_loss": float(v_loss.detach().cpu().item()),
            "entropy": float(entropy_mean.detach().cpu().item()),
            "total_loss": float(total_loss.detach().cpu().item()),
            "pred_values": values.detach().cpu().numpy(),
            "target_returns": returns.detach().cpu().numpy(),
            "ratio_deviation": float(ratio_deviation.detach().cpu().item()),
            "grad_norm": float(grad_norm)
        }

        return stats

    def _to_tensor(self, x: Any, dtype: Optional[torch.dtype] = None):
        tensor = torch.as_tensor(x, device=self.device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def _actions_to_env(self, actions: torch.Tensor):
        return actions.detach().cpu().numpy()
    
    def _get_grad_norm(self):
        total_norm = torch.norm(
            torch.stack([
                p.grad.norm(2)
                for p in self.model.parameters()
                if p.grad is not None
            ]),
            2
        )
        return total_norm.item()

    @staticmethod
    def _normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8):
        return (advantages - advantages.mean()) / (advantages.std() + eps)