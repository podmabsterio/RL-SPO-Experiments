import torch

from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.utils import build_experiment, set_seed
from src.models import ActorCritic
from src.trainer import Trainer
from src.buffer import RolloutBuffer

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_path="src/configs", config_name="mujoco")
def main(config):
    set_seed(int(config.seed))

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(config.device)

    envs_factory = instantiate(config.env)
    envs = envs_factory.make_envs()
    env_id = envs_factory.env_id

    experiment, run_name = build_experiment(config, env_id)

    backbone_model = instantiate(config.model, envs=envs)
    model = ActorCritic(model=backbone_model).to(device)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    policy_loss = instantiate(config.policy_loss)
    value_loss = instantiate(config.value_loss)

    buffer = RolloutBuffer(
        num_steps=int(config.trainer.rollout_steps),
        num_envs=int(envs_factory.num_envs),
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=str(config.device),
    )

    trainer = Trainer(
        cfg=config.trainer,
        envs=envs,
        num_envs=envs_factory.num_envs,
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        buffer=buffer,
        logger=experiment,
        policy_loss_fn=policy_loss,
        value_loss_fn=value_loss,
    )

    trainer.train()

    final_model_path = Path(config.output_dir) / f"{run_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": OmegaConf.to_container(config, resolve=True),
        },
        final_model_path,
    )

    experiment.log_asset(str(final_model_path))
    print(f"Saved final model to: {final_model_path}")

    experiment.end()


if __name__ == "__main__":
    main()
