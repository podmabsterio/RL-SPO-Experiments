# SPO Experiments

This repository contains code for reproduction of some experiments from the paper [Simple Policy Optimization](https://arxiv.org/abs/2401.16025). It was created as a task for a Reinforcement Learning course at HSE University.

In our personal opinion, the original code provided by the authors was not very well structured. This repository has the same functionality but with a better interface, since we utilize `hydra` as a configuration system. Also, the code organization is more readable and does not contain large monolithic files like in the authors' repository. Additionally, it follows the DRY principle better.

## Example usage:

* Train a 3-layer MLP on Hopper-v4 for 10,000,000 iterations on CPU with the PPO objective:

```bash
python main.py -cn=mujoco.yaml run_name="ppo_mlp3" device="cpu" env.env_id="Hopper-v4" env.num_envs=8 policy_loss=ppo trainer.total_timesteps=10000000 model=mlp3 
```

* Train ResNet-18 on Assault-v5 for 5,000,000 iterations on GPU with the SPO objective:

```bash
python main.py -cn=atari.yaml run_name="spo_resnet" device="cuda" env.env_id="ALE/Assault-v5" env.num_envs=8 policy_loss=spo trainer.total_timesteps=5000000
```

## Logs

All our logs can be found in the following CometML project: [Link](https://www.comet.com/podmabsterio/rl-experiments/view/new/panels).

They are grouped for more convenient reading in the following report: [Report](https://www.comet.com/podmabsterio/rl-experiments/reports/6yeNs8YdyHpoLZnHcombR7fkM)

