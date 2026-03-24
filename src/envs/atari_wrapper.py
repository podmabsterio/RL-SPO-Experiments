import gymnasium as gym
import ale_py
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

gym.register_envs(ale_py)


class AtariEnvFactory:
    def __init__(self, num_envs, env_id):
        self.num_envs = num_envs
        self.env_id = env_id

    def _make_env(self):
        def thunk():
            env = gym.make(self.env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayscaleObservation(env)
            env = gym.wrappers.FrameStackObservation(env, 4)
            return env

        return thunk

    def make_envs(self):
        return gym.vector.AsyncVectorEnv(
            [self._make_env() for _ in range(self.num_envs)]
        )
