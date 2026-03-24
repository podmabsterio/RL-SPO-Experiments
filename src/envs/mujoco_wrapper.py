import gymnasium as gym
import numpy as np

class MujocoEnvFactory:
    def __init__(self, num_envs, env_id, gamma):
        self.num_envs = num_envs
        self.env_id = env_id
        self.gamma = gamma

    def _make_env(self):
        def thunk():
            env = gym.make(self.env_id)
            env = gym.wrappers.FlattenObservation(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda o: np.clip(o, -10, 10), None)
            env = gym.wrappers.NormalizeReward(env, gamma=self.gamma)
            env = gym.wrappers.TransformReward(env, lambda r: float(np.clip(r, -10, 10)))
            return env
        return thunk
    
    def make_envs(self):
        return gym.vector.AsyncVectorEnv([self._make_env() for _ in range(self.num_envs)])