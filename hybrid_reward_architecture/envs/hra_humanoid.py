import numpy as np

from gym.envs.mujoco.humanoid_v3 import HumanoidEnv


class HRAHumanoid(HumanoidEnv):
    n_reward_signals = 4

    def step(self, action):
        observation, reward, done, info = super().step(action)
        info["hra_rew"] = np.array([
            info["reward_linvel"],
            info["reward_quadctrl"],
            info["reward_alive"],
            info["reward_impact"],
        ], dtype=np.float64)

        return observation, reward, done, info
