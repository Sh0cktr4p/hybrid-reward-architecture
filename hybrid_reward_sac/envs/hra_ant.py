import numpy as np

from gym.envs.mujoco.ant_v3 import AntEnv


class HRAAnt(AntEnv):
    def step(self, action):
        observation, reward, done, info = super().step(action)
        info["hra_rew"] = np.array([
            info["reward_forward"],
            info["reward_ctrl"],
            info["reward_contact"],
            info["reward_survive"],
        ], dtype=np.float64)

        return observation, reward, done, info
