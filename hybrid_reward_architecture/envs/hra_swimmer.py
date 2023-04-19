import numpy as np
from gym.envs.mujoco.swimmer_v3 import SwimmerEnv


class HRASwimmer(SwimmerEnv):
    n_reward_signals = 3

    def step(self, action):
        observation, reward, done, info = super().step(action)
        info["hra_rew"] = np.array(
            [
                info["reward_fwd"],
                info["reward_ctrl"],
            ],
            dtype=np.float64,
        )

        return observation, reward, done, info
