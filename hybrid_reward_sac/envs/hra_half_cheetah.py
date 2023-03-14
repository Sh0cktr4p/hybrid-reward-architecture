import numpy as np

from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv


class HRAHalfCheetah(HalfCheetahEnv):
    def step(self, action):
        observation, reward, done, info = super().step(action)
        info["hra_rew"] = np.array(
            [
                info["reward_run"],
                info["reward_ctrl"],
            ],
            dtype=np.float64,
        )

        return observation, reward, done, info
