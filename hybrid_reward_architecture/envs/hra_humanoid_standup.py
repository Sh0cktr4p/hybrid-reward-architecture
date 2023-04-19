import numpy as np

from gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv


class HRAHumanoidStandup(HumanoidStandupEnv):
    n_reward_signals = 3

    def step(self, action):
        observation, reward, done, info = super().step(action)
        info["hra_rew"] = np.array(
            [
                info["reward_linup"],
                info["reward_quadctrl"],
                info["reward_impact"],
            ],
            dtype=np.float64,
        )

        return observation, reward, done, info
