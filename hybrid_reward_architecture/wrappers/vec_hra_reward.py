import numpy as np

from stable_baselines3.common.vec_env import VecEnvWrapper


class VecHRAReward(VecEnvWrapper):
    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obss, rews, dones, infos = self.venv.step_wait()
        hra_rews = np.array(
            [
                info["hra_rew"] if "hra_rew" in info else [rews[idx]]
                for idx, info in enumerate(infos)
            ]
        )
        return obss, hra_rews, dones, infos
