import numpy as np

from gym.envs.mujoco.hopper_v3 import HopperEnv


class HRAHopper(HopperEnv):
    n_reward_signals = 2

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        done = self.done
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "forward_reward": forward_reward,
            "healthy_reward": healthy_reward,
            "hra_rew": np.array([forward_reward, healthy_reward], dtype=np.float64),
        }

        return observation, reward, done, info