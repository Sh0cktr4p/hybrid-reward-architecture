from typing import Any, Dict, Optional, Tuple, Type, Union
import warnings
import numpy as np
import gym
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.wrappers import TimeLimit
import torch as th
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import ReplayBufferSamples, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import safe_mean

try:
    import psutil
except ImportError:
    psutil = None


class HRAReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        n_reward_signals: int = 1,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        # Call grandparent constructor
        super(ReplayBuffer, self).__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs
        )
        print(f"n_reward_signals: {n_reward_signals}")
        self.n_reward_signals = n_reward_signals

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and
        # handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros(
                (self.buffer_size, self.n_envs) + self.obs_shape,
                dtype=observation_space.dtype,
            )

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)

        self.rewards = np.zeros((self.buffer_size, self.n_envs, self.n_reward_signals), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = (
                self.observations.nbytes + self.actions.nbytes +
                self.rewards.nbytes + self.dones.nbytes
            )

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory "
                    "to store the complete replay buffer "
                    f"{total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def _get_samples(
            self,
            batch_inds: np.ndarray,
            env: None = None
    ) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds)))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default
            # (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, self.n_reward_signals, 1), None),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class HRASAC(SAC):
    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[gym.Env, str],
        n_reward_signals: int = 1,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[HRAReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        assert not isinstance(env.observation_space, gym.spaces.Dict), \
            "Error: HRA SAC does not support Dict observation space."

        if replay_buffer_class is None:
            replay_buffer_class = HRAReplayBuffer

        # Abuse the n_critics parameter
        if policy_kwargs is None:
            policy_kwargs = {}

        if "n_critics" in policy_kwargs:
            policy_kwargs["n_critics"] *= n_reward_signals
        else:
            policy_kwargs["n_critics"] = 2 * n_reward_signals

        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {}

        replay_buffer_kwargs["n_reward_signals"] = n_reward_signals

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        self.n_reward_signals = n_reward_signals

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)

                # Reshape to compute the min per reward signal
                next_q_values = next_q_values.reshape(next_q_values.shape[0], self.n_reward_signals, -1)

                next_q_values, _ = th.min(next_q_values, dim=-1, keepdim=True)
                # add entropy term
                next_q_values -= ent_coef * next_log_prob.reshape(-1, 1, 1) / self.n_reward_signals
                # td error + entropy term
                # Shape: (batch, n_reward_signals, 1)
                target_q_values = (
                    replay_data.rewards +
                    (1 - replay_data.dones) * self.gamma * next_q_values
                )

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = th.cat(self.critic(replay_data.observations, replay_data.actions), dim=1)

            current_q_values = current_q_values.reshape(
                current_q_values.shape[0],
                self.n_reward_signals,
                -1,
            )

            # Compute critic loss
            # critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_loss = 0.5 * th.sum((current_q_values - target_q_values)**2) / current_q_values.shape[0]
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            q_values_pi = q_values_pi.reshape(q_values_pi.shape[0], self.n_reward_signals, -1)
            # Combine q values by summation
            combined_q_values_pi = th.sum(q_values_pi, dim=1)
            # Min over all critic networks
            min_qf_pi, _ = th.min(combined_q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


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


class DummyHRAWrapper(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = super().step(action)
        info["hra_rew"] = np.random.rand(5)
        return obs, rew, done, info


class HRAVecEnvWrapper(VecEnvWrapper):
    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obss, _, dones, infos = self.venv.step_wait()
        hra_rews = np.array([info["hra_rew"] for info in infos])
        return obss, hra_rews, dones, infos


class LogCallback(WandbCallback):
    def __init__(self, info_keywords=None):
        super().__init__(gradient_save_freq=100, model_save_path=f"models/{run.id}", verbose=2)
        if info_keywords is None:
            info_keywords = []

        self.info_keywords = info_keywords
        self.episode_counter = 0
        self._info_buffer = dict()
        for key in info_keywords:
            self._info_buffer[key] = []

    def _on_step(self) -> bool:
        for i in range(len(self.locals["dones"])):
            if self.locals["dones"][i]:
                self.episode_counter += 1
                for key in self.info_keywords:
                    if key in self.locals["infos"][i]:
                        self._info_buffer[key].append(self.locals["infos"][i][key])
                if (self.episode_counter + 1) % 4 == 0:
                    for key in self._info_buffer:
                        self.logger.record(
                            "rollout/{}".format(key), safe_mean(self._info_buffer[key])
                        )
                        self._info_buffer[key] = []

        return super()._on_step()


def train_HRA(run):
    info_keys = ["reward_forward", "reward_ctrl", "reward_contact", "reward_survive"]

    def env_fn():
        env = HRAAnt()
        env = TimeLimit(env, 200)
        env = Monitor(env, info_keywords=info_keys)
        return env

    vec_env = HRAVecEnvWrapper(DummyVecEnv([env_fn for _ in range(1)]))

    model = HRASAC("MlpPolicy", vec_env, verbose=2, n_reward_signals=4, tensorboard_log=f"runs/{run.id}")
    model.learn(total_timesteps=100_000, callback=LogCallback(info_keys))

    vec_env.close()


def train_basic(run):
    info_keys = ["reward_forward", "reward_ctrl", "reward_contact", "reward_survive"]

    def env_fn():
        env = AntEnv()
        env = TimeLimit(env, 200)
        env = Monitor(env, info_keywords=info_keys)
        return env

    # vec_env = DummyVecEnv([env_fn for _ in range(1)])
    env = env_fn()
    model = SAC("MlpPolicy", env, verbose=2, tensorboard_log=f"runs/{run.id}")
    model.learn(
        total_timesteps=100_000,
        callback=LogCallback(info_keys),
    )

    env.close()


if __name__ == "__main__":
    run = wandb.init(project="hra_test_ant", sync_tensorboard=True)
    # callback = WandbCallback()
    train_HRA(run)

    run.finish()
