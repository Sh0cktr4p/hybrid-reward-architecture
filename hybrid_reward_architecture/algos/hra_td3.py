from typing import Any, Dict, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.td3 import TD3
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.utils import polyak_update

from hybrid_reward_architecture.common import HRAReplayBuffer


class HRATD3(TD3):
    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        n_reward_signals: int = 1,
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[HRAReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        assert not isinstance(env.observation_space, gym.spaces.Dict), \
            "Error: HRA TD3 does not support Dict observation space."

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
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        self.n_reward_signals = n_reward_signals

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)

                # Reshape to compute the min per reward signal
                next_q_values = next_q_values.reshape(next_q_values.shape[0], self.n_reward_signals, -1)

                next_q_values, _ = th.min(next_q_values, dim=-1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = th.cat(self.critic(replay_data.observations, replay_data.actions), dim=1)

            current_q_values = current_q_values.reshape(current_q_values.shape[0], self.n_reward_signals, -1)
            # Compute critic loss
            critic_loss = 0.5 * th.sum((current_q_values - target_q_values)**2) / current_q_values.shape[0]
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                q_values = th.cat(self.critic(replay_data.observations, self.actor(replay_data.observations)), dim=1)
                q_values = q_values.reshape(q_values.shape[0], self.n_reward_signals, -1)
                q_values = th.sum(q_values, dim=1)

                # Take value of first critic
                actor_loss = -q_values[:, 0].mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
