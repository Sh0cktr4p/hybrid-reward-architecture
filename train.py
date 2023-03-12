from typing import List, Type

import gym
from gym.wrappers import TimeLimit
import wandb

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from hybrid_reward_sac.wrappers.vec_hra_reward import VecHRAReward
from hybrid_reward_sac.callbacks import LogCallback

def train(
    Algorithm: Type[OffPolicyAlgorithm],
    Env: Type[gym.Env],
    project_name: str,
    info_keys: List[str],
    n_timesteps: int = 100_000,
    n_reward_signals: int = 1,
    n_envs: int = 1,
):
    run = wandb.init(project=project_name, sync_tensorboard=True)

    def env_fn():
        env = Env()
        env = TimeLimit(env=env, max_steps=200)
        env = Monitor(env=env, info_keywords=info_keys)

    VecEnvCls = VecHRAReward if n_reward_signals == 1 else DummyVecEnv
    vec_env = VecEnvCls([env_fn for _ in range(n_envs)])

    model = Algorithm(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
    ) if n_reward_signals == 1 else Algorithm(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        n_reward_signals=n_reward_signals
    )

    model.learn(total_timesteps=n_timesteps, callback=LogCallback(run, info_keywords=info_keys))

    vec_env.close()

    run.finish()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='default')
    parser.add_argument('--n_timesteps', type=int, default=100_000)
    parser.add_argument('--n_reward_signals', type=int, default=1)
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--env', type=str, default='Ant')
    args = parser.parse_args()

    

    train(
        Algorithm=Algorithm,
        Env=Env,
        project_name=args.project_name,
        info_keys=Env.info_keys,
        n_timesteps=args.n_timesteps,
        n_reward_signals=args.n_reward_signals,
        n_envs=args.n_envs,
    )