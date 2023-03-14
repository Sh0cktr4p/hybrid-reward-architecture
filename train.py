from typing import List, Optional, Type

import gym
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
from gym.wrappers import TimeLimit
import wandb

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC, TD3

from hybrid_reward_sac.wrappers.vec_hra_reward import VecHRAReward
from hybrid_reward_sac.callbacks import LogCallback
from hybrid_reward_sac.algos import HRASAC, HRATD3
from hybrid_reward_sac.envs import HRAAnt, HRAHumanoid, HRAHalfCheetah


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


def get_algorithm_by_str(algo_str: str) -> Optional[Type[OffPolicyAlgorithm]]:
    if algo_str == "sac":
        return SAC
    elif algo_str == "hrasac":
        return HRASAC
    elif algo_str == "td3":
        return TD3
    elif algo_str == "hratd3":
        return HRATD3
    else:
        return None


def get_env_by_str(env_str: str) -> Optional[Type[gym.Env]]:
    if env_str == "ant":
        return AntEnv
    elif env_str == "hraant":
        return HRAAnt
    elif env_str == "humanoid":
        return HumanoidEnv
    elif env_str == "hrahumanoid":
        return HRAHumanoid
    elif env_str == "halfcheetah":
        return HalfCheetahEnv
    elif env_str == "hrahalfcheetah":
        return HRAHalfCheetah
    else:
        return None


def get_info_keys_by_env_str(env_str: str) -> list:
    if env_str == "ant" or env_str == "hraant":
        return [
            "reward_forward",
            "reward_ctrl",
            "reward_contact",
            "reward_survive",
        ]
    elif env_str == "humanoid" or env_str == "hrahumanoid":
        return [
            "reward_linvel",
            "reward_quadctrl",
            "reward_alive",
            "reward_impact",
        ]
    elif env_str == "halfcheetah" or env_str == "hrahalfcheetah":
        return [
            "reward_run",
            "reward_ctrl"
        ]
    else:
        return []


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='default')
    parser.add_argument('--n_timesteps', type=int, default=100_000)
    parser.add_argument('--n_reward_signals', type=int, default=1)
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument(
        '--alg',
        type=str,
        choices=[
            "sac",
            "hrasac",
            "td3",
            "hratd3",
        ],
        default="td3"
    )
    parser.add_argument(
        '--env',
        type=str,
        choices=[
            "ant",
            "hraant",
            "humanoid",
            "hrahumanoid",
            "halfcheetah",
            "hrahalfcheetah",
        ],
        default='ant'
    )
    args = parser.parse_args()

    train(
        Algorithm=get_algorithm_by_str(args.alg),
        Env=get_env_by_str(args.env),
        project_name=args.project_name,
        info_keys=get_info_keys_by_env_str(args.env),
        n_timesteps=args.n_timesteps,
        n_reward_signals=args.n_reward_signals,
        n_envs=args.n_envs,
    )
