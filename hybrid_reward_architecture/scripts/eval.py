from typing import Optional, Type
import argparse

import gym
from gym.wrappers import TimeLimit
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC, TD3

from hybrid_reward_architecture import model_save_path
from hybrid_reward_architecture.algos import HRASAC, HRATD3
from hybrid_reward_architecture.envs import HRAAnt, HRAHumanoid, HRAHalfCheetah


def eval(
    key: str,
    Algorithm: Type[OffPolicyAlgorithm],
    Env: Type[gym.Env],
    n_episodes: int = 10,
):
    env = Env()
    env = TimeLimit(env, max_episode_steps=200)
    model = Algorithm.load(f"{model_save_path}/{key}/model.zip", env=env)

    assert env is not None, "No environment stored with model"

    evaluate_policy(
        model=model,
        env=env,
        n_eval_episodes=n_episodes,
        render=True,
    )


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'key',
        type=str,
    )
    parser.add_argument(
        '--n_episodes',
        type=int,
        default=10,
    )
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

    eval(
        key=args.key,
        Algorithm=get_algorithm_by_str(args.alg),
        Env=get_env_by_str(args.env),
        n_episodes=args.n_episodes,
    )
