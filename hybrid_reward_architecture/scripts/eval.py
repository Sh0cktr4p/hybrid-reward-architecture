from typing import Optional, Type
import argparse

import gym
from gym.wrappers import TimeLimit

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC, TD3

from hybrid_reward_architecture import model_save_path
from hybrid_reward_architecture.algos import HRASAC, HRATD3

import hybrid_reward_architecture.envs  # noqa: F401


def eval(
    key: str,
    Algorithm: Type[OffPolicyAlgorithm],
    env_id: str,
    n_episodes: int = 10,
):
    env = gym.make(env_id)
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
        default='Ant-v3'
    )
    args = parser.parse_args()

    eval(
        key=args.key,
        Algorithm=get_algorithm_by_str(args.alg),
        env_id=args.env,
        n_episodes=args.n_episodes,
    )
