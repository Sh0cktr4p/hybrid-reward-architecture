from typing import List, Optional, Type
import json

import gym
import wandb

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC, TD3

from hybrid_reward_architecture import RUNS_SAVE_PATH, CONFIG_PATH
from hybrid_reward_architecture.wrappers.vec_hra_reward import VecHRAReward
from hybrid_reward_architecture.callbacks import LogCallback
from hybrid_reward_architecture.algos import HRASAC, HRATD3

import hybrid_reward_architecture.envs  # noqa: F401


def get_n_reward_signals(env_id: str) -> int:
    env = gym.make(env_id)
    return env.n_reward_signals if hasattr(env, "n_reward_signals") else 1


def train(
    Algorithm: Type[OffPolicyAlgorithm],
    env_id: str,
    project_name: str,
    info_keys: List[str],
    n_envs: int = 1,
    config: dict = {},
):
    run = wandb.init(
        project=project_name,
        config=config,
        sync_tensorboard=True
    )

    def env_fn():
        env = gym.make(env_id)
        env = Monitor(env=env, info_keywords=info_keys)
        return env

    n_reward_signals = get_n_reward_signals(env_id)

    vec_env = DummyVecEnv([env_fn for _ in range(n_envs)])

    if n_reward_signals > 1:
        vec_env = VecHRAReward(vec_env)

    algorithm_kwargs = dict(
        env=vec_env,
        verbose=1,
        tensorboard_log=f"{RUNS_SAVE_PATH}/{run.id}",
        **config["algorithm"],
    )

    if n_reward_signals > 1:
        algorithm_kwargs["n_reward_signals"] = n_reward_signals

    model = Algorithm(**algorithm_kwargs)

    model.learn(total_timesteps=config["training"]["n_timesteps"], callback=LogCallback(run, info_keywords=info_keys))

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


def get_info_keys_by_env_id(env_str: str) -> list:
    if env_str == "Ant-v3" or env_str == "HRAAnt-v3":
        return [
            "reward_forward",
            "reward_ctrl",
            "reward_contact",
            "reward_survive",
        ]
    elif env_str == "Humanoid-v3" or env_str == "HRAHumanoid-3":
        return [
            "reward_linvel",
            "reward_quadctrl",
            "reward_alive",
            "reward_impact",
        ]
    elif env_str == "HalfCheetah-v3" or env_str == "HRAHalfCheetah":
        return [
            "reward_run",
            "reward_ctrl"
        ]
    else:
        return []


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--project_name', type=str, default='humanoid')
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
    args = parser.parse_args()

    config_file_path = f"{CONFIG_PATH}/{args.config}"
    if not config_file_path.endswith(".json"):
        config_file_path += ".json"
    with open(config_file_path) as f:
        config = json.load(f)

    env_id = config["env"]
    if args.alg.startswith("hra"):
        env_id = "HRA" + env_id

    print(config)

    train(
        Algorithm=get_algorithm_by_str(args.alg),
        env_id=env_id,
        project_name=args.project_name,
        info_keys=get_info_keys_by_env_id(env_id),
        n_envs=args.n_envs,
        config=config,
    )
