from typing import Any, Dict, Optional, Union

import numpy as np

from omegaconf import DictConfig, OmegaConf
import wandb
from wandb.sdk.wandb_run import Run

import gym
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from hybrid_reward_architecture import MODEL_SAVE_PATH, RUNS_SAVE_PATH
from hybrid_reward_architecture.algos import HRATD3, TDRewardHRATD3
from hybrid_reward_architecture.callbacks import LogCallback
from hybrid_reward_architecture.wrappers.vec_hra_reward import VecHRAReward

ALGORITHMS = {
    "SAC": SAC,
    "PPO": PPO,
    "TD3": TD3,
    "HRA_TD3": HRATD3,
    "T2D4": TDRewardHRATD3,
}


def get_n_reward_signals(env_id: str) -> int:
    env = gym.make(env_id)
    return env.n_reward_signals if hasattr(env, "n_reward_signals") else 1


def _compose_environment_kwargs(config: DictConfig) -> Dict[str, Any]:
    kwargs = OmegaConf.to_container(config.environment, resolve=True)
    del kwargs["id"]
    del kwargs["info_keys"]

    return kwargs


def create_training_environment(config: DictConfig, evaluation_mode: bool = False) -> VecEnv:
    def env_fn():
        env = gym.make(config.environment.id, **_compose_environment_kwargs(config))
        env = Monitor(env=env, info_keywords=config.environment.info_keys)
        return env

    env = DummyVecEnv([env_fn for _ in range(config.run.n_envs)])
    env.seed(config.run.eval_seed if evaluation_mode else config.run.seed)

    if get_n_reward_signals(config.environment.id) > 1:
        env = VecHRAReward(env)

    return env


def _compose_model_kwargs(
    config: DictConfig,
    env: VecEnv,
    run_id: Optional[str] = None,
    save_logs: bool = False,
) -> Dict[str, Any]:
    if run_id is None:
        run_id = config.run.id

    kwargs = OmegaConf.to_container(config.algorithm, resolve=True)
    del kwargs["id"]

    kwargs["env"] = env

    if save_logs:
        kwargs["tensorboard_log"] = f"{RUNS_SAVE_PATH}/{run_id}"

    if n_reward_signals := get_n_reward_signals(config.environment.id) > 1:
        kwargs["n_reward_signals"] = n_reward_signals

    if "train_freq" in kwargs and isinstance(kwargs["train_freq"], list):
        kwargs["train_freq"] = tuple(kwargs["train_freq"])

    return kwargs


def create_model(
    config: DictConfig,
    env: VecEnv,
    run_id: Optional[str] = None,
    save_logs: bool = False,
) -> BaseAlgorithm:
    model = ALGORITHMS[config.algorithm.id](**_compose_model_kwargs(
        config=config,
        env=env,
        run_id=run_id,
        save_logs=save_logs
    ))

    return model


def load_model(
    config: DictConfig,
    env: VecEnv,
    run_id: Optional[str] = None,
    load_episode: Optional[Union[int, str]] = None,
) -> BaseAlgorithm:
    """Load a model from disk.

    If the model uses an off-policy algorithm, the replay buffer is also loaded.

    Args:
        config (Config): The config object containing information about the model
        env (VecEnv | None): The environment to use. Can be set to `None` for later initialization.
        run_id (str | None): The run id to load the model from.
            Can be set to override the value specified in the training sub-config
        load_episode (int | str | None): The episode to load the model from.
            Can be set to override the value specified in the training sub-config
            There has to exist a model file at `models/{run_id}/model_{load_episode}.zip`.
            Generally, load_episode should be set to a positive integer or `"final"`

    Returns:
        BaseAlgorithm: The loaded model

    Raises:
        ValueError: [load_episode must be a positive integer or 'final']
    """
    if run_id is None:
        run_id = config.run.id

    if load_episode is None:
        load_episode = config.run.load_episode

    if isinstance(load_episode, int) and load_episode < 0:
        raise ValueError("load_episode must be a positive integer or 'final'")

    if config.run.verbose:
        print(f"Loading model from run {run_id} at episode {load_episode}")

    model = ALGORITHMS[config.algorithm.id].load(
        f"{MODEL_SAVE_PATH}/{run_id}/model_{load_episode}.zip",
        env=env,
    )

    model.set_env(env)

    if isinstance(model, OffPolicyAlgorithm):
        model.load_replay_buffer(f"models/{run_id}/replay_buffer.pkl")

    return model


def get_model(
    config: DictConfig,
    env: VecEnv,
    run_id: Optional[str] = None,
    save_logs: bool = False,
) -> BaseAlgorithm:
    """Create a new model or load an existing model from disk,
        depending on whether a load episode is specified in the training sub-config.

    Args:
        config (Config): The config object containing information about the model
        env (VecEnv | None): The environment to use. Can be set to `None` for later initialization.
        run_id (str | None): Can be set to override the run id specified in the training sub-config,
            as this value is usually set to `None` in training runs.
        save_logs (bool): Whether to save logs to tensorboard (or WandB)

    Returns:
        BaseAlgorithm: The model created according to the config for the given environment.
    """
    if config.run.load_episode is None:
        return create_model(config=config, env=env, run_id=run_id, save_logs=save_logs)
    else:
        return load_model(config=config, env=env, run_id=run_id, load_episode=config.run.load_episode)


def init_wandb(config: DictConfig) -> Run:
    """Initialize WandB.

    If a `run_id` is already specified in the training sub-config, this run will be resumed.
    Otherwise, a new run is created.

    Args:
        config (Config): The config containing information about the run

    Returns:
        Run: The WandB run object

    Raises:
        ValueError: [Cannot load a model without a run_id]
    """
    if config.run.id is None and config.run.load_episode is not None:
        raise ValueError("Cannot load a model without a run_id")

    run_id = config.run.id

    if config.run.id is not None and config.run.load_episode is None:
        print("load_episode not specified, will ignore run_id and create a new run")
        run_id = None

    resume = None if config.run.id is None else "must"

    config_dict = OmegaConf.to_container(
        cfg=config,
        resolve=True,
        throw_on_missing=False,
    )

    return wandb.init(
        project=config.wandb_run.project,
        entity=config.wandb_run.entity,
        group=config.wandb_run.group,
        name=config.wandb_run.name,
        tags=config.wandb_run.tags,
        config=config_dict,
        save_code=False,
        monitor_gym=True,
        sync_tensorboard=True,
        id=run_id,
        resume=resume,
    )


def run_debug_training(config: DictConfig) -> BaseAlgorithm:
    """Run a training without storing any data to disk.

    This is useful for debugging purposes to avoid creating a lot of unnecessary files.

    Args:
        config (Config): The config object containing information about the model

    Returns:
        BaseAlgorithm: The trained model
    """
    env = create_training_environment(config=config, evaluation_mode=False)
    model = create_model(config=config, env=env, run_id="~~debug~~", save_logs=False)
    model.learn(
        total_timesteps=config.run.n_steps,
        log_interval=1,
        reset_num_timesteps=False,
    )

    env.close()

    return model


def run_training_tensorboard(config: DictConfig) -> BaseAlgorithm:
    """Run a training and store the logs to tensorboard.

    This avoids using WandB and stores logs only locally. Only stores the final model.

    Args:
        config (Config): The config object containing information about the model

    Returns:
        BaseAlgorithm: The trained model
    """
    run_id = "%05i" % np.random.randint(100_000) if config.run.id is None else config.run.id
    env = create_training_environment(config=config, evaluation_mode=False)
    model = create_model(config=config, env=env, run_id=run_id, save_logs=True)
    model.learn(
        total_timesteps=config.run.n_steps,
        log_interval=1,
        reset_num_timesteps=config.run.load_episode is None,
    )

    model.save(path=f"models/{run_id}/model_final")

    env.close()

    return model


def run_training_wandb(config: DictConfig) -> BaseAlgorithm:
    """Run a training and store the logs to WandB.

    The WandB run is closed at the end of the training.
    Models are saved in regular intervals and at the end of the training.
    The frequency is specified in `config.run.save_freq`.

    Link: https://wandb.ai

    Args:
        config (Config): The config object containing information about the model

    Returns:
        BaseAlgorithm: The trained model
    """
    with init_wandb(config=config) as run:
        env = create_training_environment(config=config, evaluation_mode=False)
        model = get_model(config=config, env=env, run_id=run.id, save_logs=True)
        callback = LogCallback(
            info_keywords=config.run.info_keys,
            model_save_path=f"{MODEL_SAVE_PATH}/{run.id}/model",
            model_save_freq=config.run.save_freq,
            gradient_save_freq=config.run.grad_save_freq,
            verbose=config.run.verbose,
        )

        model.learn(
            total_timesteps=config.run.n_steps,
            log_interval=1,
            reset_num_timesteps=config.run.load_episode is None,
            callback=callback,
        )

        env.close()

        return model


def run_training(config: DictConfig) -> BaseAlgorithm:
    """Run a training according to the config.

    Depending on the run_type specified in the training sub-config, logs are either
        -not stored at all (debug)
        -stored as tensorboard logs (tensorboard)
        -uploaded to WandB (wandb)

    Models are stored either
        -not at all (debug)
        -once, at the end of training (tensorboard)
        -in regular intervals and at the end of training (wandb)

    Args:
        config (Config): The config object containing information about the model

    Returns:
        BaseAlgorithm: The trained model
    """
    if config.run.type == "tensorboard":
        return run_training_tensorboard(config=config)
    elif config.run.type == "wandb":
        return run_training_wandb(config=config)
    else:
        print("Performing debug run without storing to disk")
        return run_debug_training(config=config)
