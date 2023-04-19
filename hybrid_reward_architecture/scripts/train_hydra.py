import hydra
from omegaconf import DictConfig, OmegaConf

from hybrid_reward_architecture.utils.training_utils import run_training
import hybrid_reward_architecture.envs  # noqa: F401


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config, resolve=True))

    run_training(config)


if __name__ == "__main__":
    main()
