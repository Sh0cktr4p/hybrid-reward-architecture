from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.utils import safe_mean
from hybrid_reward_architecture import MODEL_SAVE_PATH


class LogCallback(WandbCallback):
    def __init__(self, run, info_keywords=None):
        super().__init__(
            gradient_save_freq=100,
            model_save_path=f"{MODEL_SAVE_PATH}/{run.id}",
            verbose=2
        )
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
