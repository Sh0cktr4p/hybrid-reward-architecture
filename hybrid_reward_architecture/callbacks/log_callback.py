from typing import List, Optional

from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.utils import safe_mean


class LogCallback(WandbCallback):
    def __init__(
        self,
        info_keywords: Optional[List[str]] = None,
        model_save_path: Optional[str] = None,
        model_save_freq: int = 100_000,
        gradient_save_freq: int = 100,
        verbose: int = 2,
    ):
        super().__init__(
            verbose=verbose,
            model_save_path=model_save_path,
            model_save_freq=model_save_freq,
            gradient_save_freq=gradient_save_freq,
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
                if self.episode_counter % 4 == 0:
                    for key in self._info_buffer:
                        self.logger.record(
                            "rollout/{}".format(key), safe_mean(self._info_buffer[key])
                        )
                        self._info_buffer[key] = []
                if self.episode_counter % self.model_save_freq == 0:
                    self.model.save(f"{self.model_save_path}/model_{self.episode_counter}")

                    if hasattr(self.model, "save_replay_buffer"):
                        self.model.save_replay_buffer(f"{self.model_save_path}/replay_buffer")

        return True

    def _on_training_end(self) -> None:
        self.model.save(f"{self.model_save_path}/model_final")

        if hasattr(self.model, "save_replay_buffer"):
            self.model.save_replay_buffer(f"{self.model_save_path}/replay_buffer")
