import tensorflow as tf

from omegaconf import DictConfig
from hydra.utils import instantiate
from wandb.integration.keras import WandbCallback

import wandb

from railcast.models.protocols import Forecaster


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model: Forecaster = instantiate(cfg.model, series=cfg.series)

    def _build_callbacks(self) -> list:
        extra = []
        if self.cfg.wandb.mode != "disabled":
            extra.append(
                WandbCallback(
                    monitor="val_loss",
                    save_model=False,
                    log_gradients=False,
                )
            )
        return self.model.build_callbacks(extra=extra)

    def fit_and_evaluate(
        self, train_ds, valid_ds, test_ds
    ) -> tuple[tf.keras.callbacks.History, dict]:
        keras_model = self.model.keras_model
        keras_model.compile(
            optimizer=self.model.build_optimizer(),
            loss="mse",
            metrics=["mae"],
        )
        history = keras_model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=self.cfg.model.training.epochs,
            callbacks=self._build_callbacks(),  # add: extra-callbacks
            verbose=1,
        )

        results = keras_model.evaluate(test_ds, verbose=0, return_dict=True)
        if self.cfg.wandb.mode != "disabled":
            wandb.log({f"test_{k}": v for k, v in results.items()})
        return history, results
