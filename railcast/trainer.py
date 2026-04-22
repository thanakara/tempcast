import tensorflow as tf

from omegaconf import DictConfig
from hydra.utils import instantiate
from wandb.integration.keras import WandbCallback

import wandb

from railcast.callbacks import CustomCallback
from railcast.models.base import BaseForecaster


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model: BaseForecaster = instantiate(cfg.model, series=cfg.series)

    def _build_callbacks(self) -> list:
        extra: tf.keras.callbacks.Callback = []

        if not self.cfg.verbose:
            extra.append(CustomCallback(cfg=self.cfg))

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
        self,
        train_ds: tf.data.Dataset,
        valid_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
        train_steps: int,
        valid_steps: int,
        test_steps: int,
    ) -> tuple[tf.keras.callbacks.History, dict[str, float]]:
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
            steps_per_epoch=train_steps,  # when epoch-ends
            validation_steps=valid_steps,  # when valid-ends
            callbacks=self._build_callbacks(),  # add: extra-callbacks
            verbose=1 if self.cfg.verbose else 0,
        )

        results = keras_model.evaluate(
            test_ds, steps=test_steps, verbose=0, return_dict=True
        )
        if self.cfg.wandb.mode != "disabled":
            wandb.log({f"test_{k}": v for k, v in results.items()})
        return history, results
