import tensorflow as tf

from omegaconf import DictConfig
from hydra.utils import instantiate
from wandb.integration.keras import WandbMetricsLogger

import wandb

from tempcast.core import state
from tempcast.utils import get_checkpoint_dir, reconstruct_job_id
from tempcast.callbacks import CustomCallback, EpochTrackerCallback
from tempcast.models.base import BaseForecaster


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model: BaseForecaster = instantiate(cfg.model, series=cfg.series)

    def _build_callbacks(self, checkpoint_path: str, epoch_path: str) -> list:
        extra: list[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=False,
                save_weights_only=True,
                verbose=0,
            ),
            EpochTrackerCallback(epoch_path=epoch_path),
        ]

        if not self.cfg.verbose:
            extra.append(CustomCallback(cfg=self.cfg))

        if self.cfg.wandb.mode != "disabled":
            extra.append(WandbMetricsLogger(log_freq="epoch"))
        return self.model.build_callbacks(extra=extra)

    def fit_and_evaluate(
        self,
        train_ds: tf.data.Dataset,
        valid_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
        train_steps: int,
        valid_steps: int,
        test_steps: int,
    ) -> tf.keras.callbacks.History:
        tf.keras.utils.set_random_seed(self.cfg.seed)

        keras_model = self.model.keras_model
        keras_model.compile(
            optimizer=self.model.build_optimizer(),
            loss="mse",
            metrics=["mae"],
        )

        if self.cfg.resume and self.cfg.run_id:
            job_from_run_id = reconstruct_job_id(self.cfg, self.cfg.run_id)
            checkpoint_dir = get_checkpoint_dir(self.cfg, job_from_run_id)
            keras_model.load_weights(checkpoint_dir / "model.weights.h5")

        else:
            checkpoint_dir = get_checkpoint_dir(self.cfg, state.get_job_id())

        checkpoint_path = str(checkpoint_dir / "model.weights.h5")
        epoch_path = checkpoint_dir / "latest_epoch"

        initial_epoch = 0
        if self.cfg.resume and self.cfg.run_id and epoch_path.exists():
            initial_epoch = int(epoch_path.read_text().strip())

        total_epochs = initial_epoch + self.cfg.model.training.epochs

        history = keras_model.fit(
            train_ds,
            validation_data=valid_ds,
            initial_epoch=initial_epoch,
            epochs=total_epochs,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            callbacks=self._build_callbacks(checkpoint_path, str(epoch_path)),
            verbose=1 if self.cfg.verbose else 0,
        )

        results = keras_model.evaluate(
            test_ds, steps=test_steps, verbose=0, return_dict=True
        )
        if self.cfg.wandb.mode != "disabled":
            wandb.log({f"test_{k}": v for k, v in results.items()})
        return history
