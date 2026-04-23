from typing import override
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
from rich.progress import Progress
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback

import wandb

from railcast import process
from railcast.core import state
from railcast.utils import (
    load_run_id,
    save_run_id,
    generate_job_id,
    get_checkpoint_dir,
    reconstruct_job_id,
)

load_dotenv(override=True)


class WandBCallback(Callback):
    @override
    def on_job_start(self, config: DictConfig, *, task_function, **kwargs) -> None:
        self._job_id = generate_job_id(config.model.arch.name, config.series.is_mulvar)
        state.set_job_id(job_id=self._job_id)
        self._checkpoint_dir = get_checkpoint_dir(config, state.get_job_id())

        if config.wandb.mode != "disabled":
            cfg_masked_copy = OmegaConf.masked_copy(
                config, ["wandb", "series", "model", "seed"]
            )
            if config.resume and config.run_id:
                job_name = reconstruct_job_id(config, config.run_id)
                self._checkpoint_dir = get_checkpoint_dir(config, job_name)

            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            run_id = load_run_id(self._checkpoint_dir) if config.resume else None
            resume_mode = "allow" if config.resume else None

            wandb.login(key=config.wandb.api_key)
            run = wandb.init(
                entity=config.wandb.entity,
                project=config.wandb.project,
                mode=config.wandb.mode,
                name=state.get_job_id() if not config.resume else None,
                id=run_id,
                resume=resume_mode,
                config=OmegaConf.to_object(cfg_masked_copy),
            )

            # save
            save_run_id(run.id, self._checkpoint_dir)

    @override
    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs) -> None:
        seq_length = config.series.seq_length
        steps_ahead = config.series.steps_ahead
        model: tf.keras.Model = job_return.return_value

        series_type = "mulvar" if config.series.is_mulvar else "univar"
        *_, test_series = process._split_series(cfg=config)
        X = test_series.values[np.newaxis, :seq_length]
        y_pred_ahead = model.predict(X, verbose=0)

        start = test_series.index[seq_length]
        Y_pred = pd.Series(
            y_pred_ahead.squeeze(),
            index=pd.date_range(
                start,
                periods=steps_ahead,
                freq="D",
            ),
        )

        rail_series = test_series["rail"] if config.series.is_mulvar else test_series

        _, ax = plt.subplots(figsize=(8, 4))
        (
            rail_series[seq_length - 2 * steps_ahead : seq_length + steps_ahead] * 1e6
        ).plot(
            marker=".",
            label=f"{series_type}_rail_test__ground_truth",
            ax=ax,
        )
        (Y_pred * 1e6).plot(
            marker="x",
            color="r",
            label=f"{self._job_id}__forecast",
            grid=True,
            ax=ax,
        )
        ax.vlines(start, 0, 1e6, colors="k", linestyles="--")
        ax.set_ylim([200_000, 900_000])
        plt.legend(loc="lower left", fontsize=12)

        if config.resume and config.run_id:
            image_stem = reconstruct_job_id(config, config.run_id)
        else:
            image_stem = state.get_job_id()

        plot_path = Path("plots") / f"{image_stem}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path.absolute(), format="png", dpi=150)
        plt.close()

        if config.wandb.mode != "disabled":
            wandb.finish()


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.total_epochs = cfg.model.training.epochs
        self.progress = Progress()
        self.task = None

    @override
    def on_train_begin(self, logs=None):
        self.time_start = datetime.now()
        self.progress.start()
        self.task = self.progress.add_task(
            f"Epoch 0/{self.total_epochs}", total=self.total_epochs
        )

    @override
    def on_epoch_end(self, epoch, logs=None):
        self.progress.update(
            self.task, advance=1, description=f"Epoch {epoch + 1}/{self.total_epochs}"
        )

    @override
    def on_train_end(self, logs=None):
        self.progress.stop()
        self.time_end = datetime.now()
        duration = (self.time_end - self.time_start).total_seconds()
        print(f"Training Duration: {duration:.5f} seconds.")


class EpochTrackerCallback(tf.keras.callbacks.Callback):
    def __init__(self, epoch_path: str):
        super().__init__()
        self.epoch_path = Path(epoch_path)

    @override
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_path.write_text(str(epoch + 1))
