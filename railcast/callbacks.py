from typing import override
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from rich.progress import Progress
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback

from railcast import process
from railcast.utils import generate_job_id


class PlotCallback(Callback):
    @override
    def on_job_start(self, config, *, task_function, **kwargs):
        print("__on_job_start__")
        print(f"Job ID: {generate_job_id()}")

    @override
    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs):
        seq_length = config.series.seq_length
        steps_ahead = config.series.steps_ahead
        model: tf.keras.Model = job_return.return_value
        model_arch_name = config.model.arch.name
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
            label="rail_test__ground_truth",
            ax=ax,
        )
        (Y_pred * 1e6).plot(
            marker="x",
            color="r",
            label=f"{model_arch_name}__forecast",
            grid=True,
            ax=ax,
        )
        ax.vlines(start, 0, 1e6, colors="k", linestyles="--")
        ax.set_ylim([200_000, 900_000])
        plt.legend(loc="lower left", fontsize=12)
        plt.show()
        print("__on_job_end__")


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.total_epochs = cfg.model.training.epochs
        self.progress = Progress()
        self.task = None

    @override
    def on_train_begin(self, logs=None):
        print("__on_train_begin__")
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
        print("__on_train_end__")
        print(f"Training Duration: {duration:.5f} seconds.")
