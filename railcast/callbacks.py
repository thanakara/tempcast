import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback

from railcast.process import _split_series


class PlotCallback(Callback):
    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs):
        seq_length = config.series.seq_length
        steps_ahead = config.series.steps_ahead
        model: tf.keras.Model = job_return.return_value
        model_arch_name = config.model.arch.name
        *_, test_series = _split_series(cfg=config)
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
