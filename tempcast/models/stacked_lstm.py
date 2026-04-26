from typing import override

import tensorflow as tf

from omegaconf import DictConfig

from tempcast.models.base import BaseForecaster


class StackedLSTMForecaster(BaseForecaster):
    def __init__(self, arch: DictConfig, training: DictConfig, series: DictConfig):
        super().__init__(arch, training, series)

    @override
    def _build_keras_model(self):
        X = self.inputs
        for i, units in enumerate(self.arch.units):
            is_last = i == len(self.arch.units) - 1
            X = tf.keras.layers.LSTM(
                units=units,
                dropout=self.arch.dropout,
                recurrent_dropout=self.arch.recurrent_dropout,
                return_sequences=not is_last,
                name=f"lstm_{i}",
            )(X)

        outputs = tf.keras.layers.Dense(
            units=self.series.steps_ahead,
            name="forecast",
        )(X)

        return tf.keras.Model(
            inputs=[self.inputs],
            outputs=[outputs],
            name=self.arch.name,
        )
