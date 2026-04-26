from typing import override

import tensorflow as tf

from omegaconf import DictConfig

from tempcast.models.base import BaseForecaster


class SimpleRNNForecaster(BaseForecaster):
    def __init__(self, arch: DictConfig, training: DictConfig, series: DictConfig):
        super().__init__(arch, training, series)

    @override
    def _build_keras_model(self):
        X = tf.keras.layers.SimpleRNN(
            units=self.arch.units,
            activation=self.arch.activation,
            dropout=self.arch.dropout,
            return_sequences=False,
            name=self.arch.name,
        )(self.inputs)

        outputs = tf.keras.layers.Dense(
            units=self.series.steps_ahead,
            name="forecast",
        )(X)

        return tf.keras.Model(
            inputs=[self.inputs],
            outputs=[outputs],
            name=self.arch.name,
        )
