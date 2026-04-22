from typing import override

import tensorflow as tf

from railcast.models.base import BaseForecaster


class GRUForecaster(BaseForecaster):
    def __init__(self, arch, training, series):
        super().__init__(arch, training, series)

        n_features = len(series.features) if series.is_mulvar else 1
        inputs = tf.keras.layers.Input(shape=[None, n_features])

        X = inputs
        for i, units in enumerate(arch.units):
            is_last = i == len(arch.units) - 1
            X = tf.keras.layers.GRU(
                units=units,
                dropout=arch.dropout,
                recurrent_dropout=arch.recurrent_dropout,
                return_sequences=not is_last,
                name=f"gru_{i}",
            )(X)

        outputs = tf.keras.layers.Dense(series.steps_ahead, name="forecast")(X)
        self._model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=arch.name)

    @override
    @property
    def keras_model(self):
        return self._model
