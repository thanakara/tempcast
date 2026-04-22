from typing import override

import tensorflow as tf

from railcast.models.base import BaseForecaster


class ConvLSTMForecaster(BaseForecaster):
    def __init__(self, arch, training, series):
        super().__init__(arch, training, series)

        n_features = len(series.features) if series.is_mulvar else 1
        inputs = tf.keras.layers.Input(shape=[None, n_features])

        # conv1d-block
        X = tf.keras.layers.Conv1D(
            filters=arch.filters,
            kernel_size=arch.kernel_size,
            padding="causal",  # no future-leakage
            activation=arch.activation,
            name="conv1d",
        )(inputs)

        X = tf.keras.layers.MaxPooling1D(
            pool_size=arch.pool_size,
            name="maxpool",
        )(X)

        # lstm-block
        for i, units in enumerate(arch.units):
            is_last = i == len(arch.units) - 1
            X = tf.keras.layers.LSTM(
                units=units,
                dropout=arch.dropout,
                recurrent_dropout=arch.recurrent_dropout,
                return_sequences=not is_last,
                name=f"lstm_{i}",
            )(X)

        outputs = tf.keras.layers.Dense(units=series.steps_ahead, name="forecast")(X)
        self._model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name=arch.name)

    @override
    @property
    def keras_model(self):
        return self._model
