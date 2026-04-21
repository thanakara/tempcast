import tensorflow as tf

from omegaconf import DictConfig


class SimpleRNNForecaster:
    def __init__(
        self, arch: DictConfig, training: DictConfig, series: DictConfig, **kwargs
    ):
        super().__init__(**kwargs)

        self.arch = arch
        self.train_cfg = training
        self.series = series

        # univar: 1
        # mulvar: n
        n_features = len(series.features) if series.is_mulvar else 1

        inputs = tf.keras.layers.Input(shape=[None, n_features])

        X = tf.keras.layers.SimpleRNN(
            units=arch.units,
            activation=arch.activation,
            dropout=arch.dropout,
            return_sequences=False,
            name=arch.name,
        )(inputs)

        outputs = tf.keras.layers.Dense(units=series.steps_ahead, name="forecast")(X)

        # store
        self._model = tf.keras.Model(inputs=inputs, outputs=outputs, name=arch.name)

    @property
    def keras_model(self) -> tf.keras.Model:
        return self._model

    def build_optimizer(self):
        optimizers = {
            "adam": tf.keras.optimizers.Adam,
            "sgd": tf.keras.optimizers.SGD,
            "rmsprop": tf.keras.optimizers.RMSprop,
        }
        cls = optimizers.get(self.train_cfg.optimizer)
        if cls is None:
            raise ValueError(
                f"Unknown optimizer: {self.train_cfg.optimizer}. "
                f"Choose from: {list(optimizers.keys())}"
            )
        return cls(learning_rate=self.train_cfg.lr)

    def build_callbacks(self, extra: list = None):
        if extra is None:
            extra = []
        callbacks = []

        if self.train_cfg.early_stopping:
            es = self.train_cfg.early_stopping
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=es.monitor,
                    patience=es.patience,
                    min_delta=es.min_delta,
                    restore_best_weights=True,
                    verbose=1,
                )
            )

        return callbacks + extra
